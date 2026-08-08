"""
FastAPI inference service for the fine-tuned legal compliance model.

Loads the merged model (or DAPT base + LoRA adapter, if ADAPTER_PATH is
set) once at startup and serves it behind a single /generate endpoint.
Matches the FastAPI-backend pattern used across the rest of the
portfolio rather than introducing a new framework just for this project.

Run locally:
    uvicorn inference.app_fastapi:app --host 0.0.0.0 --port 8000

Env vars:
    MODEL_PATH     - path or HF id of the base/merged model (required)
    ADAPTER_PATH   - optional LoRA adapter path, applied on top of MODEL_PATH
    DEVICE         - "cuda", "cpu", or "auto" (default: "auto")
"""
import os
import time
from contextlib import asynccontextmanager
from typing import Optional

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = os.environ.get("MODEL_PATH", "Qwen/Qwen2.5-1.5B-Instruct")
ADAPTER_PATH = os.environ.get("ADAPTER_PATH")
DEVICE = os.environ.get("DEVICE", "auto")

PROMPT_TEMPLATE = (
    "You are a legal compliance auditor. Review the following scenario, "
    "identify any clause violations, and recommend remediation.\n\n"
    "Scenario:\n{scenario}\n\nAudit:"
)

_model_state = {}


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device_map = "auto" if DEVICE == "auto" else None
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map=device_map,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
    if DEVICE not in ("auto", None):
        model = model.to(DEVICE)

    if ADAPTER_PATH:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, ADAPTER_PATH)

    model.eval()
    return tokenizer, model


@asynccontextmanager
async def lifespan(app: FastAPI):
    tokenizer, model = load_model()
    _model_state["tokenizer"] = tokenizer
    _model_state["model"] = model
    yield
    _model_state.clear()


app = FastAPI(
    title="Legal Compliance SLM API",
    description="DAPT + QLoRA fine-tuned Qwen2.5-1.5B for compliance auditing",
    version="1.0.0",
    lifespan=lifespan,
)


class GenerateRequest(BaseModel):
    scenario: str = Field(..., min_length=1, description="The compliance scenario to audit")
    max_new_tokens: int = Field(default=256, ge=1, le=1024)
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)


class GenerateResponse(BaseModel):
    audit: str
    latency_ms: float
    model_path: str
    adapter_path: Optional[str] = None


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "model" in _model_state}


@app.post("/generate", response_model=GenerateResponse)
def generate(request: GenerateRequest):
    if "model" not in _model_state:
        raise HTTPException(status_code=503, detail="Model not loaded")

    tokenizer = _model_state["tokenizer"]
    model = _model_state["model"]

    prompt = PROMPT_TEMPLATE.format(scenario=request.scenario.strip())
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    start = time.perf_counter()
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=request.max_new_tokens,
            do_sample=request.temperature > 0,
            temperature=max(request.temperature, 1e-5),
            pad_token_id=tokenizer.eos_token_id,
        )
    latency_ms = (time.perf_counter() - start) * 1000

    full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    audit = full_text[len(prompt):].strip()

    return GenerateResponse(
        audit=audit,
        latency_ms=round(latency_ms, 1),
        model_path=MODEL_PATH,
        adapter_path=ADAPTER_PATH,
    )
