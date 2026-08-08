"""
Gradio demo UI — a thin client over the FastAPI service in
inference/app_fastapi.py, for a shareable one-click demo (e.g. for a
portfolio link or a recruiter to try in-browser) without needing them
to hit the API directly.

Run the FastAPI service first:
    uvicorn inference.app_fastapi:app --host 0.0.0.0 --port 8000

Then:
    python inference/app_gradio.py
    # or: python inference/app_gradio.py --api_url http://localhost:8000
"""
import argparse

import gradio as gr
import requests

DEFAULT_API_URL = "http://localhost:8000"

EXAMPLES = [
    "A vendor retains customer PII for 7 years with no documented retention basis.",
    "A data processor transfers EU customer data to a third-party analytics "
    "provider outside the EEA without a signed Standard Contractual Clause.",
    "A company's incident response plan does not specify a breach notification "
    "timeline to the supervisory authority.",
]


def build_query_fn(api_url: str):
    def query(scenario: str, max_new_tokens: int, temperature: float):
        if not scenario or not scenario.strip():
            return "Enter a scenario to audit."
        try:
            response = requests.post(
                f"{api_url}/generate",
                json={
                    "scenario": scenario,
                    "max_new_tokens": int(max_new_tokens),
                    "temperature": float(temperature),
                },
                timeout=120,
            )
            response.raise_for_status()
            data = response.json()
            return f"{data['audit']}\n\n---\n_latency: {data['latency_ms']} ms_"
        except requests.exceptions.RequestException as e:
            return f"Error calling inference API at {api_url}: {e}"

    return query


def build_demo(api_url: str) -> gr.Blocks:
    with gr.Blocks(title="Legal Compliance SLM") as demo:
        gr.Markdown(
            "# Legal Compliance Auditor\n"
            "DAPT + QLoRA fine-tuned Qwen2.5-1.5B. Describe a compliance "
            "scenario and get a clause-violation + remediation audit."
        )
        with gr.Row():
            with gr.Column():
                scenario_input = gr.Textbox(
                    label="Scenario", lines=5, placeholder="Describe the situation to audit..."
                )
                max_tokens_slider = gr.Slider(
                    minimum=32, maximum=512, value=256, step=32, label="Max new tokens"
                )
                temperature_slider = gr.Slider(
                    minimum=0.0, maximum=1.0, value=0.2, step=0.05, label="Temperature"
                )
                submit_btn = gr.Button("Run Audit", variant="primary")
            with gr.Column():
                output_box = gr.Textbox(label="Audit Result", lines=10)

        gr.Examples(examples=EXAMPLES, inputs=scenario_input)

        submit_btn.click(
            fn=build_query_fn(api_url),
            inputs=[scenario_input, max_tokens_slider, temperature_slider],
            outputs=output_box,
        )

    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api_url", default=DEFAULT_API_URL)
    parser.add_argument("--share", action="store_true", help="Create a public Gradio share link")
    args = parser.parse_args()

    demo = build_demo(args.api_url)
    demo.launch(share=args.share)


if __name__ == "__main__":
    main()
