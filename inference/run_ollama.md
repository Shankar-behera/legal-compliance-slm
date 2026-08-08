# Serving the model locally with Ollama

After `src/export/convert_gguf.py` produces `models/gguf/legal-slm.Q4_K_M.gguf`,
download that file from Drive to your local machine, then:

1. **Create a `Modelfile`** in the same directory as the `.gguf` file:

   ```
   FROM ./legal-slm.Q4_K_M.gguf

   TEMPLATE """You are a legal compliance auditor. Review the following scenario, identify any clause violations, and recommend remediation.

   Scenario:
   {{ .Prompt }}

   Audit:"""

   PARAMETER temperature 0.2
   PARAMETER num_predict 256
   ```

2. **Register the model with Ollama:**

   ```bash
   ollama create legal-compliance-slm -f Modelfile
   ```

3. **Run it:**

   ```bash
   ollama run legal-compliance-slm "A vendor stores customer PII for 7 years with no documented retention basis."
   ```

4. **Or call it from code** (matches the pattern in `src/eval/eval_hallucination.py` if you want to benchmark the quantized GGUF version against the pre-quantization merged model):

   ```python
   import requests

   response = requests.post(
       "http://localhost:11434/api/generate",
       json={"model": "legal-compliance-slm", "prompt": "...", "stream": False},
   )
   print(response.json()["response"])
   ```

Keep `temperature` low (0.1–0.3) for this use case — compliance auditing wants
consistent, low-variance output, not creative generation.
