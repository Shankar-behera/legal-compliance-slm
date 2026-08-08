# Model Card — Legal Compliance SLM

## Model Details

- **Base model:** Qwen2.5-1.5B-Instruct
- **Adaptation method:** Domain-Adaptive Continued Pre-Training (DAPT) on
  regulatory text, followed by QLoRA Supervised Fine-Tuning (SFT) on
  synthetic compliance audit instruction pairs
- **Parameters:** 1.5B base + LoRA adapter (r=16, ~20–40MB)
- **Training hardware:** Single Google Colab T4 (16GB VRAM)
- **License:** Inherits the base model's license (Qwen2.5 license terms);
  the fine-tuning data and adapter weights are this project's own

## Intended Use

- **Primary use case:** A portfolio/demonstration project showing an
  end-to-end DAPT+QLoRA pipeline for domain adaptation under hardware
  constraints, applied to the legal compliance domain.
- **Intended users:** Reviewers of this portfolio (recruiters, engineers),
  and anyone using it as a reference implementation for their own
  DAPT/QLoRA project.
- **Appropriate tasks:** Drafting a *first-pass* compliance audit
  (scenario → possible clause violation → possible remediation) as a
  starting point for a human reviewer.

## Out-of-Scope / Inappropriate Use

- **This is not a substitute for legal advice or a licensed compliance
  professional.** It must not be used as the sole basis for a real
  compliance, regulatory, or legal decision.
- Not validated for jurisdictions or regulatory frameworks beyond what's
  represented in the training corpus (primarily GDPR- and US-Code-adjacent
  text) — outputs for other legal systems are unreliable by default.
- Not intended for production deployment without a human-in-the-loop
  review step and a proper legal/compliance evaluation, given the model's
  scale (1.5B) and the synthetic nature of the SFT data.

## Training Data

- **DAPT corpus:** ~10–20M tokens of publicly available regulatory text
  (see `docs/dataset_stats.md` for the measured count from the actual
  run, not just the target).
- **SFT data:** 3,000–5,000 synthetic instruction pairs in a
  scenario → clause violation → remediation format. Synthetic data means
  the model's compliance "knowledge" is bounded by what was represented
  in that generation process — it has not seen real case law or real
  enforcement outcomes.

## Limitations

- **Hallucinated citations.** Small models fine-tuned on synthetic data
  can fabricate plausible-sounding but incorrect clause/statute
  citations. See `docs/benchmark_results.md` for the measured
  hallucination rate — treat any citation the model produces as
  something to verify against a primary source, not as ground truth.
- **Narrow domain coverage.** The DAPT corpus and SFT pairs cover a
  subset of compliance topics (data protection/privacy-leaning); the
  model will generalize poorly to compliance domains outside that
  (e.g. financial services regulation, employment law) unless retrained
  on that data.
- **Small-model reasoning limits.** At 1.5B parameters, multi-step legal
  reasoning is shallower than larger models — expect it to be more
  reliable at surface-level pattern matching (a clause resembling a
  known violation type) than at novel, multi-clause interactions.
- **No adversarial or red-team evaluation performed.** This has not been
  tested against deliberately misleading or edge-case scenario inputs.

## Ethical Considerations

- **Risk of over-trust.** A model that produces fluent, confident-sounding
  compliance audits can be mistaken for authoritative advice by a
  non-expert user. Any deployment of this model (even the demo apps in
  this repo) should visibly disclose that outputs are unverified and
  require human review — the FastAPI/Gradio demos in this repo are
  labeled accordingly and should stay that way if extended.
- **Data provenance.** DAPT training text should be sourced only from
  data that is legally available for this use (public regulatory text,
  properly licensed corpora) — verify licensing before substituting in
  a different or larger corpus.
- **No personal data.** Training data should not include real individuals'
  PII; synthetic scenarios in this project's SFT set are fictional
  compliance situations, not real case data.

## Evaluation

See `docs/benchmark_results.md` for the base / after-DAPT / after-SFT
comparison (perplexity, exact match, F1, hallucination rate, latency),
produced by `src/eval/benchmark.py` against a fixed benchmark set —
regenerate this after any retraining rather than reusing stale numbers.
