# Implementation plan

This plan follows the milestones in `BLUEPRINT_v2.md`. Checked items must have
tests or an artifact in the repository; a design note alone does not count.

## M0 — reproducible reliability kernel

- [x] Canonical blueprint and append-only decision log
- [x] Initial repository/package skeleton
- [x] Normalized tool-call parser and Pydantic validation
- [x] Typed, JSON-serializable execution events
- [x] Shared gate engine with explicit audit and enforce modes
- [x] Execution-backed composite reward and adversarial gaming tests
- [x] Spawn-safe best-effort subprocess sandbox and escape/resource tests
- [x] Exact `pass^k` and `pass@k` estimators with known-answer tests
- [x] Versioned trajectory JSONL round trip
- [x] End-to-end CPU fixture and local check wrapper
- [ ] First GitHub Actions run green on Windows and Linux
- [ ] Ten-to-fifteen-source verified literature scan
- [ ] Qwen2.5-versus-Qwen3 measured smoke test and decision
- [ ] Function-calling dataset license comparison and decision
- [ ] Repository license decision after the license table exists
- [ ] File Meta gated-model access requests

## M1 onward

See `BLUEPRINT_v2.md` for the full experimental matrix. Work does not proceed
to paid compute or headline evaluation until M0's reward-gaming tests are green.

## Open specification issues

These are tracked rather than silently guessed:

- Define a correction-specific condition and metric for “self-correction.”
- Resolve the M1 R1 requirement versus the M4 LangGraph schedule.
- Define H1 gap closure and token aggregation exactly.
- Define denominators and matched-run protocol for H2 and H3.
- Resolve the 8B-at-R2 escalation target.
- Validate that the Phase B model combination fits the stated GPU budget.
- Correct the reference-policy implementation after the SFT/LoRA smoke test.
