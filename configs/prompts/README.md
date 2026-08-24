# Prompt registry

Prompts are pinned in code, not here, and this file explains where to find them.

Phase A prompts live in `src/evaluation/rungs.py` as `SYSTEM_PROMPT` and
`USER_PROMPT`. Phase B prompts live in `scripts/run_phase_b_eval.py`. Both are
hashed into every artifact they produce, under `prompt_sha256`, so a run can be
tied to the exact wording that generated it.

They sit in code because the evaluator, the trajectory generator and the GRPO
trainer all have to use the same string. A copy in a config file would be a
second source of truth, and the failure it invites is silent: a run would
measure a prompt the model was never trained against, and nothing would say so.

This directory stays for prompts that are genuinely configuration, meaning
prompts that vary per run rather than per phase. There are none yet.
