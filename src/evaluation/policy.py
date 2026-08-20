"""Build the one-generation policy that both evaluation and data generation use.

`evaluation.rungs.run_episode` takes a policy: messages in, text out. How that
text is produced has to be identical whether the episode is being scored or
being kept as training data, because a trajectory generated under one rendering
and trained under another teaches the model to answer a prompt it will never be
given.

The rendering therefore goes through the model's own tool interface via
`apply_chat_template(tools=...)`. Describing the format in prose instead
produced a well-formed opening tag and then an early stop, because these
checkpoints were trained on the native interface rather than on a description
of it. That mistake cost a full baseline run and scored 0.0 everywhere.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence


def build_policy(
    *,
    model: Any,
    tokenizer: Any,
    torch: Any,
    tools: Sequence[dict[str, Any]],
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    seed: int,
    enable_thinking: bool = False,
    device: str = "cuda:0",
) -> Callable[[list[dict[str, str]]], str]:
    """One sampled generation per call, rendered through the native interface.

    The seed is applied per generation rather than once per policy so that a
    task's run index reproduces exactly on a rerun, independently of how many
    episodes happened to run before it.
    """

    pad = tokenizer.pad_token_id
    if pad is None:
        pad = tokenizer.eos_token_id
    tool_payloads = list(tools)

    def policy(messages: list[dict[str, str]]) -> str:
        try:
            rendered = tokenizer.apply_chat_template(
                messages,
                tools=tool_payloads,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
        except TypeError:
            # Older templates reject the thinking switch rather than ignoring it.
            rendered = tokenizer.apply_chat_template(
                messages,
                tools=tool_payloads,
                tokenize=False,
                add_generation_prompt=True,
            )
        inputs = tokenizer(rendered, return_tensors="pt").to(device)
        torch.manual_seed(seed)
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=pad,
            )
        return tokenizer.decode(
            generated[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )

    return policy


__all__ = ["build_policy"]
