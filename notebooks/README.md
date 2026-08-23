# Notebooks

Reusable logic belongs in `src/` and `scripts/`; a notebook only configures
and invokes it. Notebooks are gitignored (`notebooks/*.ipynb`) because they are
operator entry points, not results; upload the one you need to Colab.

## `colab_primary_arm.ipynb` — the primary-model SFT arm on one Colab GPU

Drives `scripts/run_primary_arm_pipeline.py`: Qwen3-14B writes verified
trajectories over the frozen train split, Qwen3-4B is fine-tuned on them with
three seeds, each run is selected on dev and tested exactly once, and each is
compared against the 4B base and the frozen Llama-3.1-8B comparator. It
produces the artifacts that `results/` expects, named `<kind>-<slug>-<commit>`,
under a run folder on Google Drive, plus a summary table with paired contrasts.

Why it exists: the README headline is the 1.7B scale-check model. The
pre-registered comparison wants the primary 4B trained, which needs a teacher
larger than 4B (D-072) and therefore a 24 GB card.

Budget: about 10 hours on an L4 for three seeds, less if the local episode
files are reused. Every stage is resumable: if the session dies, run the
notebook again and it continues at the first missing artifact.

What it does not do: write into the repository, build the artifact manifest,
or record a decision. The last cell prints those commands; they stay human.

`tool_reliability_sft_grpo.ipynb` is an older self-contained demo that shares
no code with the repository and should not be used for any reported number.
