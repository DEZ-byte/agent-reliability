"""Train and measure an SFT arm end to end, resumably, from a teacher to a table.

This is the chain behind the README headline, driven as one command so it can
run unattended on a rented or Colab GPU:

    teacher trajectories -> dataset -> base episodes -> comparator episodes
    -> per seed: SFT -> dev selection -> test once -> paired comparisons
    -> summary table

Every step is one of the existing scripts, invoked as a subprocess with the
same flags an operator would type. Nothing here measures anything itself.

Four properties matter for a session that can die at any minute:

* **Everything is written under `--run-dir`, outside the repository.** The
  measuring scripts refuse a dirty worktree, and `git status` counts an
  untracked result as dirt, so a pipeline that wrote into `results/` would
  block its own next step.
* **A stage is skipped only when its artifact exists, was executed, and holds
  no error.** The runners record a failed model load as result data with exit
  code 0; that is not a finished stage. A failed artifact is renamed
  `*.failed-<time>.json` so the next run measures again. Checkpoint selection
  also resumes inside a stage (`--resume`), checking adapter weight hashes.
* **Test is touched once per arm.** The test stage only runs after selection
  has named a checkpoint, and a finished test artifact is never re-run.
* **A smoke run cannot pose as a real one.** `--limit`/`--max-steps` move every
  output into a `smoke-…` subdirectory of the run dir, and the summary says so.

Artifacts are named `<kind>-<slug>-<commit7>.json` so they can be copied into
`results/` as they are. Checkpoint and scratch directories carry the commit too,
so a code change cannot alias an earlier run's weights or dev scores. The freeze
itself (manifest, commits, decision entry) stays a human step, on purpose.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

PROJECT_ROOT: Final = Path(__file__).resolve().parents[1]
SCRIPTS: Final = PROJECT_ROOT / "scripts"
EVAL_TEST: Final = PROJECT_ROOT / "configs" / "eval.yaml"
REGISTRY_PATH: Final = PROJECT_ROOT / "configs" / "model_candidates.json"

# Rough minutes per stage. All rows except the teacher are measured on an RTX
# 4060 Laptop (8 GB) with the 1.7B; generation is bandwidth-bound, so a Colab
# L4 lands near these and a 4B runs slower. The teacher row is an
# extrapolation for a 14B at --batch-size 16 (the 4B teacher measured 2 h 54
# at batch 1); it has not been measured.
ESTIMATES_MINUTES: Final = {
    "teacher": 60,
    "dataset": 1,
    "base": 60,
    "comparator": 35,
    "sft": 10,
    "select": 85,
    "test": 60,
    "compare": 1,
    "contamination": 25,
}

DECISION_ID: Final = re.compile(r"^(D-\d{3}|none)$")


class PipelineError(RuntimeError):
    """A stage did not produce the artifact it promised."""


def _slug(model_id: str) -> str:
    return model_id.split("/")[-1].lower()


def _git(*args: str) -> str:
    completed = subprocess.run(
        ["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True, check=False
    )
    return completed.stdout.strip()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def registry_revision(model_id: str) -> str | None:
    """The pinned revision for a registered model, else None."""

    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    for entries in registry["roles"].values():
        for entry in entries:
            if entry["id"] == model_id:
                return entry["revision"]
    return None


def executed(path: Path) -> bool:
    """True when the artifact exists, parses, was executed, and holds no error.

    The runners catch a failed model load and write it as `results: [{...,
    "error": ...}]` with `executed: true` and exit 0. That file is a record of
    a failure, not a finished stage; treating it as done would skip the stage
    on every resume and hand an empty episodes file to the comparison.
    """

    if not path.is_file():
        return False
    try:
        payload = _load(path)
    except ValueError:
        return False
    if "executed" in payload and not payload["executed"]:
        return False
    results = payload.get("results")
    if isinstance(results, list) and any(
        isinstance(entry, dict) and "error" in entry for entry in results
    ):
        return False
    return True


def entry_for(payload: dict[str, Any], candidate: str) -> dict[str, Any] | None:
    """The results entry for one candidate in a runner artifact."""

    for entry in payload.get("results", []):
        if entry.get("candidate") == candidate:
            return entry
    return None


class Pipeline:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        base_dir = Path(args.run_dir).resolve()
        self.smoke = bool(args.limit or args.max_steps)
        self.run_dir = (
            base_dir / f"smoke-limit{args.limit or 0}-steps{args.max_steps or 0}"
            if self.smoke
            else base_dir
        )
        self.results = self.run_dir / "results"
        self.commit = _git("rev-parse", "--short=7", "HEAD") or "unknown"
        self.log_path = self.run_dir / "pipeline-log.jsonl"
        self.teacher_slug = _slug(args.teacher)
        pinned = registry_revision(args.teacher)
        self.teacher_revision = pinned or args.teacher_revision
        self.teacher_revision_source = "registry" if pinned else "explicit"
        self.reused: dict[str, dict[str, Path]] = {}
        for directory in (
            self.results,
            self.run_dir / "data" / "teacher",
            self.run_dir / "data" / "sft",
            self.run_dir / "configs",
            self.run_dir / "checkpoints",
            self.run_dir / "scratch",
        ):
            directory.mkdir(parents=True, exist_ok=True)

    # -- plumbing -----------------------------------------------------------

    def _log(self, **record: Any) -> None:
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"at": _now(), **record}) + "\n")

    def _run(self, stage: str, command: list[str], produces: Path) -> None:
        """Run one script unless its artifact already exists and executed."""

        if executed(produces):
            print(f"[skip] {stage}: {produces.name} already executed", flush=True)
            self._log(stage=stage, status="skipped", artifact=str(produces))
            return
        if self.args.dry_run:
            print(f"[plan] {stage}: {' '.join(command)}", flush=True)
            return
        estimate = ESTIMATES_MINUTES.get(stage.split(":")[0], "?")
        print(f"\n[run ] {stage} (~{estimate} min on a 4060/L4)", flush=True)
        print("       " + " ".join(command), flush=True)
        started = time.time()
        completed = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
        seconds = round(time.time() - started, 1)
        status = "ok" if completed.returncode == 0 and executed(produces) else "failed"
        detail = None
        if status == "failed" and produces.is_file():
            # Keep the evidence, but out of the way of the next attempt.
            failed = produces.with_name(
                f"{produces.stem}.failed-{int(started)}{produces.suffix}"
            )
            produces.rename(failed)
            detail = str(failed)
            try:
                errors = [
                    entry.get("error")
                    for entry in _load(failed).get("results", [])
                    if isinstance(entry, dict) and "error" in entry
                ]
            except ValueError:
                errors = []
            if errors:
                print(f"[error] {errors[0]}", flush=True)
        self._log(
            stage=stage,
            status=status,
            seconds=seconds,
            returncode=completed.returncode,
            artifact=str(produces),
            failed_artifact=detail,
        )
        print(f"[{status}] {stage} in {seconds/60:.1f} min", flush=True)
        if status != "ok":
            raise PipelineError(
                f"{stage} failed (exit {completed.returncode}) or left no "
                f"executed, error-free artifact at {produces}"
            )

    def _measure_flags(self) -> list[str]:
        flags = ["--run-load", "--allow-download"]
        if self.args.limit:
            flags += ["--limit", str(self.args.limit)]
        return flags

    def _find_reusable(self, pattern: str, candidate: str | None) -> dict[str, Path] | None:
        """Episodes plus their summary from --reuse-dir, if both qualify.

        Both files must exist with the same stem, the summary must be an
        executed test-split artifact (`eval_config` naming eval.yaml, or an
        older artifact without the field) and must carry the candidate without
        an error. The split check exists because `episodes-phase_a-dev-*.jsonl`
        would otherwise match the same glob.
        """

        if not self.args.reuse_dir:
            return None
        for episodes in sorted(Path(self.args.reuse_dir).glob(pattern)):
            if "-dev-" in episodes.name:
                continue
            summary = episodes.with_name(
                episodes.name.replace("episodes-", "", 1)
            ).with_suffix(".json")
            if not executed(summary):
                continue
            payload = _load(summary)
            config = payload.get("eval_config")
            if config is not None and Path(str(config)).name != EVAL_TEST.name:
                continue
            if candidate is not None and entry_for(payload, candidate) is None:
                continue
            return {"episodes": episodes, "summary": summary}
        return None

    # -- stages -------------------------------------------------------------

    def teacher(self) -> Path:
        rows = self.run_dir / "data" / "teacher" / f"{self.teacher_slug}-train.jsonl"
        summary = self.results / f"sft-candidates-{self.teacher_slug}-{self.commit}.json"
        command = [
            sys.executable, str(SCRIPTS / "generate_sft_trajectories.py"),
            "--model", self.args.teacher,
            "--candidates", str(rows),
            "--summary", str(summary),
            "--batch-size", str(self.args.batch_size),
            *self._measure_flags(),
        ]
        if self.teacher_revision_source == "explicit":
            if not self.teacher_revision:
                raise PipelineError(
                    f"{self.args.teacher} is not in the registry; pass --teacher-revision"
                )
            command += ["--revision", self.teacher_revision]
        self._run("teacher", command, summary)
        return rows

    def dataset(self, student: str, rows: Path) -> Path:
        slug = _slug(student)
        dataset = self.run_dir / "data" / "sft" / f"phase_a-{self.teacher_slug}-teacher-{slug}.jsonl"
        manifest = self.run_dir / "configs" / f"sft_phase_a-{self.teacher_slug}-{slug}.json"
        summary = self.results / f"sft-dataset-{slug}-{self.teacher_slug}-{self.commit}.json"
        command = [
            sys.executable, str(SCRIPTS / "build_sft_dataset.py"),
            "--candidates", str(rows),
            "--dataset", str(dataset),
            "--summary", str(summary),
            "--manifest", str(manifest),
            "--tokenizer", student,
        ]
        self._run(f"dataset:{slug}", command, summary)
        return dataset

    def base_episodes(self, student: str) -> Path:
        slug = _slug(student)
        reusable = self._find_reusable("episodes-phase_a-*.jsonl", student) or self._find_reusable(
            f"episodes-baseline-{slug}-*.jsonl", student
        )
        if reusable is not None:
            print(f"[reuse] base episodes for {student}: {reusable['episodes']}", flush=True)
            self.reused[f"base:{student}"] = reusable
            self._log(
                stage=f"base:{slug}",
                status="reused",
                artifact=str(reusable["episodes"]),
                summary=str(reusable["summary"]),
                sha256=_sha256(reusable["episodes"]),
            )
            return reusable["episodes"]
        summary = self.results / f"baseline-{slug}-{self.commit}.json"
        episodes = self.results / f"episodes-baseline-{slug}-{self.commit}.jsonl"
        command = [
            sys.executable, str(SCRIPTS / "run_phase_a_baseline.py"),
            "--config", str(EVAL_TEST),
            "--candidate", student,
            "--output", str(summary),
            "--episodes", str(episodes),
            *self._measure_flags(),
        ]
        self._run(f"base:{slug}", command, summary)
        return episodes

    def comparator_episodes(self) -> Path | None:
        if self.args.skip_comparator:
            return None
        reusable = self._find_reusable("episodes-comparator-*.jsonl", self.args.comparator)
        if reusable is not None:
            print(f"[reuse] comparator episodes: {reusable['episodes']}", flush=True)
            self.reused["comparator"] = reusable
            self._log(
                stage="comparator",
                status="reused",
                artifact=str(reusable["episodes"]),
                summary=str(reusable["summary"]),
                sha256=_sha256(reusable["episodes"]),
            )
            return reusable["episodes"]
        summary = self.results / f"comparator-8b-{self.commit}.json"
        episodes = self.results / f"episodes-comparator-8b-{self.commit}.jsonl"
        command = [
            sys.executable, str(SCRIPTS / "run_phase_a_baseline.py"),
            "--config", str(EVAL_TEST),
            "--candidate", self.args.comparator,
            "--output", str(summary),
            "--episodes", str(episodes),
            *self._measure_flags(),
        ]
        self._run("comparator", command, summary)
        return episodes

    def sft(self, student: str, dataset: Path, seed: int) -> Path:
        slug = _slug(student)
        out_dir = self.run_dir / "checkpoints" / f"{slug}-sft-seed{seed}-{self.commit}"
        summary = self.results / f"sft-run-{slug}-seed{seed}-{self.commit}.json"
        command = [
            sys.executable, str(SCRIPTS / "train_sft.py"),
            "--dataset", str(dataset),
            "--model", student,
            "--output-dir", str(out_dir),
            "--summary", str(summary),
            "--seed", str(seed),
            "--run-load", "--allow-download",
        ]
        if self.args.limit:
            command += ["--limit", str(self.args.limit)]
        if self.args.max_steps:
            command += ["--max-steps", str(self.args.max_steps)]
        self._run(f"sft:{slug}:seed{seed}", command, summary)
        return out_dir

    def select(self, student: str, adapter_dir: Path, seed: int) -> Path:
        slug = _slug(student)
        summary = self.results / f"sft-selection-{slug}-seed{seed}-{self.commit}.json"
        scratch = self.run_dir / "scratch" / f"select-{slug}-seed{seed}-{self.commit}"
        command = [
            sys.executable, str(SCRIPTS / "select_checkpoint.py"),
            "--adapter-dir", str(adapter_dir),
            "--base-model", student,
            "--summary", str(summary),
            "--scratch", str(scratch),
            "--resume",
        ]
        if self.args.limit:
            command += ["--limit", str(self.args.limit)]
        self._run(f"select:{slug}:seed{seed}", command, summary)
        if self.args.dry_run and not summary.is_file():
            return adapter_dir / "checkpoint-<selected-on-dev>"
        return Path(_load(summary)["selected"]["path"])

    def test(self, student: str, checkpoint: Path, seed: int) -> Path:
        slug = _slug(student)
        summary = self.results / f"sft-test-{slug}-seed{seed}-{self.commit}.json"
        episodes = self.results / f"episodes-sft-test-{slug}-seed{seed}-{self.commit}.jsonl"
        command = [
            sys.executable, str(SCRIPTS / "run_phase_a_baseline.py"),
            "--config", str(EVAL_TEST),
            "--candidate", student,
            "--adapter", str(checkpoint),
            "--teacher", self.args.teacher,
            "--teacher-deviation", self.args.teacher_deviation,
            "--output", str(summary),
            "--episodes", str(episodes),
            *self._measure_flags(),
        ]
        self._run(f"test:{slug}:seed{seed}", command, summary)
        return episodes

    def compare(
        self,
        *,
        name: str,
        baseline_episodes: Path,
        baseline_candidate: str | None,
        baseline_label: str,
        treatment_episodes: Path,
        treatment_candidate: str,
        treatment_label: str,
        rungs: list[str],
    ) -> Path:
        summary = self.results / f"{name}-{self.commit}.json"
        command = [
            sys.executable, str(SCRIPTS / "compare_arms.py"),
            "--baseline-episodes", str(baseline_episodes),
            "--treatment-episodes", str(treatment_episodes),
            "--baseline-label", baseline_label,
            "--treatment-label", treatment_label,
            "--treatment-candidate", treatment_candidate,
            "--summary", str(summary),
        ]
        if baseline_candidate:
            command += ["--baseline-candidate", baseline_candidate]
        for rung in rungs:
            command += ["--rung", rung]
        self._run(f"compare:{name}", command, summary)
        return summary

    def contamination(self, student: str, checkpoint: Path, seed: int) -> None:
        slug = _slug(student)
        summary = self.results / f"contamination-sft-{slug}-seed{seed}-{self.commit}.json"
        command = [
            sys.executable, str(SCRIPTS / "probe_contamination.py"),
            "--candidate", student,
            "--adapter", str(checkpoint),
            "--output", str(summary),
            *self._measure_flags(),
        ]
        self._run(f"contamination:{slug}:seed{seed}", command, summary)

    # -- the chain ----------------------------------------------------------

    def run(self) -> dict[str, Any]:
        self._log(
            stage="pipeline",
            status="start",
            commit=self.commit,
            smoke=self.smoke,
            teacher_revision=self.teacher_revision,
            args=vars(self.args),
        )
        rows = self.teacher()
        comparator = self.comparator_episodes()
        arms: list[dict[str, Any]] = []
        for student in self.args.student:
            slug = _slug(student)
            dataset = self.dataset(student, rows)
            base = self.base_episodes(student)
            for seed in self.args.seed:
                adapter_dir = self.sft(student, dataset, seed)
                checkpoint = self.select(student, adapter_dir, seed)
                episodes = self.test(student, checkpoint, seed)
                label = (
                    f"{student.split('/')[-1]} SFT seed {seed} {checkpoint.name} "
                    f"(teacher {self.args.teacher})"
                )
                arm: dict[str, Any] = {
                    "student": student,
                    "seed": seed,
                    "checkpoint": str(checkpoint),
                    "test": str(self.results / f"sft-test-{slug}-seed{seed}-{self.commit}.json"),
                }
                arm["vs_base"] = str(
                    self.compare(
                        name=f"sft-comparison-{slug}-seed{seed}",
                        baseline_episodes=base,
                        baseline_candidate=student,
                        baseline_label=f"{student.split('/')[-1]} base",
                        treatment_episodes=episodes,
                        treatment_candidate=student,
                        treatment_label=label,
                        rungs=["R0", "R1"],
                    )
                )
                if comparator is not None:
                    arm["vs_comparator"] = str(
                        self.compare(
                            name=f"h1-comparison-{slug}-seed{seed}",
                            baseline_episodes=comparator,
                            baseline_candidate=self.args.comparator,
                            baseline_label=f"{self.args.comparator.split('/')[-1]} scaffolded (R1)",
                            treatment_episodes=episodes,
                            treatment_candidate=student,
                            treatment_label=label,
                            rungs=["R1"],
                        )
                    )
                if self.args.contamination and seed == self.args.seed[0]:
                    self.contamination(student, checkpoint, seed)
                arms.append(arm)
        summary = self.summarise(arms, comparator)
        self._log(stage="pipeline", status="done")
        return summary

    # -- reporting ----------------------------------------------------------

    def summarise(self, arms: list[dict[str, Any]], comparator: Path | None) -> dict[str, Any]:
        if self.args.dry_run:
            return {}
        rows: list[dict[str, Any]] = []

        def metrics_rows(label: str, artifact: Path, candidate: str, source: str) -> None:
            if not artifact.is_file():
                rows.append({"arm": label, "rung": "—", "missing": str(artifact)})
                return
            entry = entry_for(_load(artifact), candidate)
            if entry is None or "error" in entry:
                rows.append({"arm": label, "rung": "—", "missing": str(artifact)})
                return
            for rung, block in entry["rungs"].items():
                m = block["metrics"]
                rows.append(
                    {
                        "arm": label,
                        "rung": rung,
                        "pass^1": m.get("pass^1"),
                        "pass^1_ci95": m.get("pass^1_ci95"),
                        "pass^4": m.get("pass^4"),
                        "pass^4_ci95": m.get("pass^4_ci95"),
                        "pass@4": m.get("pass@4"),
                        "tokens_per_episode": block.get("generated_tokens_per_episode"),
                        "no_arithmetic_rate": block.get("no_arithmetic_rate"),
                        "source": source,
                        "artifact": str(artifact),
                    }
                )

        for student in self.args.student:
            slug = _slug(student)
            reused = self.reused.get(f"base:{student}")
            if reused:
                metrics_rows(f"{student.split('/')[-1]} base", reused["summary"], student, f"reused {reused['summary'].name}")
            else:
                metrics_rows(f"{student.split('/')[-1]} base", self.results / f"baseline-{slug}-{self.commit}.json", student, "this run")
        for arm in arms:
            metrics_rows(
                f"{arm['student'].split('/')[-1]} SFT seed {arm['seed']}", Path(arm["test"]), arm["student"], "this run"
            )
        if comparator is not None:
            reused = self.reused.get("comparator")
            if reused:
                metrics_rows(self.args.comparator.split("/")[-1], reused["summary"], self.args.comparator, f"reused {reused['summary'].name}")
            else:
                metrics_rows(self.args.comparator.split("/")[-1], self.results / f"comparator-8b-{self.commit}.json", self.args.comparator, "this run")

        contrasts: list[dict[str, Any]] = []
        for arm in arms:
            for key in ("vs_base", "vs_comparator"):
                path = arm.get(key)
                if not path or not Path(path).is_file():
                    continue
                payload = _load(Path(path))
                for c in payload["comparisons"]:
                    contrasts.append(
                        {
                            "treatment": payload["treatment_label"],
                            "baseline": payload["baseline_label"],
                            "rung": c["rung"],
                            "k": c["k"],
                            "difference": c["difference"],
                            "difference_ci95": c["difference_ci95"],
                            "p_permutation_two_sided": c["p_permutation_two_sided"],
                            "tasks_compared": c["tasks_compared"],
                            "artifact": path,
                        }
                    )

        summary = {
            "schema_version": 1,
            "created_at_utc": _now(),
            "kind": "primary_arm_pipeline_summary",
            "source_commit": _git("rev-parse", "HEAD") or "unknown",
            "smoke": self.smoke,
            "limit": self.args.limit,
            "max_steps": self.args.max_steps,
            "teacher": {
                "id": self.args.teacher,
                "revision": self.teacher_revision,
                "revision_source": self.teacher_revision_source,
                "deviation": self.args.teacher_deviation,
            },
            "students": self.args.student,
            "seeds": self.args.seed,
            "note": (
                "Numbers here are copied from the per-stage artifacts named in "
                "each row; those artifacts are the record. A smoke run must "
                "never be quoted."
            ),
            "arms": rows,
            "paired_contrasts": contrasts,
        }
        path = self.results / f"summary-{self.commit}.json"
        path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        (self.results / f"summary-{self.commit}.md").write_text(
            render_markdown(summary), encoding="utf-8"
        )
        print("\n" + render_markdown(summary), flush=True)
        return summary


def render_markdown(summary: dict[str, Any]) -> str:
    def f(x: Any) -> str:
        return "—" if x is None else f"{x:.4f}"

    def ci(x: Any) -> str:
        return "—" if not x else f"[{x[0]:.3f}, {x[1]:.3f}]"

    teacher = summary["teacher"]
    lines = [
        f"# SFT arm summary ({summary['source_commit'][:7]})",
        "",
        f"Teacher: `{teacher['id']}` @ `{(teacher.get('revision') or '?')[:12]}` "
        f"({teacher.get('revision_source', '?')}, deviation {teacher.get('deviation', '?')}); "
        f"students: {', '.join('`' + s + '`' for s in summary['students'])}; seeds: {summary['seeds']}",
        "",
    ]
    if summary.get("smoke"):
        lines += [
            f"**SMOKE RUN (--limit {summary.get('limit')}, --max-steps {summary.get('max_steps')}). Do not quote.**",
            "",
        ]
    lines += [
        "| arm | rung | pass^1 | 95% CI | pass^4 | 95% CI | pass@4 | tokens/episode | no-arithmetic rate | source |",
        "| :-- | :-- | --: | :-- | --: | :-- | --: | --: | --: | :-- |",
    ]
    for r in summary["arms"]:
        if "missing" in r:
            lines.append(f"| {r['arm']} | — | — | — | — | — | — | — | — | missing: {Path(r['missing']).name} |")
            continue
        lines.append(
            f"| {r['arm']} | {r['rung']} | {f(r['pass^1'])} | {ci(r['pass^1_ci95'])} | "
            f"{f(r['pass^4'])} | {ci(r['pass^4_ci95'])} | {f(r['pass@4'])} | "
            f"{f(r['tokens_per_episode'])} | {f(r['no_arithmetic_rate'])} | {r.get('source', '')} |"
        )
    lines += ["", "## Paired contrasts (treatment minus baseline, same tasks and seeds)", ""]
    lines += [
        "| treatment | baseline | rung | k | difference | 95% CI | p (perm.) | tasks |",
        "| :-- | :-- | :-- | --: | --: | :-- | --: | --: |",
    ]
    for c in summary["paired_contrasts"]:
        lines.append(
            f"| {c['treatment']} | {c['baseline']} | {c['rung']} | {c['k']} | "
            f"{c['difference']:+.4f} | {ci(c['difference_ci95'])} | "
            f"{c['p_permutation_two_sided']:.4f} | {c['tasks_compared']} |"
        )
    lines += [
        "",
        "Every Phase A accuracy figure is reported beside its no-arithmetic rate "
        "(D-062/D-064). pass^1 is 'solved once'; pass^4 is 'solved four of four'.",
        "",
    ]
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--run-dir", required=True, help="all outputs go here, outside the repo")
    parser.add_argument(
        "--student", action="append", default=None, help="HF id; repeatable (default Qwen/Qwen3-4B)"
    )
    parser.add_argument("--teacher", default="Qwen/Qwen3-14B")
    parser.add_argument(
        "--teacher-revision",
        default="40c069824f4251a91eefaf281ebe4c544efd3e18",
        help=(
            "used only when the teacher is not in configs/model_candidates.json; "
            "a registered teacher always uses its pinned revision (D-072 pins "
            "Qwen3-14B at this default)"
        ),
    )
    parser.add_argument(
        "--teacher-deviation",
        required=True,
        help=(
            "decision id (D-NNN) recording how the teacher deviates from section "
            "5.2's plan, or 'none'; it is frozen into each arm's single test artifact"
        ),
    )
    parser.add_argument(
        "--seed", action="append", type=int, default=None,
        help="SFT seed; repeatable (default 20260826 20260827 20260828)",
    )
    parser.add_argument("--batch-size", type=int, default=16, help="teacher generation batch (R0 only)")
    parser.add_argument("--comparator", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--skip-comparator", action="store_true")
    parser.add_argument(
        "--contamination", action="store_true", help="probe the first seed's selected checkpoint"
    )
    parser.add_argument(
        "--reuse-dir", default=None,
        help="directory holding episodes-*.jsonl AND their summary .json to reuse instead of re-measuring",
    )
    parser.add_argument("--limit", type=int, default=None, help="SMOKE ONLY: tasks per split")
    parser.add_argument("--max-steps", type=int, default=None, help="SMOKE ONLY: SFT steps")
    parser.add_argument("--dry-run", action="store_true", help="print the commands, run nothing")
    args = parser.parse_args(argv)
    if not DECISION_ID.match(args.teacher_deviation):
        parser.error("--teacher-deviation must be a decision id like D-078, or 'none'")
    args.student = args.student or ["Qwen/Qwen3-4B"]
    args.seed = args.seed or [20260826, 20260827, 20260828]
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.limit or args.max_steps:
        print(
            "[smoke] --limit/--max-steps set: outputs go to a smoke-… subdirectory "
            "and must not be quoted",
            flush=True,
        )
    pipeline = Pipeline(args)
    try:
        pipeline.run()
    except PipelineError as error:
        print(f"[stop] {error}", file=sys.stderr, flush=True)
        print("[stop] re-run the same command to resume at the failed stage", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
