# Data provenance

Two datasets are selected and pinned. `data/LICENSES.md` carries the
provenance table and the release checklist; this file is only the map.

- **`openai/gsm8k`**, config `main`, MIT, public (D-061). The Phase A task
  source. Split manifests are frozen in `configs/splits/phase_a_gsm8k.json`
  as 1,000 train / 100 dev / 150 test, and
  `scripts/build_phase_a_splits.py --check` rebuilds them from the pinned
  revision and fails if one byte would differ.
- **`Salesforce/xlam-function-calling-60k`**, CC BY 4.0 in its licence field,
  automatic access gate (D-058). Format grounding. Adopted with its
  research-only card wording recorded as accepted rather than resolved, and
  with the CC BY attribution duties in the `data/LICENSES.md` checklist. How
  much of it is mixed in is decided at M1 by the measured format error rate.
  Glaive was rejected.

Contamination on the Phase A source is measured, not assumed: see D-064 and
`results/contamination-qwen3-57b2bc2.json`.

Raw and processed datasets stay out of Git. Only immutable source IDs,
revisions, licences, and split manifests are committed.
