# Split manifests

Frozen task lists. Every published number was measured on one of these, and a
loader refuses any row whose content hash has moved.

`phase_a_gsm8k.json` holds the GSM8K splits: 1000 train, 100 dev, 150 test.
Train comes from the upstream train split; dev and test come from upstream test
(D-061). Rows store a dataset index and a hash of the question and answer, so
the tasks are pinned without vendoring the dataset.

`phase_b_orders.json` holds the order-support splits: 200 train, 50 dev, 150
test. These tasks are generated rather than sampled, so the manifest stores each
one in full under its own hash. Rebuilding it with
`scripts/build_phase_b_split.py` reproduces the file byte for byte.

`sft_phase_a.json` records which train tasks survived teacher generation and
retention, which is what the SFT dataset was actually built from.

Splits are disjoint by task ID and by content hash, and a test enforces both.
Test is evaluated once per arm.
