# Model and dataset provenance

**What this is for.** A permissive licence on this repository does not make the
models or datasets permissive. Each one carries its own terms, and two of them
carry terms that would quietly constrain anything released from this project.
This file records which, and at exactly which revision.

Verified on **2026-08-18** against the publishers' Hugging Face repositories and
the available license text or repository metadata. The revisions below are full
default-branch commit SHAs, not floating `main` references, so a claim here can
be rechecked against the exact bytes it was made about.

Since verification, D-048 selected the Qwen3 bundle under a
`public-portfolio-permissive` release scope, and the four Qwen candidates have
been downloaded and measured. The non-commercial term on
`Qwen/Qwen2.5-3B-Instruct` is why the technically stronger bundle was not
chosen.

This is an engineering provenance record, not legal advice. A repository code
license will not relicense model weights, adapters, generated training data, or
third-party datasets. Preserve the applicable upstream terms with every
released artifact.

## Models

| Role | Verified checkpoint and revision | Access | License at the verified revision | Release implications and caveats |
| --- | --- | --- | --- | --- |
| Primary candidate | [`Qwen/Qwen2.5-3B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) @ [`aa8e72537993ba99e69dfaafa59ed015b17504d1`](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct/tree/aa8e72537993ba99e69dfaafa59ed015b17504d1) | Public; no click-through gate | Hugging Face metadata uses `license: other` with `license_name: qwen-research`; the linked text is the [Qwen Research License Agreement](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct/blob/aa8e72537993ba99e69dfaafa59ed015b17504d1/LICENSE) | **Non-commercial research/evaluation only.** Commercial use requires a separate Alibaba Cloud license. Redistribution requires the agreement, change notices, and the specified Qwen attribution notice. A distributed model trained or improved with the materials or their outputs must display “Built with Qwen” or “Improved using Qwen.” Conservatively treat released fine-tunes/adapters as covered derivatives. |
| Primary candidate | [`Qwen/Qwen3-4B`](https://huggingface.co/Qwen/Qwen3-4B) @ [`1cfa9a7208912126459214e8b04321603b3df60c`](https://huggingface.co/Qwen/Qwen3-4B/tree/1cfa9a7208912126459214e8b04321603b3df60c) | Public; no click-through gate | [`apache-2.0`](https://huggingface.co/Qwen/Qwen3-4B/blob/1cfa9a7208912126459214e8b04321603b3df60c/LICENSE) | Weights and derivatives may be used and redistributed, including commercially, subject to Apache 2.0. Include the license, mark modified files, retain applicable notices/NOTICE content, and preserve the patent/trademark limitations. This is the requested original hybrid-thinking `Qwen3-4B`, not a later `*-2507` checkpoint. |
| Scale candidate | [`Qwen/Qwen2.5-1.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) @ [`989aa7980e4cf806f80c7fef2b1adb7bc71aa306`](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/tree/989aa7980e4cf806f80c7fef2b1adb7bc71aa306) | Public; no click-through gate | [`apache-2.0`](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/blob/989aa7980e4cf806f80c7fef2b1adb7bc71aa306/LICENSE) | Apache 2.0 redistribution duties apply: include the license, mark changes, and retain applicable attribution and NOTICE content. |
| Scale candidate | [`Qwen/Qwen3-1.7B`](https://huggingface.co/Qwen/Qwen3-1.7B) @ [`70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`](https://huggingface.co/Qwen/Qwen3-1.7B/tree/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e) | Public; no click-through gate | [`apache-2.0`](https://huggingface.co/Qwen/Qwen3-1.7B/blob/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e/LICENSE) | Apache 2.0 redistribution duties apply: include the license, mark changes, and retain applicable attribution and NOTICE content. |
| Cross-family check | [`meta-llama/Llama-3.2-3B-Instruct`](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct) @ [`0cb88a4f764b7a12671c53f0838cd831a0843b95`](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct/tree/0cb88a4f764b7a12671c53f0838cd831a0843b95) | **Manual gate**; a Hugging Face account must submit contact information and accept Meta's terms | [`llama3.2`](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct/blob/0cb88a4f764b7a12671c53f0838cd831a0843b95/LICENSE.txt), Llama 3.2 Community License plus the incorporated [Acceptable Use Policy](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct/blob/0cb88a4f764b7a12671c53f0838cd831a0843b95/USE_POLICY.md) | Redistribution requires the agreement, “Built with Llama,” and the prescribed NOTICE. A distributed model improved with Llama materials or outputs must have a name beginning with “Llama.” The separate-license threshold applies when the licensee or affiliates exceeded 700M monthly active users in the preceding calendar month on the Llama 3.2 version's release date. This candidate is text-only; the multimodal EU clause is not the operative model restriction here. |
| Scaffolded comparator | [`meta-llama/Llama-3.1-8B-Instruct`](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) @ [`0e9e39f249a16976918f6564b8830bc894c89659`](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct/tree/0e9e39f249a16976918f6564b8830bc894c89659) | **Manual gate**; a Hugging Face account must submit contact information and accept Meta's terms | [`llama3.1`](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct/blob/0e9e39f249a16976918f6564b8830bc894c89659/LICENSE), Llama 3.1 Community License plus the incorporated [Acceptable Use Policy](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct/blob/0e9e39f249a16976918f6564b8830bc894c89659/USE_POLICY.md) | Redistribution requires the agreement, “Built with Llama,” and the prescribed NOTICE. A distributed model improved with Llama materials or outputs must have a name beginning with “Llama.” The separate-license threshold applies when the licensee or affiliates exceeded 700M monthly active users in the preceding calendar month on the Llama 3.1 version's release date. The blueprint uses this checkpoint for inference only, not training. |
| User simulator | [`Qwen/Qwen2.5-14B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-14B-Instruct) @ [`cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8`](https://huggingface.co/Qwen/Qwen2.5-14B-Instruct/tree/cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8) | Public; no click-through gate | [`apache-2.0`](https://huggingface.co/Qwen/Qwen2.5-14B-Instruct/blob/cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8/LICENSE) | Apache 2.0 redistribution duties apply. The planned role is evaluation-only; simulator outputs still need provenance if any are later retained or released. |

The Qwen primary and scale winners remain pending the measured smoke test. The
license table must not be used as a proxy for the required tool-template,
VRAM, throughput, and training-stack measurements.

Each registry entry also has an independent release gate. A
`release_eligibility` value of `pending` requires a null `release_decision`.
Changing it to `eligible` or `ineligible` requires the recorded `D-###`
decision that resolved the intended release scope. The four Qwen smoke entries
also record `smoke_bundle`, so a license decision cannot silently pair primary
and scale checkpoints from different generations.

## Function-calling datasets

| Candidate and verified revision | Access | Stated license | Provenance, release implications, and unresolved caveats |
| --- | --- | --- | --- |
| [`Salesforce/xlam-function-calling-60k`](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k) @ [`26d14ebfe18b1f7b524bd39b404b50af5dc97866`](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k/tree/26d14ebfe18b1f7b524bd39b404b50af5dc97866) | **Automatic click-through gate.** The form requests name, country, and affiliation and requires agreement to follow the license and cite APIGen. | Card metadata says [`cc-by-4.0`](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k/blob/26d14ebfe18b1f7b524bd39b404b50af5dc97866/README.md); the official [CC BY 4.0 legal code](https://creativecommons.org/licenses/by/4.0/legalcode.en) permits sharing and adaptation, including commercially, with attribution, a license link, and change indication. | The publisher documents 60,000 APIGen-generated examples, format checking, real function execution, semantic verification, and a 600-example human audit with over 95% reported correctness. Attribute Salesforce/APIGen and cite the paper in released data/model documentation. The card also calls the release “for research purposes only” in its ethical section; that wording sits beside the CC BY 4.0 identifier and should be clarified before commercial reuse. There is one training split, so create project-owned train/dev/test manifests and never treat the source split as held-out evaluation. |
| [`glaiveai/glaive-function-calling-v2`](https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2) @ [`e7f4b6456019f5d8bcb991ef0dd67d8ff23221ac`](https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2/tree/e7f4b6456019f5d8bcb991ef0dd67d8ff23221ac) | Public; no click-through gate | The revision's metadata-only [`README.md`](https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2/blob/e7f4b6456019f5d8bcb991ef0dd67d8ff23221ac/README.md) says `apache-2.0`. | The repository has no full LICENSE file and its README contains no narrative provenance, generation method, verification protocol, attribution statement, or third-party-rights analysis. Applying a software-oriented Apache tag to dataset records without accompanying license text also leaves release handling less clear. The viewer exposes one train split (112,960 rows), so project-owned splits and deduplication would still be required. Do not ingest it until the missing provenance and license scope are resolved. |

## Phase A task dataset

| Dataset and verified revision | Access | Stated license | Notes |
| --- | --- | --- | --- |
| [`openai/gsm8k`](https://huggingface.co/datasets/openai/gsm8k) @ `740312add88f781978c0658806c59bc2815b9866`, config `main` | Public; no gate | `mit` | 7,473 train and 1,319 test grade-school math problems. Verified on 2026-08-20 through the Hub API. MIT permits use and redistribution with the copyright and permission notice preserved, which suits the `public-portfolio-permissive` scope without the caveat that applies to xLAM (D-058). Only split manifests are committed here (`configs/splits/phase_a_gsm8k.json`); the data itself is fetched from the pinned revision and never redistributed from this repository. |

### Recommendation

Prefer **`Salesforce/xlam-function-calling-60k`** for the M1 format-grounding
source, subject to accepting its access conditions and resolving the card's
“research purposes only” wording. Its CC BY attribution load is explicit and
manageable, and its publisher provides materially stronger generation and
verification provenance. Keep Glaive as a fallback only if its publisher adds
or confirms full dataset license terms and adequate provenance. The registry
therefore keeps both candidates at `pending`; this recommendation is not the
dataset decision itself.

## Release checklist

1. Load every artifact by the immutable SHA in `configs/model_candidates.json`.
2. Save the upstream card and license text with the run manifest.
3. Record the exact source IDs of every retained example and generated
   trajectory; commit only split/selection manifests, not raw upstream data.
4. Put required license copies, attribution, change notices, citations, and
   model-name/branding notices beside any released weights or adapters.
5. Run license and provenance review again before a public artifact release;
   upstream cards and terms can change after the verification date.
6. For anything derived from `Salesforce/xlam-function-calling-60k`, CC BY 4.0
   requires all of: attribution to Salesforce, a link to the licence, the
   copyright and disclaimer notice, and an explicit statement that changes were
   made. The access gate additionally requires citing APIGen. D-058 adopts this
   dataset with its licence conflict stated rather than resolved: the declared
   field is `cc-by-4.0`, while the card's ethics prose describes the release as
   research-only. Read D-058 before releasing anything trained on it.
