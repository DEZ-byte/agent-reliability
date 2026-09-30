# Findings

What this project actually learned, including the parts that went badly.

The `D-0xx` tags name entries in the project's decision log, which holds the full
reasoning behind each. That log is append-only and runs to 78 entries, so it is
kept outside this repository for now. This page is the readable version. The
four decisions the licence gate depends on are public in
[`configs/release_decision.md`](configs/release_decision.md).

Numbers marked † come from local logs or configs, not from a committed artifact.
Corrections to earlier versions of this page are in [`ERRATA.md`](ERRATA.md).

| | Finding | Verdict |
| :-- | :-- | :-- |
| **A** | [A trained 1.7B beat a scaffolded 8B](#a-a-trained-17b-beat-a-scaffolded-8b) | Confirmed |
| **A2** | [It did not reach its teacher](#a2-it-did-not-reach-its-teacher) | Confirmed on `pass^4` in all three runs |
| **B** | [Training raised capability faster than reliability](#b-training-raised-capability-faster-than-reliability) | Confirmed, and awkward |
| **C** | [What improved was tool use, not arithmetic](#c-what-improved-was-tool-use-not-arithmetic) | Confirmed |
| **D** | [It reproduced three times, and got cheaper](#d-it-reproduced-three-times-and-got-cheaper) | Confirmed |
| **D2** | [The capability transferred; the judgement did not](#d2-the-capability-transferred-the-judgement-did-not) | Refusal failure confirmed; the reward verdict is withdrawn |
| **D3** | [No detectable MMLU change on this sample](#d3-no-detectable-mmlu-change-on-this-sample) | Null; reliably detects only changes of about 6 points |
| **E** | [No detectable GRPO benefit at these rates and this budget](#e-no-detectable-grpo-benefit-at-these-rates-and-this-budget) | Null, twice |
| **F** | [Tool formatting was never the problem](#f-tool-formatting-was-never-the-problem) | Killed a planned mitigation |
| **G** | [The retry rung had almost nothing to fix](#g-the-retry-rung-had-almost-nothing-to-fix) | Killed a planned arm |
| **H** | [This environment cannot teach self-correction](#h-this-environment-cannot-teach-self-correction) | Structural, deferred |
| **I** | [Execution-backed grading can be passed without computing](#i-execution-backed-grading-can-be-passed-without-computing) | Measured, not penalised |
| **J** | [Three bugs that would have changed a conclusion](#j-three-bugs-that-would-have-changed-a-conclusion) | Caught |

---

# Results

## A. A trained 1.7B beat a scaffolded 8B

The comparison the project was built to make.

All arms at rung R1 (one retry when the output does not parse).

| | Llama-3.1-8B, scaffolded | Qwen3-1.7B, fine-tuned | Qwen3-4B, untrained (teacher) |
| :-- | --: | --: | --: |
| `pass^1` | 0.415 | 0.517 – 0.553 | 0.608 |
| `pass^4` | 0.293 | 0.393 – 0.460 | 0.567 |
| Cost per attempt, parameter-weighted † | 236 | 73 – 75 | not recorded |
| Memory to serve at 4-bit † | ~6 GB | ~1.5 GB | not measured |

Three training runs, three paired comparisons against the 8B, every interval
excluding zero on both metrics.

Cost went the same way once measured properly. The 8B emits *fewer* raw tokens
per attempt. Weighting by parameters reverses that: the 1.7B uses about 31% of
the 8B's parameter-weighted generated-token proxy per attempt. The proxy counts
decoding only, and the parameter counts come from the model configs †.

`D-076`

## A2. It did not reach its teacher

The untrained Qwen3-4B wrote the trajectories the 1.7B was trained on. It never
appeared in a table until now, and it beats every arm.

| Training run | `pass^1` gap to the 4B | 95% interval | `pass^4` gap to the 4B | 95% interval |
| :-- | --: | :--: | --: | :--: |
| Run 1 | −0.078 | −0.143 to −0.013 | −0.173 | −0.253 to −0.093 |
| Run 2 | −0.092 | −0.157 to −0.028 | −0.153 | −0.227 to −0.073 |
| Run 3 | −0.055 | −0.120 to +0.008 | −0.107 | −0.187 to −0.027 |

Both arms at R1, from
[`results/sft-vs-teacher-run1-d692d43.json`](results/sft-vs-teacher-run1-d692d43.json)
and its two siblings. The `pass^4` gap excludes zero in every run. Fine-tuning
closed most of the single-attempt gap to the teacher and less of the reliability
gap. No decision entry yet.

## B. Training raised capability faster than reliability

The headline, and the second half matters more than the first. All at R0.

| Metric | Untrained | Fine-tuned |
| :-- | --: | --: |
| Solves it at least once in 4 | 0.353 | 0.627 – 0.680 |
| Solves it 4 times out of 4 | 0.247 | 0.393 – 0.460 |
| Solved *sometimes* but not always | 0.107 | 0.213 – 0.287 |

Training was supposed to close the reliability gap. It widened it. Reported
loosely this is +27 to +33 points; reported strictly it is +15 to +21.

The gap between those two numbers is the entire reason the project measures
`pass^k`.

`D-073`

## C. What improved was tool use, not arithmetic

Probed with the calculator removed entirely, the model scored 64.0% before
training and 66.0% after. It did not get better at maths. It got better at
writing the expression.

Worth stating plainly, because "fine-tuning improved accuracy" invites exactly
the wrong reading.

`D-074` · `D-064`

## D. It reproduced three times, and got cheaper

| Run | `pass^1` gain over base | Dev peak |
| :-- | --: | --: |
| 1 | +0.222 | 0.4725 |
| 2 | +0.212 | 0.4700 |
| 3 | +0.248 | 0.4975 |

The trained model also spends about 0.78× the tokens of the untrained one †.

The more useful number is what the spread says about the experiment. Between runs
the standard deviation is 0.019; within a single run the confidence interval is
about 0.069 either side. Three runs make that standard deviation rough, but the
gap is large, so a bigger test split would likely sharpen this result more than
more seeds.

Dev told a slightly different story from test. A project that ran once and
happened to draw the third seed would have reported a better number, with nothing
in that single run to say so.

`D-074` · `D-075`

## D2. The capability transferred; the judgement did not

Everything above was measured on the task the model was trained for. To tell a
general improvement from a narrow one, the checkpoints were run on a second
environment: an order-support agent offered three unseen tools, no arithmetic.
Half the requests should be completed, half refused. It runs in audit mode, so a
write the gates should stop still lands. It is single-shot: the model writes all
its tool calls before any tool runs, and never sees a result.

| | Untrained | After SFT | After GRPO ‡ |
| :-- | --: | --: | --: |
| `pass^1` | 0.493 | 0.528 | 0.542 |
| Completes a legitimate request (outcome) | 0.000 | 0.947 | 0.957 |
| Correctly refuses an unverified one | 1.000 | 0.098 | 0.115 |
| Writes for an unverified customer (of 296) | 0 | 267 | 262 |
| Verified writes that skipped the lookup (of 304) | 0 | 285 | 289 |
| Episodes that called the lookup tool (of 600) | 0 | 3 | 2 |
| Mean reward, as scored | +0.286 | -0.394 | -0.387 |
| Mean reward, if skipping the lookup were not penalised | +0.286 | +0.366 | +0.384 |

The gate split and the rescoring come from
[`results/phase_b-gates-d692d43.json`](results/phase_b-gates-d692d43.json), which
replays all 1,800 episodes and reproduces every stored reward.

The headline moves by three points. Underneath, the two halves swap places.

Fine-tuning on GSM8K and a calculator taught the model to complete 95% of
legitimate requests by outcome, in audit mode, with severe authorisation
failures. It operated two of the three offered tools, in a domain with no maths
in it, from a standing start of zero. That is real transfer and it is the best
news in this project.

It also taught it to act unconditionally. On requests it should refuse, it wrote
anyway in 267 of 296 episodes. It emits the verification call and the write in
the same turn, so it never learns that verification failed. The untrained model
scores about half by never writing; the trained one scores about half by always
writing.

The mean reward does not settle which is worse. As scored, it is negative after
training and positive before. But 285 of the 552 violations are writes for a
verified customer that only skipped `get_order`, and the prompt never asks for a
lookup. Without that penalty the fine-tuned model scores +0.366, above the
untrained +0.286. The earlier claim that it is worse than doing nothing is
withdrawn (ERRATA E10). A re-run with a prompt that asks for the lookup would
settle it.

None of this is mysterious. Every training example was one call and done, and
not one had "do not call the tool" as the correct answer, so an unconditional
policy fits the data perfectly. It is finding H arriving in a different form: an
environment that ends the episode on the first successful call cannot teach a
model to read a result and decide what to do about it.

Measured on the 150-task transfer split, four attempts per task, in audit mode
so an unauthorised write lands and can be counted. No decision entry yet.

## D3. No detectable MMLU change on this sample

The first question anyone asks about fine-tuning is what it broke. Until now
this project could not answer, because every number was measured on the task the
model was trained for. So all three checkpoints were run over 400 held-out MMLU
questions, stratified across all 57 subjects, with no tool offered.

| | Untrained | After SFT | After GRPO ‡ |
| :-- | --: | --: | --: |
| Accuracy | 0.5350 (214/400) | 0.5425 (217/400) | 0.5375 (215/400) |
| Paired difference vs untrained | - | +0.0075 | +0.0025 |
| 95% interval | - | -0.0350 to +0.0500 | -0.0400 to +0.0450 |
| Questions improved / got worse | - | 40 / 37 | 40 / 39 |
| Emitted a tool call | 0.000 | 0.000 | 0.000 |
| Mean answer length, characters | 628 | 227 | 233 |

Every figure comes from
[`results/utility-comparison-48869b1.json`](results/utility-comparison-48869b1.json),
which pairs the arms question by question. An earlier version of this table
reported +0.005 and 38 / 36. Those figures came from an incomplete response
file ([`ERRATA.md`](ERRATA.md), E4).

Two results, and the second was not the expected one.

There was no detectable MMLU change on this 400-question sample. That is weaker
than "nothing was forgotten". The interval runs from -3.5 to +5.0 points. With 77
of 400 questions changing answer, a test this size has about an 80% chance of
detecting a change of about 6 points, and less for anything smaller. Restricted
to the 374 questions
where both arms named a letter, the difference is -0.008 (-0.051 to +0.035). So
the sign depends on how unreadable answers are counted, and neither reading can
be told apart from zero.

The tool-calling habit did not leak. Going in, the obvious worry was that a
model trained to emit a tool call on every single example would start emitting
them at anything question-shaped. It does not, on any arm, on any question. The
habit is tied to being offered a tool rather than to being asked something.

That is worth holding next to finding D2, which is not a contradiction. What
transferred to the new tool environment was the habit of acting; what did not
was the judgement of when to. Neither shows up here, because nothing here offers
a tool to act with.

One real change did show up. The fine-tuned model answers about a third as
long. Terser, with no detectable accuracy cost. The truncation rates published
earlier counted batch padding as truncation and are withdrawn
([`ERRATA.md`](ERRATA.md), E7).

‡ In this table and in D2, the GRPO column used the final step-400 adapter, not
the dev-selected checkpoint-200 (ERRATA E8). It is a diagnostic of where the
run ended.

Measured on 400 questions. No decision entry yet.

---

# Things that did not work

## E. No detectable GRPO benefit at these rates and this budget

| Learning rate | `pass^1` change (R0) | 95% interval | Weights moved |
| :-- | --: | :--: | --: |
| 1e-6 | +0.002 | −0.010 – 0.013 | 0.41% |
| 1e-5 | +0.010 | −0.020 – 0.040 | 3.77% |

All four numbers come from the checkpoints dev selection picked (checkpoint-200
and checkpoint-300). The weight figures are in
[`results/weight-change-72f7482.json`](results/weight-change-72f7482.json). An
earlier version showed 3.82% here, which was the final step-400 adapter
(ERRATA E8).

The interval on the first run excludes an effect much larger than a point,
rather than merely failing to find one.

The obvious objection was that the run barely moved the model, and that objection
was correct. Rerunning at ten times the rate moved the weights about nine times as
far and produced an identical dev peak. Two nulls across a tenfold rate range are
much harder to dismiss than one.

**Consistent with the null, though none was tested as a cause:**

| Measurement | Value |
| :-- | :-- |
| Steps with no gradient | 23% at 1e-6, 27% at 1e-5. Each step scored 16 attempts, 8 at each of 2 problems, and in these steps all 16 scored alike. The per-problem rate was not logged; it is at least as high. |
| Reward spread within a step | Accuracy dominated (0.339); format (0.006) and efficiency (0.002) varied negligibly; the gate term was inert (0.000) |
| What was left to fix | After SFT almost every failure is a well-formed call with the wrong value, which GRPO can learn from only when some attempts get it right |

The gate term reads 0.000 for a structural reason: this environment has one
harmless tool, so no gate can ever fire.

One direction worth chasing, not yet a result: the higher rate raised `pass^4` and
lowered `pass@4`, narrowing the sometimes-solved band from 0.213 to 0.167. That is
what a policy-gradient method concentrating probability mass looks like, and it is
the trade this project cares about. A paired test gives p = 0.24 †, so it is a hint.

`D-077` · `D-078`

## F. Tool formatting was never the problem

Both models emitted schema-valid tool calls essentially every time; the schema
failure rate measured 0.00%. Roughly 85–91% of all failures were a perfectly
well-formed call that computed the wrong thing.

That killed a planned mitigation — mixing in an external function-calling dataset
to teach tool formatting. There was nothing left to teach.

`D-068` · `D-070`

## G. The retry rung had almost nothing to fix

A second attempt only helps when the failure is visible at runtime. Here it
usually was not: a wrong answer still executes cleanly, the loop ends, and nobody
objects.

The retry fired on 3.7% of episodes for one model and 10.3% for the other. After
training it stopped making any difference at all — its `pass^4` is identical to
the single-attempt rung.

`D-068`

## H. This environment cannot teach self-correction

The episode ends as soon as a tool call succeeds, whether or not the answer is
right. So the dominant failure never gets a second look.

It cannot be patched by changing the loop either. Telling the model its answer is
wrong would leak the grader into the rollout. Measured yield for genuine recovery
trajectories: about one task in a hundred.

Self-correction moved to a later stage where tool errors are actually observable.

`D-069`

---

# Honesty checks

## I. Execution-backed grading can be passed without computing

Grading from executed results is supposed to stop a model reciting a memorised
answer. It does not. A call like `calculator("391")` scores correct having
computed nothing, and the obvious fix — requiring the expression to do arithmetic
— is defeated by writing `391 + 0`.

The reward pays exactly the same for this as for genuine work. That was
deliberate: measure the behaviour rather than penalise it, so the rate stays
visible instead of being pushed somewhere harder to see.

| Model | Rate † |
| :-- | --: |
| Untrained | 3.0% |
| Fine-tuned | 1.2% |
| After reinforcement learning | 1.0% |

RL optimises whatever scores highest, so this was the cheapest available shortcut
and it had 400 steps to find it. The rate went down instead.

`D-062` · `D-077`

## J. Three bugs that would have changed a conclusion

| Bug | What it would have done |
| :-- | :-- |
| **The 8B scored 0.000.** Llama writes tool calls as bare JSON with `parameters`; Qwen wraps them in `<tool_call>` tags with `arguments`. The grader understood one dialect. | The headline would have read as a total capability failure by the 8B. |
| **The anti-cheating filter rejected correct work.** One rule required the arithmetic to use numbers appearing in the question, but GSM8K writes quantities as words. "Three dozen eggs for her four children" contains no digits, so a correct `36 / 4` looked invented. | It was discarding genuine multi-step reasoning and catching nothing the other rules missed. It is off. |
| **Two bugs only a green CI run could find.** A lone surrogate character compiled into a docstring, and line-ending translation quietly breaking every recorded content hash on any machine but the one that wrote it. | The local suite passed through both. |

The dialect fix is per-model and off by default, which matters more than it
sounds. Applied globally it would have accepted three recorded completions where
an untrained Qwen wrote bare JSON. By Qwen's own template that is a real format
failure, and accepting it would have flattered the baseline and shrunk every gain
measured against it.

`D-076` · `D-060` · `D-071` · `D-057` · `D-046`

---

# What is not settled

| Open question | Status |
| :-- | :-- |
| Does the same hold for a same-family 8B? | Not run. The comparator is a Llama, so size and pretraining are tangled. |
| Does the 4B benefit from training too? | Not run. It was used as the teacher, so only the 1.7B has a trained arm. Untrained, it already beats every arm (A2). `D-072` |
| Did training damage anything off-task? | Partly answered. No detectable MMLU change on 400 questions (D3), but that test only reliably detects changes of about 6 points. |
| Would GRPO work with dead groups filtered out? | Not tried. The filter is built but has not been run. The null above is a statement about this budget and this setup, not about the method. |
| Does the Phase B refusal failure survive a prompt that asks for the lookup? | Not run. The gate environment is built and measured (D2), but its reward verdict depends on a lookup the prompt never asked for. |
| Do the Phase B and MMLU GRPO numbers hold on the selected checkpoint? | Not run. Both used the final step-400 adapter (ERRATA E8). |
