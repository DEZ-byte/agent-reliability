# Findings

What this project actually learned, including the parts that went badly.

The `D-0xx` tags name entries in the project's decision log, which holds the full
reasoning behind each. That log is append-only and runs to 78 entries, so it is
kept outside this repository for now. This page is the readable version. The
four decisions the licence gate depends on are public in
[`configs/release_decision.md`](configs/release_decision.md).

| | Finding | Verdict |
| :-- | :-- | :-- |
| **A** | [A trained 1.7B beat a scaffolded 8B](#a-a-trained-17b-beat-a-scaffolded-8b) | Confirmed |
| **B** | [Training raised capability faster than reliability](#b-training-raised-capability-faster-than-reliability) | Confirmed, and awkward |
| **C** | [What improved was tool use, not arithmetic](#c-what-improved-was-tool-use-not-arithmetic) | Confirmed |
| **D** | [It reproduced three times, and got cheaper](#d-it-reproduced-three-times-and-got-cheaper) | Confirmed |
| **D2** | [The capability transferred; the judgement did not](#d2-the-capability-transferred-the-judgement-did-not) | New, and the sharpest result here |
| **E** | [Reinforcement learning added nothing after SFT](#e-reinforcement-learning-added-nothing-after-sft) | Null, twice |
| **F** | [Tool formatting was never the problem](#f-tool-formatting-was-never-the-problem) | Killed a planned mitigation |
| **G** | [The retry rung had almost nothing to fix](#g-the-retry-rung-had-almost-nothing-to-fix) | Killed a planned arm |
| **H** | [This environment cannot teach self-correction](#h-this-environment-cannot-teach-self-correction) | Structural, deferred |
| **I** | [Execution-backed grading can be passed without computing](#i-execution-backed-grading-can-be-passed-without-computing) | Measured, not penalised |
| **J** | [Three bugs that would have changed a conclusion](#j-three-bugs-that-would-have-changed-a-conclusion) | Caught |

---

# Results

## A. A trained 1.7B beat a scaffolded 8B

The comparison the project was built to make.

| | Llama-3.1-8B, scaffolded | Qwen3-1.7B, fine-tuned |
| :-- | --: | --: |
| `pass^1` | 0.415 | 0.515 – 0.552 |
| `pass^4` | 0.293 | 0.393 – 0.460 |
| Cost per task, parameter-weighted | 236 | 72 |
| Memory to serve at 4-bit | ~6 GB | ~1.5 GB |

Three training runs, three paired comparisons, every interval excluding zero on
both metrics.

Cost went the same way once measured properly. The 8B emits *fewer* raw tokens
per task. Weighting by parameters reverses that, which is the honest comparison
when the question is what it costs to serve.

`D-076`

## B. Training raised capability faster than reliability

The headline, and the second half matters more than the first.

| Metric | Untrained | Fine-tuned |
| :-- | --: | --: |
| Solves it at least once in 4 | 0.353 | 0.627 – 0.680 |
| Solves it 4 times out of 4 | 0.247 | 0.393 – 0.460 |
| Solved *sometimes* but not always | 0.107 | 0.213 – 0.287 |

Training was supposed to close the reliability gap. It widened it. Reported
loosely this is about +33 points; reported strictly it is +15.

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

The trained model also spends about 0.78× the tokens of the untrained one.

The more useful number is what the spread says about the experiment. Between runs
the standard deviation is 0.019; within a single run the confidence interval is
about 0.069 either side. Training is steadier than 150 test tasks can measure, so
the way to sharpen this result is a bigger test split, not more seeds.

Dev told a slightly different story from test. A project that ran once and
happened to draw the third seed would have reported a better number, with nothing
in that single run to say so.

`D-074` · `D-075`

## D2. The capability transferred; the judgement did not

Everything above was measured on the task the model was trained for. To tell a
general improvement from a narrow one, the checkpoints were run on a second
environment: an order-support agent, three unseen tools, no arithmetic. Half the
requests should be completed, half refused.

| | Untrained | After SFT | After GRPO |
| :-- | --: | --: | --: |
| `pass^1` | 0.493 | 0.528 | 0.542 |
| Completes a legitimate request | 0.000 | 0.947 | 0.957 |
| Correctly refuses an unverified one | 1.000 | 0.098 | 0.115 |
| Writes without the right to | 0.000 | 0.920 | 0.918 |
| Mean reward | +0.286 | -0.394 | -0.387 |

The headline moves by three points. Underneath, the two halves swap places.

Fine-tuning on GSM8K and a calculator taught the model to operate three tools it
had never seen, in a domain with no maths in it, from a standing start of zero.
That is real transfer and it is the best news in this project.

It also taught it to act unconditionally. On requests it should refuse it writes
anyway nine times in ten, having called the verification tool, received
`authenticated: false`, and proceeded regardless. It never calls the lookup tool
at all. The untrained model scores about half by never acting; the trained one
scores about half by always acting.

Mean reward is the summary that matters: negative after training, positive
before. In an environment with something to protect, the fine-tuned model is
worse than the one that does nothing.

None of this is mysterious. Every training example was one call and done, and
not one had "do not call the tool" as the correct answer, so an unconditional
policy fits the data perfectly. It is finding H arriving in a different form: an
environment that ends the episode on the first successful call cannot teach a
model to read a result and decide what to do about it.

Measured on the 150-task transfer split, four attempts per task, in audit mode
so an unauthorised write lands and can be counted. No decision entry yet.

---

# Things that did not work

## E. Reinforcement learning added nothing after SFT

| Learning rate | `pass^1` change | 95% interval | Weights moved |
| :-- | --: | :--: | --: |
| 1e-6 | +0.002 | −0.010 – 0.013 | 0.41% |
| 1e-5 | +0.010 | −0.020 – 0.040 | 3.82% |

The interval on the first run excludes an effect larger than about a point,
rather than merely failing to find one.

The obvious objection was that the run barely moved the model, and that objection
was correct. Rerunning at ten times the rate moved the weights nearly ten times as
far and produced an identical dev peak. Two nulls across a tenfold rate range are
much harder to dismiss than one.

**Why:**

| Cause | Measurement |
| :-- | :-- |
| Nearly a quarter of steps carried no gradient | All 8 attempts scored alike, and a group-relative advantage is zero there |
| Only one reward term varied | Accuracy spread 0.339, format 0.006, efficiency 0.002, gate exactly 0.000 |
| Nothing left to reach | SFT had already removed every failure a preference signal can fix |

The gate term reads 0.000 for a structural reason: this environment has one
harmless tool, so no gate can ever fire.

One direction worth chasing, not yet a result: the higher rate raised `pass^4` and
lowered `pass@4`, narrowing the sometimes-solved band from 0.213 to 0.167. That is
what a policy-gradient method concentrating probability mass looks like, and it is
the trade this project cares about. A paired test gives p = 0.24, so it is a hint.

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

| Model | Rate |
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
| Does the 4B benefit from training too? | Not run. It was used as the teacher, so only the 1.7B has a trained arm. `D-072` |
| Did training damage anything off-task? | Not measured. Nine tasks regressed and two now score zero, but there is no before/after outside this task. |
| Would GRPO work with dead groups filtered out? | Not tried. The null above is a statement about this budget and this setup, not about the method. |
| Can a gate-bearing environment be built? | Deferred to a later stage. Until then a quarter of the reward surface is inert. |
