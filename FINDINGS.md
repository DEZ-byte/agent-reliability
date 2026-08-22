# What this project actually found

Nine things worth knowing, in the order they were learned, including the ones
that went badly.

The `D-0xx` tags name entries in the project's decision log, which holds the
full reasoning behind each. That log is append-only and runs to 74 entries, so
it is kept outside this repository for now. This page is the readable version.

---

### A trained 1.7B beat a scaffolded 8B, and cost a third as much

The comparison the project was built to make. An untrained Llama-3.1-8B with
retry scaffolding scored `pass^1` 0.415 and `pass^4` 0.293. The fine-tuned
Qwen3-1.7B scored 0.517 to 0.553 and 0.393 to 0.460 across three runs. Every
paired interval excludes zero.

Cost went the same way once measured properly. The 8B emits fewer tokens per
task, but weighting by parameters it costs 236 billion-parameter-tokens against
72, and needs roughly 6 GB to serve against 1.5 GB. Compared on raw token counts
the conclusion would have reversed.

`D-076`

### The 8B's first score was 0.000, and it was my bug

Llama writes tool calls as bare JSON with `parameters`. Qwen wraps them in
`<tool_call>` tags with `arguments`. The grader understood one of them, so a
perfectly correct Llama call scored zero and the headline would have read as a
total capability failure.

The fix is per-model and off by default, which matters more than it sounds.
Applied globally it would have accepted three recorded completions where an
untrained Qwen wrote bare JSON. By Qwen's own template that is a real format
failure, and accepting it would have improved the baseline and shrunk every
gain measured against it.

`D-076` · `D-060`

### Tool formatting was never the problem

Both models emitted schema-valid tool calls essentially every time. The schema
failure rate measured 0.00%. Roughly 85–91% of failures were a perfectly
well-formed call that computed the wrong thing.

That killed a planned mitigation: mixing in an external function-calling dataset
to teach tool formatting. There was nothing to teach.

`D-068` ·
`D-070`

### The retry rung had almost nothing to fix

Giving the model a second attempt after a failure only helps when the failure is
visible at runtime. Here it usually wasn't: a wrong answer still executes fine,
so the loop ends and nobody objects. The second attempt fired on 3.7% of
episodes for one model and 10.3% for the other.

After training, retry stopped making any difference at all. Its `pass^4` is
identical to the single-attempt rung.

`D-068`

### Phase A can't teach self-correction, and that's structural

The environment ends an episode as soon as a tool call succeeds, whether or not
the answer is right. So the dominant failure never gets a second look.

It can't be fixed by changing the loop either. Telling the model its answer is
wrong would leak the grader into the rollout. Measured yield for genuine
recovery trajectories: one task in a hundred. Self-correction moved to a later
stage where tool errors are actually observable.

`D-069`

### A model can pass an execution-backed grader without computing anything

Grading from executed results is supposed to stop a model reciting a memorised
answer. It doesn't. `calculator("391")` scores correct having computed nothing,
and the obvious fix (require the expression to do arithmetic) is defeated by
writing `391 + 0`.

Measured at 3.0% before training and 1.0% after, under a filter that catches the
decorated form.

`D-062`

### The anti-cheating filter had to be measured before it was trusted

One rule required the arithmetic to use numbers that appear in the question.
It rejected correct work, because GSM8K writes quantities as words. "Three dozen
eggs for her four children" contains no digits, so the correct `36 / 4` looks
invented.

It was rejecting genuine multi-step reasoning and catching nothing the other
rules missed, so it's off. Measuring first is what caught it.

`D-071`

### Training raised capability and widened the reliability gap

This is the headline, and the second half matters more than the first.

`pass@4` nearly doubled. `pass^4`, where every attempt has to succeed, rose less
than half as much. The band of tasks solved sometimes but not always went from
0.107 to 0.287.

Reported as `pass@4` this is +33 points. Reported strictly it's +15.

`D-073`

### It reproduced three times, and it was cheaper

Three training runs, each varying initialisation and data order, landed at
+0.222, +0.212 and +0.248 on `pass^1`. The trained model also spends 0.78× the
tokens of the untrained one.

The more useful number is what the spread says about the experiment. Between
runs the standard deviation is 0.019; within a single run the confidence
interval is about 0.069 either side. The training is steadier than 150 test
tasks can measure, so the way to sharpen this result is a bigger test split,
not more seeds.

Dev told a slightly different story: peaks of 0.4725, 0.4700 and 0.4975. A
project that ran once and happened to draw the third seed would have reported a
better number, with nothing in that single run to say so.

`D-074` · `D-075`

### What improved was tool use, not arithmetic

Probed with no calculator, the model scored 64.0% before training and 66.0%
after. It didn't get better at maths. It got better at writing the expression.

Worth stating because "fine-tuning improved accuracy" invites the wrong reading.

`D-074` ·
`D-064`

### Two bugs that only a green CI run could find

A lone surrogate character compiled into a docstring, and line-ending
translation quietly breaking every recorded content hash on any machine other
than the one that wrote it. The local test suite passed through both.

Also here: the model bundle was chosen on licence rather than on measurement,
and the record says so, including which technical check the chosen bundle fails.

`D-057` ·
`D-046`

---

### What isn't settled

The headline comparison, small model trained versus larger model scaffolded, has
not been run. The 8B comparator is registered in the config and has never been
executed.

The teacher was Qwen3-4B rather than the larger model originally planned, so
only the smaller model has a trained arm.
`D-072`
records that deviation and what it costs the claim.
