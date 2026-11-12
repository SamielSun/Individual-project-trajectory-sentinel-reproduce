# Why the grounding channel helps some corpora and not others

Mechanism-level diagnosis, run before proposing any further detector change.
The headline is that two different failure mechanisms were being averaged
together, and separating them changes what the numbers mean.

`langgraph7b_real2` and `autogen7b_real2` are matched by construction -- same
collector, same model (`qwen2.5:7b`), same two tasks (72/71 and 72/72), same
four failure classes, same 120 healthy attempts. A difference between them
therefore isolates the framework from the task, which is the only basis on
which anything below is called a framework effect.

## The finding that reframes the question

Detection under `e+m+g/mean`, stratified by post-onset runway -- the number of
scoreable steps at or after onset:

| corpus | runway <= 2 | runway 3-5 | runway > 5 |
|---|---|---|---|
| langgraph | 2/5 | 6/12 | **7/7** |
| autogen | 3/11 | 4/8 | **4/4** |
| long_drift | 0/1 | -- | **41/119** |

Two separate mechanisms, not one:

**Runway-bound.** On the framework corpora, detection is 100% (11 of 11) once
runway exceeds 5 steps, and falls to 27-40% at runway <= 2. Within each band
LangGraph and AutoGen detect at the same rate. **There is no framework effect
on detector quality once runway is controlled.** What differs is opportunity:
47.8% of AutoGen's injected episodes end within 2 scoreable steps of onset
against 20.8% of LangGraph's.

**Information-bound.** On the long drift corpus runway is ample -- 119 of 120
episodes exceed 5 -- and detection is still only 41/119. Runway is not the
constraint there; the signal is. This is the same conclusion the conceptor
study reached about goal drift, reached again from a different direction.

## Why AutoGen's runway is shorter, and why that is not an observability gap

On `arxiv_paper_search`, matched task and model:

| framework | median T | tool calls/ep | non-tool steps/ep |
|---|---|---|---|
| LangGraph | 6 | 2.12 | 3.65 |
| AutoGen | 4 | 2.75 | 1.00 |

AutoGen makes **more** tool calls and records **fewer** steps. LangGraph's
StateGraph externalises the model's reasoning as separate scoreable events --
those steps average 300-350 characters of real agent output, not filler -- while
AutoGen's AssistantAgent folds the same reasoning into fewer messages. Both
frameworks emit the same action vocabulary and comparable totals
(838/143 vs 764/137 tool_call/synthesis events).

So a step is not the same quantity in the two frameworks, and "runway in
steps" is not comparable across them. AutoGen's runway of 3 carries more agent
work than LangGraph's 4. This is a genuine architectural difference, not a
telemetry mapping defect, and it means cross-framework detection comparisons
are confounded by the step unit unless runway is controlled.

## What the grounding channel actually does

Adding `g` under `mean`, measured as the rank correlation between the `e+m`
and `e+m+g` episode maxima, and the separation of healthy from injected:

| corpus | Spearman rho | separation `e+m` -> `e+m+g` |
|---|---|---|
| langgraph | 0.9865 | 0.875 -> 0.883 (+0.008) |
| autogen | **0.9987** | 0.880 -> 0.879 (**-0.001**) |
| long_drift | **0.4879** | 0.597 -> **0.974** (+0.376) |

The channel is close to binary in effect. Where it re-ranks episodes it is
transformative; where it does not, it is a near-monotone rescale that changes
essentially nothing.

**The AutoGen "regression" is not a regression.** At rho = 0.9987 the ordering
of episodes is virtually untouched and separation moves by one thousandth. The
-0.092 context_corruption and -0.054 goal_drift movements reported earlier were
fold noise, consistent with every one of them being non-significant under Holm.
`g` is inert on AutoGen, not harmful.

`g` re-ranks when two conditions hold together: the failure is one the channel
can see, and there is post-onset runway for it to accumulate. Its first
informative step is 1-2 steps after onset in every corpus measured, which is
why the corpus with 47.8% of episodes at runway <= 2 gets nothing from it.

## Per-mechanism classification of every AutoGen movement

| observation | classification | evidence |
|---|---|---|
| context_corruption -0.092 | fold noise, not a mechanism | non-significant (Holm 0.193); rho 0.9987 means `g` barely reorders anything |
| goal_drift -0.054 | `g` uninformative at this runway | post-onset lift 0.835 vs 1.703 (langgraph) and 21.6 (long_drift); 47.8% of episodes at runway <= 2 |
| tool_cascade 0.410 | signal present, runway-limited | `g` post-onset lift 3.484; detection is 4/4 at runway > 5 |
| aggregate -0.024 | composition, not capability | within-runway-band detection matches LangGraph exactly |

## One further mechanism, recorded because it explains the false alarms too

AutoGen's healthy `e` null is bimodal: median 0.084, p90 20.6, max 48.7, with
a std/IQR ratio of 11.5 against LangGraph's 0.9. The loud mode is healthy runs
where the agent issues repeated off-task `wikipedia_search` calls that return
arbitrary content. That is the same behaviour behind every false alarm on the
deployment corpus, where it appears in 100% of healthy weather episodes.
Normalizing by an IQR that describes only the quiet mode is what lets those
steps dominate the fusion.

Note that repetition alone does not explain the corpus ordering: repeated tool
calls appear in 54% (langgraph), 66% (autogen), 100% (long_drift) and 56%
(deployment) of healthy episodes, and the corpus with the most repetition is
the one `g` helps most. Runway, not repetition, is the discriminating variable.

## A candidate that the evidence rules out

Step-index-conditional calibration -- letting theta depend on how many steps
have been scored -- is **not** justified. The healthy null's dispersion moves
in opposite directions across corpora as the scored-step count rises:

| corpus | n=1 | n=2 | n=3 | n=4 |
|---|---|---|---|---|
| deployment | 0.932 | 1.262 | 2.006 | 2.247 |
| langgraph | 0.958 | 0.679 | 0.471 | 0.507 |
| autogen | 0.000 | 1.113 | 1.298 | 4.715 |
| long_drift | 1.779 | 2.069 | 1.292 | 0.809 |

There is no consistent direction to condition on. Fitting one would be fitting
corpus-specific structure.
