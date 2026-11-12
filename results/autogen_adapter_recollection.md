# Recollecting the AutoGen corpus under the corrected adapter

The AutoGen adapter dropped `ThoughtEvent`, so an AutoGen step carried the tool
call and its result but never the assistant's own words, while the LangGraph
adapter carried them. Every AutoGen-versus-LangGraph statement made before the
fix was therefore partly a statement about the adapter rather than about the
framework. The fix is in `derail/harness/frameworks.py`; this is what it bought.

`autogen7b_real2` is kept unchanged as pre-fix evidence. `autogen7b_real3` is a
new corpus collected under the corrected adapter with a fresh seed base
(63311), not a re-run of the same episodes — so the two are independent draws
and any difference between them confounds the adapter change with the sampling
of tasks and model behaviour.

## Causality

The fix records text that is already available when the step is created, and
does not reach back into a step that has been scored. From live event ordering:

```
ToolCallExecutionEvent    the previous step completes
ThoughtEvent              "It seems there was an error with the tool.
                           Let's try the search again."
ToolCallRequestEvent      the step that text belongs to is created
ToolCallExecutionEvent    that step's results arrive
```

Every `ThoughtEvent` immediately precedes its own `ToolCallRequestEvent`, and
none arrives after the execution event for its step. The thought and the call
come from the same model call, so both are known before the step exists. The
text cannot contain the result of the step it is attached to. Four tests in
`tests/test_telemetry_contracts.py` pin this: thought capture, no extra step,
buffer reset between turns, and that the execution branch only ever touches
`steps[-1]`.

## What changed in the corpus

| | autogen7b_real2 | autogen7b_real3 |
|---|---|---|
| manifest episodes | 143 | 144 |
| healthy, scoreable | 96 | 112 |
| unscoreable | 6 | 0 |
| injected, scoreable | 23 | 24 |
| landing failures | 24 | 10 |
| steps per episode | 6.30 | 5.61 |
| tool-call steps | 764 | 666 |
| **carrying assistant text** | **0 (0.0%)** | **114 (17.1%)** |
| mean assistant characters | 0.0 | 147.6 |

The mechanical effect is unambiguous and is the point of the fix: AutoGen steps
now carry the same kind of evidence LangGraph steps carry. The share is 17%
against LangGraph's 24% (202 of 838), so the two adapters are closer but not
identical — qwen2.5 emits a thought before only some of its tool calls under
AutoGen, which is a property of the model and the framework prompt rather than
of the adapter.

The corpus is also cleaner: no unscoreable episodes against six, and landing
failures roughly halved. That is a side effect of the fresh draw, not of the
adapter.

## What did not change: detection

Detection per episode, taken as the majority verdict over 40 replicate splits
so the comparison is between corpora rather than between thresholds. Wilson
95% intervals; Fisher exact per row, Holm-corrected within each arm.

| arm | class | real2 | real3 | Fisher p |
|---|---|---|---|---|
| e+m/max | ALL | 11/23 0.478 [0.29,0.67] | 12/24 0.500 [0.31,0.69] | 1.000 |
| e+m/max | goal_drift realized | 1/3 0.333 | 3/3 1.000 | 0.400 |
| e+m/max | healthy FA | 3/96 0.031 | 4/112 0.036 | 1.000 |
| e+m/mean | ALL | 11/23 0.478 | 11/24 0.458 | 1.000 |
| e+m/mean | healthy FA | 3/96 0.031 | 4/112 0.036 | 1.000 |
| e+m+g/mean | ALL | 11/23 0.478 | 11/24 0.458 | 1.000 |
| e+m+g/mean | tool_cascade | 2/5 0.400 | 2/6 0.333 | 1.000 |
| e+m+g/mean | healthy FA | 2/96 0.021 | 4/112 0.036 | 0.689 |

Nothing is significant, before or after Holm, in any arm. `looping` stays at
6/6 in both corpora and every arm. Overall detection moves by at most one
episode. The healthy false-alarm rate is unchanged at about 3%.

The one row that looks like a gain — realized goal drift going from 1/3 to 3/3
under `e+m/max` — rests on three episodes in each corpus and has a Fisher p of
0.40. It is not evidence of anything.

## Reading

The fix was necessary and is now correct: an AutoGen trace records what the
agent said as well as what it called, the adapters represent a turn the same
way, and the framework comparison is no longer confounded by a missing channel.

It did not buy detection. With six injected episodes per class, this comparison
can only resolve a large effect, and there is no sign of even a moderate one.
The earlier conclusion that AutoGen and LangGraph detection is identical once
post-onset runway is controlled therefore survives the fix rather than being
overturned by it — the adapter was not what made AutoGen look different.

Comparison: `_internal/deployment_healthy_collection/compare_autogen_adapter.py`;
figures in `autogen_adapter_comparison.json`.
