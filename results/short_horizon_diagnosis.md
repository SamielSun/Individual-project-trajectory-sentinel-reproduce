# Short-horizon diagnosis: washout and CUSUM slack

**SUPPORTING / diagnostic evidence, on development data.** Every row refits the whole per-deployment calibration at that setting, on the same healthy fold, at the same 5% budget. Shipped setting is washout 3, k 0.5. Nothing here selects a value: a setting chosen because it maximises a column of this table would be tuned on the corpus it is measured on.

## autogen

| washout | k | theta | detection | healthy FA | injected episodes scoring 0 |
|---|---|---|---|---|---|
| 0 | 0.0 | refused | -- | -- | matmul: Input operand 1 has a mismatch in its core dimension 0, with gufunc signature (n?,k),(k,m?)->(n?,m?) (size 4 is  |
| 0 | 0.1 | refused | -- | -- | matmul: Input operand 1 has a mismatch in its core dimension 0, with gufunc signature (n?,k),(k,m?)->(n?,m?) (size 4 is  |
| 0 | 0.25 | refused | -- | -- | matmul: Input operand 1 has a mismatch in its core dimension 0, with gufunc signature (n?,k),(k,m?)->(n?,m?) (size 4 is  |
| 0 | 0.5 | refused | -- | -- | matmul: Input operand 1 has a mismatch in its core dimension 0, with gufunc signature (n?,k),(k,m?)->(n?,m?) (size 4 is  |
| 1 | 0.0 | 39.41 | 0.783 | 0.083 | 0 |
| 1 | 0.1 | 39.11 | 0.783 | 0.083 | 0 |
| 1 | 0.25 | 38.66 | 0.783 | 0.083 | 1 |
| 1 | 0.5 | 37.91 | 0.783 | 0.083 | 2 |
| 2 | 0.0 | 33.58 | 0.870 | 0.083 | 0 |
| 2 | 0.1 | 33.48 | 0.870 | 0.083 | 1 |
| 2 | 0.25 | 33.33 | 0.870 | 0.083 | 2 |
| 2 | 0.5 | 33.08 | 0.870 | 0.083 | 2 |
| 3 | 0.0 | 9.68 | 0.913 | 0.125 | 2 |
| 3 | 0.1 | 8.88 | 0.913 | 0.125 | 2 |
| 3 | 0.25 | 7.72 | 0.913 | 0.125 | 2 |
| 3 | 0.5 | 5.97 | 0.913 | 0.125 | 2 |

## langgraph

| washout | k | theta | detection | healthy FA | injected episodes scoring 0 |
|---|---|---|---|---|---|
| 0 | 0.0 | refused | -- | -- | matmul: Input operand 1 has a mismatch in its core dimension 0, with gufunc signature (n?,k),(k,m?)->(n?,m?) (size 7 is  |
| 0 | 0.1 | refused | -- | -- | matmul: Input operand 1 has a mismatch in its core dimension 0, with gufunc signature (n?,k),(k,m?)->(n?,m?) (size 7 is  |
| 0 | 0.25 | refused | -- | -- | matmul: Input operand 1 has a mismatch in its core dimension 0, with gufunc signature (n?,k),(k,m?)->(n?,m?) (size 7 is  |
| 0 | 0.5 | refused | -- | -- | matmul: Input operand 1 has a mismatch in its core dimension 0, with gufunc signature (n?,k),(k,m?)->(n?,m?) (size 7 is  |
| 1 | 0.0 | 280.99 | 0.667 | 0.042 | 0 |
| 1 | 0.1 | 280.39 | 0.667 | 0.042 | 0 |
| 1 | 0.25 | 279.49 | 0.667 | 0.042 | 0 |
| 1 | 0.5 | 277.99 | 0.667 | 0.042 | 0 |
| 2 | 0.0 | 282.03 | 0.667 | 0.042 | 0 |
| 2 | 0.1 | 281.53 | 0.667 | 0.042 | 0 |
| 2 | 0.25 | 280.78 | 0.667 | 0.042 | 0 |
| 2 | 0.5 | 279.53 | 0.667 | 0.042 | 0 |
| 3 | 0.0 | 193.86 | 0.625 | 0.000 | 0 |
| 3 | 0.1 | 193.46 | 0.625 | 0.000 | 0 |
| 3 | 0.25 | 192.86 | 0.625 | 0.000 | 0 |
| 3 | 0.5 | 191.86 | 0.625 | 0.000 | 0 |

## Per-episode: every short-horizon injected episode

Selected by trajectory shape alone -- at most 3 steps between onset and the end of the episode -- so the same rule picks the same episodes in any corpus. Aggregate detection can hold still while individual episodes move in opposite directions, so the washout question is settled here rather than in the tables above.

| framework | episode | class | T | tau | washout | causal post-onset steps | first scored | max CUSUM | theta | alarm | detected |
|---|---|---|---|---|---|---|---|---|---|---|---|
| autogen | autogen-real-context_corruption-000 | context_corruption | 4 | 2 | 1 | 2 | 1 | 40.03 | 37.91 | 3 | yes |
| autogen | autogen-real-context_corruption-000 | context_corruption | 4 | 2 | 2 | 2 | 2 | 52.93 | 33.08 | 3 | yes |
| autogen | autogen-real-context_corruption-000 | context_corruption | 4 | 2 | 3 | 1 | 3 | 54.52 | 5.97 | 3 | yes |
| autogen | autogen-real-context_corruption-002 | context_corruption | 4 | 2 | 1 | 2 | 1 | 32.38 | 37.91 | - | no |
| autogen | autogen-real-context_corruption-002 | context_corruption | 4 | 2 | 2 | 2 | 2 | 39.79 | 33.08 | 3 | yes |
| autogen | autogen-real-context_corruption-002 | context_corruption | 4 | 2 | 3 | 1 | 3 | 36.93 | 5.97 | 3 | yes |
| autogen | autogen-real-context_corruption-004 | context_corruption | 4 | 2 | 1 | 2 | 1 | 58.31 | 37.91 | 3 | yes |
| autogen | autogen-real-context_corruption-004 | context_corruption | 4 | 2 | 2 | 2 | 2 | 57.29 | 33.08 | 3 | yes |
| autogen | autogen-real-context_corruption-004 | context_corruption | 4 | 2 | 3 | 1 | 3 | 39.84 | 5.97 | 3 | yes |
| autogen | autogen-real-goal_drift-000 | goal_drift | 5 | 3 | 1 | 2 | 1 | 23.97 | 37.91 | - | no |
| autogen | autogen-real-goal_drift-000 | goal_drift | 5 | 3 | 2 | 2 | 2 | 16.55 | 33.08 | - | no |
| autogen | autogen-real-goal_drift-000 | goal_drift | 5 | 3 | 3 | 2 | 3 | 11.29 | 5.97 | 3 | yes |
| autogen | autogen-real-goal_drift-002 | goal_drift | 4 | 2 | 1 | 2 | 1 | 0.00 | 37.91 | - | no |
| autogen | autogen-real-goal_drift-002 | goal_drift | 4 | 2 | 2 | 2 | 2 | 0.00 | 33.08 | - | no |
| autogen | autogen-real-goal_drift-002 | goal_drift | 4 | 2 | 3 | 1 | 3 | 0.00 | 5.97 | - | no |
| autogen | autogen-real-goal_drift-004 | goal_drift | 4 | 2 | 1 | 2 | 1 | 0.00 | 37.91 | - | no |
| autogen | autogen-real-goal_drift-004 | goal_drift | 4 | 2 | 2 | 2 | 2 | 0.00 | 33.08 | - | no |
| autogen | autogen-real-goal_drift-004 | goal_drift | 4 | 2 | 3 | 1 | 3 | 0.00 | 5.97 | - | no |
| autogen | autogen-real-goal_drift-005 | goal_drift | 5 | 2 | 1 | 3 | 1 | 83.29 | 37.91 | 4 | yes |
| autogen | autogen-real-goal_drift-005 | goal_drift | 5 | 2 | 2 | 3 | 2 | 94.63 | 33.08 | 3 | yes |
| autogen | autogen-real-goal_drift-005 | goal_drift | 5 | 2 | 3 | 2 | 3 | 76.82 | 5.97 | 3 | yes |
| autogen | autogen-real-looping-004 | looping | 5 | 2 | 1 | 3 | 1 | 222.36 | 37.91 | 3 | yes |
| autogen | autogen-real-looping-004 | looping | 5 | 2 | 2 | 3 | 2 | 200.52 | 33.08 | 3 | yes |
| autogen | autogen-real-looping-004 | looping | 5 | 2 | 3 | 2 | 3 | 183.40 | 5.97 | 3 | yes |
| autogen | autogen-real-tool_cascade-000 | tool_cascade | 5 | 3 | 1 | 2 | 1 | 35.64 | 37.91 | - | no |
| autogen | autogen-real-tool_cascade-000 | tool_cascade | 5 | 3 | 2 | 2 | 2 | 41.04 | 33.08 | 4 | yes |
| autogen | autogen-real-tool_cascade-000 | tool_cascade | 5 | 3 | 3 | 2 | 3 | 31.15 | 5.97 | 4 | yes |
| autogen | autogen-real-tool_cascade-001 | tool_cascade | 6 | 4 | 1 | 2 | 1 | 49.99 | 37.91 | 5 | yes |
| autogen | autogen-real-tool_cascade-001 | tool_cascade | 6 | 4 | 2 | 2 | 2 | 44.65 | 33.08 | 5 | yes |
| autogen | autogen-real-tool_cascade-001 | tool_cascade | 6 | 4 | 3 | 2 | 3 | 40.72 | 5.97 | 5 | yes |
| autogen | autogen-real-tool_cascade-004 | tool_cascade | 5 | 2 | 1 | 3 | 1 | 52.00 | 37.91 | 4 | yes |
| autogen | autogen-real-tool_cascade-004 | tool_cascade | 5 | 2 | 2 | 3 | 2 | 53.85 | 33.08 | 3 | yes |
| autogen | autogen-real-tool_cascade-004 | tool_cascade | 5 | 2 | 3 | 2 | 3 | 46.30 | 5.97 | 3 | yes |
| langgraph | langgraph-real-context_corruption-002 | context_corruption | 5 | 3 | 1 | 2 | 1 | 200.55 | 277.99 | - | no |
| langgraph | langgraph-real-context_corruption-002 | context_corruption | 5 | 3 | 2 | 2 | 2 | 187.29 | 279.53 | - | no |
| langgraph | langgraph-real-context_corruption-002 | context_corruption | 5 | 3 | 3 | 2 | 3 | 104.13 | 191.86 | - | no |
| langgraph | langgraph-real-goal_drift-000 | goal_drift | 4 | 2 | 1 | 2 | 1 | 11.51 | 277.99 | - | no |
| langgraph | langgraph-real-goal_drift-000 | goal_drift | 4 | 2 | 2 | 2 | 2 | 8.37 | 279.53 | - | no |
| langgraph | langgraph-real-goal_drift-000 | goal_drift | 4 | 2 | 3 | 1 | 3 | 4.75 | 191.86 | - | no |
| langgraph | langgraph-real-tool_cascade-000 | tool_cascade | 4 | 2 | 1 | 2 | 1 | 119.50 | 277.99 | - | no |
| langgraph | langgraph-real-tool_cascade-000 | tool_cascade | 4 | 2 | 2 | 2 | 2 | 113.43 | 279.53 | - | no |
| langgraph | langgraph-real-tool_cascade-000 | tool_cascade | 4 | 2 | 3 | 1 | 3 | 75.74 | 191.86 | - | no |
| langgraph | langgraph-real-tool_cascade-002 | tool_cascade | 5 | 2 | 1 | 3 | 1 | 245.29 | 277.99 | - | no |
| langgraph | langgraph-real-tool_cascade-002 | tool_cascade | 5 | 2 | 2 | 3 | 2 | 251.75 | 279.53 | - | no |
| langgraph | langgraph-real-tool_cascade-002 | tool_cascade | 5 | 2 | 3 | 2 | 3 | 165.98 | 191.86 | - | no |
| langgraph | langgraph-real-tool_cascade-003 | tool_cascade | 7 | 5 | 1 | 2 | 1 | 153.75 | 277.99 | - | no |
| langgraph | langgraph-real-tool_cascade-003 | tool_cascade | 7 | 5 | 2 | 2 | 2 | 152.32 | 279.53 | - | no |
| langgraph | langgraph-real-tool_cascade-003 | tool_cascade | 7 | 5 | 3 | 2 | 3 | 92.05 | 191.86 | - | no |
| langgraph | langgraph-real-tool_cascade-004 | tool_cascade | 6 | 3 | 1 | 3 | 1 | 490.87 | 277.99 | 4 | yes |
| langgraph | langgraph-real-tool_cascade-004 | tool_cascade | 6 | 3 | 2 | 3 | 2 | 495.41 | 279.53 | 4 | yes |
| langgraph | langgraph-real-tool_cascade-004 | tool_cascade | 6 | 3 | 3 | 3 | 3 | 279.63 | 191.86 | 5 | yes |

### Verdict per episode the shipped washout misses

`margin ratio` is the decision statistic over that washout's own threshold; 1.0 is an alarm. It is the ratio and not the raw score that decides, because a shorter washout raises both.

| episode | T | tau | ratio @w1 | ratio @w2 | ratio @w3 | any washout detects |
|---|---|---|---|---|---|---|
| autogen-real-goal_drift-002 | 4 | 2 | 0.00 | 0.00 | 0.00 | no |
| autogen-real-goal_drift-004 | 4 | 2 | 0.00 | 0.00 | 0.00 | no |
| langgraph-real-context_corruption-002 | 5 | 3 | 0.72 | 0.67 | 0.54 | no |
| langgraph-real-goal_drift-000 | 4 | 2 | 0.04 | 0.03 | 0.02 | no |
| langgraph-real-tool_cascade-000 | 4 | 2 | 0.43 | 0.41 | 0.39 | no |
| langgraph-real-tool_cascade-002 | 5 | 2 | 0.88 | 0.90 | 0.87 | no |
| langgraph-real-tool_cascade-003 | 7 | 5 | 0.55 | 0.54 | 0.48 | no |

Short-horizon episodes the shipped washout misses that some other washout recovers: **0** (none).
Short-horizon episodes the shipped washout detects that washout 1 loses: **3** (autogen-real-context_corruption-002, autogen-real-goal_drift-000, autogen-real-tool_cascade-000).


Per-step channel streams for every row above are in `results/tables/short_horizon_per_episode.csv` (`e_cusum_by_step`, `m_cusum_by_step`, `fused_cusum_by_step`, and the raw increments behind them).
