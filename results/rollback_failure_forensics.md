# Forensic accounting: every closed-loop repair that failed

**SUPPORTING / diagnostic evidence, on development data.** Derived entirely from the committed outcomes of the live rollback/retry study; no agent was re-run. The primary arm is the deployable one -- the frozen monitor's own causal alarm chooses the checkpoint. The oracle arm is given ground-truth tau and is an upper bound only.

Retry budget is `max_steps - checkpoint_k` against a global `MAX_STEPS = 12`: the continuation does NOT get a fresh budget, so a late checkpoint leaves fewer steps to finish the task than the original run had.

## Primary arm

| episode | class | tau | alarm | ck | retry budget | outcome | root cause |
|---|---|---|---|---|---|---|---|
| rollback-arxiv_paper_search-tool_cascade-00 | tool_cascade | 4 | 4 | 4 | 8 | recovered | recovered |
| rollback-arxiv_paper_search-tool_cascade-01 | tool_cascade | 2 | - | 0 | 12 | not_triggered | detection_gap_no_trigger |
| rollback-multi_city_weather-tool_cascade-00 | tool_cascade | 4 | 6 | 4 | 8 | recovered | recovered |
| rollback-multi_city_weather-tool_cascade-01 | tool_cascade | 2 | - | 0 | 12 | not_triggered | detection_gap_no_trigger |
| rollback-arxiv_paper_search-looping-00 | looping | 2 | 4 | 2 | 10 | recovered | recovered |
| rollback-arxiv_paper_search-looping-01 | looping | 3 | 4 | 3 | 9 | recovered | recovered |
| rollback-multi_city_weather-looping-00 | looping | 2 | 4 | 2 | 10 | recovered | recovered |
| rollback-multi_city_weather-looping-01 | looping | 2 | 4 | 2 | 10 | recovered | recovered |
| rollback-arxiv_paper_search-context_corruption-00 | context_corruption | 2 | - | 0 | 12 | not_triggered | detection_gap_no_trigger |
| rollback-arxiv_paper_search-context_corruption-01 | context_corruption | 2 | 7 | 7 | 5 | recovered | recovered |
| rollback-multi_city_weather-context_corruption-00 | context_corruption | 3 | - | 0 | 12 | not_triggered | detection_gap_no_trigger |
| rollback-multi_city_weather-context_corruption-01 | context_corruption | 3 | - | 0 | 12 | not_triggered | detection_gap_no_trigger |
| rollback-arxiv_paper_search-wrong_document-00 | wrong_document | 3 | 6 | 6 | 6 | still_wrong | rollback_discards_verifier_required_tool_call |
| rollback-arxiv_paper_search-wrong_document-01 | wrong_document | 3 | 11 | 11 | 1 | recovered | recovered |
| rollback-multi_city_weather-wrong_document-00 | wrong_document | 3 | - | 0 | 12 | not_triggered | detection_gap_no_trigger |
| rollback-multi_city_weather-wrong_document-01 | wrong_document | 3 | 10 | 9 | 3 | halted | retry_budget_exhausted_by_late_checkpoint |

### Primary root-cause counts

- **detection_gap_no_trigger**: 6
- **rollback_discards_verifier_required_tool_call**: 1
- **retry_budget_exhausted_by_late_checkpoint**: 1

## Oracle arm -- what remains unrepairable with a perfect trigger

| episode | class | tau | ck | outcome | root cause |
|---|---|---|---|---|---|
| rollback-arxiv_paper_search-context_corruption-00 | context_corruption | 2 | 2 | still_wrong | rollback_discards_verifier_required_tool_call |
| rollback-arxiv_paper_search-context_corruption-01 | context_corruption | 2 | 2 | still_wrong | rollback_discards_verifier_required_tool_call |

## Per-class closed-loop recovery (primary arm)

| class | recovered / n |
|---|---|
| context_corruption | 1 / 4 |
| looping | 4 / 4 |
| tool_cascade | 2 / 4 |
| wrong_document | 1 / 4 |
