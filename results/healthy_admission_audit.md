# What is in the healthy null

**SUPPORTING / diagnostic evidence.** Nothing here deletes, relabels or hides an episode: every admitted run stays in the corpus and is counted below. It reports the composition of the healthy populations every per-deployment threshold is derived from, by re-running each task's own predeclared `success_fn` over the committed traces.

Collection admitted healthy runs by an outcome-independent rule (`collect_framework_real_traces.check_admission`), which cannot silently drop difficult-but-valid behaviour but also never consults the verifier -- so the corpus records `success = None` for every healthy episode.

## The two distinctions

**Verified healthy** means no injection AND the task's own verifier says the run finished the task. A run that simply failed is an unlabelled failure, not a healthy negative.

**Scoreable** means the detector had at least one valid scoring opportunity. It is a property of the detector, recorded per row as `esn_cusum_max/washout=3`: a detector that produces a score sooner makes the same episodes scoreable with no change to the admission rule. Unscoreable episodes must not enter a threshold distribution as artificial zeros, and must not pad a false-alarm denominator with runs that could never alarm.

**Monitoring coverage** = scoreable verified-healthy / all verified-healthy. Reported because it is the fraction of healthy traffic the detector can say anything about at all.

## langgraph7b_real2

- attempted: **120**
- verified healthy: **117**
- rejected healthy attempts: **3** (retained in the corpus, counted here, never used as negatives)
- scoreable verified healthy: **117**
- unscoreable verified healthy: **0**
- **monitoring coverage: 1.000**
- task mix: arxiv_paper_search 60, multi_city_weather 60
- model: qwen2.5:7b
- trajectory length: min 4, median 7, max 10

Rejection reasons:
- 3 x task verifier returned False: a run that did not finish its task is an unlabelled failure, not a healthy negative

## autogen7b_real2

- attempted: **120**
- verified healthy: **102**
- rejected healthy attempts: **18** (retained in the corpus, counted here, never used as negatives)
- scoreable verified healthy: **96**
- unscoreable verified healthy: **6**
- **monitoring coverage: 0.941**
- task mix: arxiv_paper_search 60, multi_city_weather 60
- model: qwen2.5:7b
- trajectory length: min 2, median 5, max 12

Rejection reasons:
- 18 x task verifier returned False: a run that did not finish its task is an unlabelled failure, not a healthy negative

## False-alarm denominators these numbers imply

Both are reported for every future evaluation, because they diverge and each answers a different question: operational FA over all verified-healthy episodes is what a deployment sees; scoreable FA over those the detector could act on is what the detector is responsible for.

| corpus | verified healthy (operational denominator) | scoreable (scoreable denominator) | monitoring coverage |
|---|---|---|---|
| langgraph7b_real2 | 117 | 117 | 1.000 |
| autogen7b_real2 | 102 | 96 | 0.941 |

## Is the verifier safe to admit episodes with?

| task | declared in | signature | source sha | mentions detector/label terms | tools |
|---|---|---|---|---|---|
| arxiv_paper_search | `derail/harness/tasks.py` | `(resp, steps)` | `ce08388f754124f3` | none | arxiv_search, wikipedia_search |
| multi_city_weather | `derail/harness/tasks.py` | `(resp, steps)` | `55a1f01766778521` | none | get_weather, wikipedia_search |

Determinism was re-checked by evaluating every episode twice: 240 of 240 agree. The verifier receives only the final response text and the recorded steps, and a recorded step carries no failure class, no onset and no monitor output -- so it cannot see the label or the detector it is used to evaluate.
