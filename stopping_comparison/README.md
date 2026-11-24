# Call budgets and early stopping for action counting

This post hoc comparison uses the same saved reports as the original stopping analysis. The investigators selected these simple comparison policies after seeing the original results. They used the same policies on each cohort and candidate domain. Every analysis reuses the saved reports under one policy definition per comparison.

## Inputs and outcomes

Each question has four ordered slots, U1, U2, U3 and C4. A policy that requests a slot pays one call even when the model returns no usable count. After its last request, the policy scores every candidate using the usable reports it has seen. Report `r` contributes `-2*abs(r-c) + int(r==c)` to candidate `c`. A missing report contributes zero. The smallest maximizing candidate wins. A prefix with no usable reports yields no answer.

The fixed candidate domain contains counts 1 through 5. The offered domain contains the numeric choices for the question. The reference for each policy is the four-slot result in its own domain. Agreement counts two unanswered results as equal. Accuracy uses all questions in the cohort, including unanswered questions. The item table retains the predicted count, the reference count and the gold count so agreement and correctness can be examined separately.

The cohorts are the original 206 MLVU development questions, their 103-question held-out half, 60 official test questions with DeepSeek responses, the same 60 questions with Qwen responses, and a 32-question Qwen panel drawn from the development holdout. The overlapping groups are reported separately.

## Policies

| Policy | Requests |
| --- | --- |
| Fixed 1, 2, 3 or 4 | Request that many slots in order. |
| First usable | Request slots in order until one yields a count or all four have been tried. Use that count, clipped to 1 through 5 in the fixed domain. |
| Three agree | Stop after slot 3 if all three values are usable and identical. Otherwise request slot 4. |
| Two agree | Stop after slot 2 if both values are usable and identical. Otherwise request both remaining slots. |
| Exact | Stop at the first prefix for which every admissible completion yields the same four-slot decision. |

Three equal usable reports guarantee the four-slot decision in both candidate domains used here. Let their common count be `x` and its closest candidate be `c`. The offered domain contains `x`, and the fixed domain clips `x` to the interval endpoints if needed. Against any rival candidate at distance `d` from `c`, three copies give `c` a lead of at least `6*d`. The final report can reduce that lead by at most `2*d + 1`. As `d` is a positive integer, the lead remains positive.

Two equal usable reports define an empirical heuristic. For example, with options 1, 2, 3 and 4, the sequence `2, 2, 1, 1` yields 2 after the second report and 1 after the fourth in both domains. The item table lists every observed disagreement for this policy.

## Fixed-domain replay

| Cohort | Policy | Calls | Same four-slot decisions | Correct |
| --- | --- | ---: | ---: | ---: |
| Held-out development, 103 | First usable | 103 | 83 | 79 |
| Held-out development, 103 | Fixed 3 | 309 | 95 | 81 |
| Held-out development, 103 | Three agree | 345 | 103 | 79 |
| Held-out development, 103 | Two agree | 256 | 100 | 80 |
| Held-out development, 103 | Exact | 292 | 103 | 79 |
| Official DeepSeek test, 60 | First usable | 75 | 55 | 31 |
| Official DeepSeek test, 60 | Fixed 3 | 180 | 56 | 31 |
| Official DeepSeek test, 60 | Three agree | 208 | 60 | 33 |
| Official DeepSeek test, 60 | Two agree | 162 | 60 | 33 |
| Official DeepSeek test, 60 | Exact | 180 | 60 | 33 |

On the official DeepSeek test, the four-slot procedure returned a count on 59 questions. One question had four unusable reports. The exact rule reproduced both the 59 counts and this unanswered result. On the two Qwen cohorts, a single report reproduced the four-slot decision for every question under either domain. With a single report, the fixed-domain prediction is its value clipped to 1 through 5 and the offered-domain prediction is the report itself.

### Which questions stop earlier?

On held-out development, exact stopping used fewer calls than two-agree on 12 questions, the same number on 50 and more on 41. On the official test, those counts were 8, 26 and 26. Relative to three-agree, exact stopping saved calls on 50 development and 28 test questions and tied on every other question. Two-agree changed three development decisions, all after slot 2, and matched all 60 test decisions. These comparisons use the fixed candidate range.

On the official test, the first usable policy reached its final slot after one, two, three or four calls on 51, 4, 4 and 1 questions, respectively. It answered 59 of 60. Using just the first report answered 51. The raw first report matched 41 four-slot decisions and got 33 questions correct. Clipping the first report to 1 through 5 gave 47 matches and 29 correct. On held-out development, each question had a valid first response. Both single-report rules and first usable answered all 103, matched 83 four-slot decisions and got 79 correct. All denominators include unanswered questions. [Paired results and stopping slots](paired_stops.json)

Run the following commands from the package root. Add `--domain offered` to score against the offered-option four-slot reference.

```sh
python -B stopping_comparison/paired_stops.py --root . --out replay_paired_stops.json
cmp replay_paired_stops.json stopping_comparison/paired_stops.json
```

### Correctness of changed fixed-three decisions

On the official test, the full fixed-domain count corrects three fixed-three errors and introduces one. On held-out development videos the full count corrects three and introduces five. The directions explain the respective 31 versus 33 and 81 versus 79 correct totals.

| Official test item | Reference | Three reports | Four reports | Four-report outcome |
| --- | ---: | ---: | ---: | --- |
| AC_6 | 2 | 1 | 2 | Corrected |
| AC_30 | 1 | 2 | 1 | Corrected |
| AC_52 | 2 | 2 | 1 | Introduced error |
| AC_56 | 4 | 2 | 4 | Corrected |

The original [item table](items.csv) retains these decisions and their annotations.

## Reproduce the comparison

Python and NumPy are sufficient. From the extracted anonymous package root, run the command below with the comparison script from this directory.

```sh
python -B stopping_comparison/compare.py --package . --out-dir replay_comparison
cmp replay_comparison/items.csv stopping_comparison/items.csv
cmp replay_comparison/results.json stopping_comparison/results.json
```

The commands above run from the repository root. The scorer verifies every archived exact-stopping item and every original summary field in both domains, including file digests and the two complete-response test subsets.

The [item table](items.csv) has one row per domain, recorded question and policy. Held-out development rows carry the split label `dev_held_out` and also contribute to `development_all`. The [results file](results.json) provides all seven policies, both domains and all five cohorts. It records total requests, total usable reports among those requests, matching and differing four-slot decisions, questions answered, questions correct, and accuracy on all questions.

## Six groups of frame samples

The same 32 held-out videos each have six groups of four separately sampled inputs per model. Within a group the prefix order is the original random call 1, 2, 3, 4. DeepSeek uses 128 selected frames per slot. Qwen uses 16 positions sampled from each corresponding sequence. Each model therefore contributes 192 video-group records on 32 distinct videos. Across-group totals count repeated observations of those videos.

| Group | DeepSeek exact calls | Same decisions | Correct | Qwen exact calls | Same decisions | Correct |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 94 | 32 | 18 | 96 | 32 | 10 |
| 2 | 99 | 32 | 22 | 96 | 32 | 12 |
| 3 | 96 | 32 | 20 | 95 | 32 | 14 |
| 4 | 98 | 32 | 20 | 97 | 32 | 12 |
| 5 | 90 | 32 | 20 | 94 | 32 | 14 |
| 6 | 96 | 32 | 20 | 95 | 32 | 13 |

This table uses the fixed domain. The exact rule used 573 of 768 planned slots for each model. Across the same groups, three fixed calls matched 177 of 192 four-slot decisions for DeepSeek and 180 of 192 for Qwen. The two-agree rule used 562 and 458 calls and changed four and three decisions, respectively. With the offered domain, exact stopping reproduced every decision with 619 calls for DeepSeek and 597 for Qwen.

From the extracted package root, the following command reproduces the [per-group results](seed_groups.json) and [item-level records](seed_items.csv). It also checks each fixed-domain group's four-slot accuracy, answer yield and number of usable reports against the original group ledger analysis.

```sh
python -B stopping_comparison/seed_groups.py --package . --out-dir replay_comparison
cmp replay_comparison/seed_groups.json stopping_comparison/seed_groups.json
cmp replay_comparison/seed_items.csv stopping_comparison/seed_items.csv
```

### Accuracy across the six frame draws

Each four-call group sampled four sets of 128 frames from the same 256-frame pool. DeepSeek returned 637 of 768 counts. It produced 124 empty final answers and exhausted transport retries on seven slots. All 192 video-group pairs had at least one usable report, and 105 had all four. Across all 32 videos, the mean single-call accuracy was 52.60%. Requiring four valid reports gave 40.10% (difference −12.50 percentage points, group SD 1.91). Scoring all available reports gave 62.50% (difference +9.90 points, SD 3.75, range +4.69 to +15.63). The single-call baseline averages the accuracy of the four constituent calls. [DeepSeek group results](../seed_replication/main_results.json)

Eight videos yielded four valid DeepSeek counts in every group. On these videos, mean four-report accuracy was 97.92% and mean single-call accuracy was 90.10%. The gain was 7.81 points (paired 95% CI +0.52 to +17.19, group SD 4.74, range +3.13 to +15.63). A post hoc comparison on these same eight videos gave a gain of 2.08 points when single reports were clipped to 1 through 5 and a difference of −0.52 points when both methods used the offered options. [DeepSeek group results](../seed_replication/main_results.json) · [Candidate-range comparisons](../api_range_sensitivity/results.json)

Qwen used 16 evenly spaced positions from each corresponding 128-frame sample and returned 754 of 768 counts. On the 29 videos with four valid reports in every group, mean four-report accuracy was 40.8%, against 38.5% for the mean single call (difference +2.30 points, paired 95% CI −0.86 to +5.75, group SD 2.76, range 0 to +6.90). On eight of those 29 videos, the four-report estimate changed across groups. Across all 32 videos, requiring four valid reports gave a mean difference of +1.82 points (range −2.34 to +5.47). [Qwen group results](../open_weight_seed/results.json)

### Eight calls on the same frame draws

The eight-call grouping was chosen after data collection. For each video, we concatenated groups 1 and 2, 3 and 4, and 5 and 6, preserving call order within each group. This yields three sequences per video and 96 per model using the same 768 archived calls on 32 distinct videos. Each policy is compared with the eight-slot decision in its own candidate domain.

| Fixed-domain policy | DeepSeek calls | Matches out of 96 | Qwen calls | Matches out of 96 |
| --- | ---: | ---: | ---: | ---: |
| All eight | 768 | 96 | 768 | 96 |
| Exact | 551 | 96 | 514 | 96 |
| Two agree | 432 | 92 | 294 | 92 |
| Three agree | 608 | 96 | 393 | 94 |

With the offered options, exact stopping preserved all 96 decisions in 576 DeepSeek calls and 524 Qwen calls. On Qwen video `count_9`, the eight reports were 4, 4, 4, 2, 2, 4, 2 and 2. Three-agree returned 4. Four votes for each count gave equal final scores, and the tie rule selected 2, the annotation. [Results for both domains and every changed agreement-rule decision](seed_horizons.json)

The replay checks the archived four-call group summaries before evaluating eight calls. At every observed prefix through the exact stop, an independent batch scorer checks whether any admissible completion changes the decision. The scores add over reports, so enumeration over multisets of offered values and blanks covers every possible order. The script requires a counterexample at every rejected prefix and invariance at the selected prefix. Run it from the package root.

```sh
python -B stopping_comparison/seed_horizons.py --package . --out replay_seed_horizons.json
cmp replay_seed_horizons.json stopping_comparison/seed_horizons.json
```

## Official API token usage

The [token replay](token_results.json) joins each of the 240 original API usage records to its question, slot and frozen request digest. It sums recorded prompt and completion tokens over the slots that each policy retains, including unusable model answers. Across all four slots the API recorded 8,602,600 tokens. In the fixed domain exact stopping retains 6,474,459 tokens (24.74% fewer) and fixed three retains 6,436,394 (25.18% fewer). They retain 180 calls apiece and use different subsets of the original responses. These are retrospective token totals. Deployment latency and financial charges require separate measurements.

```sh
python -B stopping_comparison/token_replay.py --package . --items stopping_comparison/items.csv --usage stopping_usage/official_api_usage.jsonl --out replay_comparison/token_results.json
cmp replay_comparison/token_results.json stopping_comparison/token_results.json
```

The API separately recorded `usage.completion_tokens_details.reasoning_tokens` for every slot. The [reasoning ledger](../stopping_usage/reasoning_usage.jsonl) retains all 240 counters and frozen request digests, with [source checksums](../stopping_usage/reasoning_provenance.json). These tokens are a subset of completion tokens. Exact stopping retains 697,054 of 899,398 reasoning tokens (22.50% fewer); fixed three retains 658,986 (26.73% fewer). The [reasoning replay](reasoning_results.json) covers all seven policies and both candidate domains using the same retained slots as the total-token replay.

```sh
python -B stopping_comparison/reasoning_replay.py --package . --out replay_comparison/reasoning_results.json
cmp replay_comparison/reasoning_results.json stopping_comparison/reasoning_results.json
```
