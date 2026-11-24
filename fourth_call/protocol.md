# Fourth-call frame selection on MLVU Action Count

This local analysis plan was recorded before inspecting either new fourth-call arm. The original development results were already known. The experiment tests the new fourth-input contrast on those videos.

## Paired design

Use the same three archived integer reports for both arms of each question, in the order `uniform`, `random`, `recency`. All three historical selectors retained the same uniform frames. Arm A adds a new independent call on 128 uniformly spaced frames. Arm B adds a new call on 128 change-selected frames from a 256-frame pool. Use the archived sampling, image encoding, question, option order and request settings. Interleave the new A and B calls through the same configured endpoint during the same run. Record the model alias, timestamps, selected frame indices, complete final text, completion status and termination reason. Retain unsuccessful calls. A blank or unsupported final answer contributes an explicit failure status. Reasoning text cannot supply its count.

The primary cohort is the original 103 questions held out from parameter fitting. Sort the 206 filenames lexicographically and take the last 103 indices in `numpy.random.default_rng(0).permutation(206)`. The optional secondary cohort covers all 206 questions and includes the tuning half. Membership is fixed before acquisition. Do not stop on an interim effect. Historical fourth calls may be examined separately as pilot data.

## Scoring and endpoints

1. The primary outcome is exact-count accuracy with the smaller maximizing candidate selected on a tie. For each candidate 1 through 5, sum `-2 * abs(report - candidate) + int(report == candidate)`. This is the fitted score multiplied by 2.5, so integer arithmetic preserves mathematical ties. The primary contrast is B minus A. Report arm-specific accuracy and mean absolute error, paired wins and losses, the two-sided exact McNemar test and the 95% question-paired percentile bootstrap interval for the accuracy difference. Use 20,000 resamples and NumPy generator seed zero.
2. Report the larger maximizing candidate, the original left-to-right binary64 implementation, majority vote with ties settled by first report order, and the nearest-even rounded median as sensitivity analyses. Use all four reports in the same order in each arm. Compare fixed-score aggregation with vote within each arm. Also compare each four-call arm with the common three-call aggregate on the same matched questions. The primary three-call rule equals the median clipped to the interval 1 through 5.
3. Report all planned questions, including counts by call status for each arm and the IDs of unpaired cases. A question missing a usable fourth response in either arm has no paired estimate. Calculate paired statistics on questions with usable responses in both arms and identify these as conditional when incomplete. Also report accuracy-difference bounds across the entire planned cohort by allowing each missing arm to be either correct or incorrect. Do not substitute earlier fourth calls, parse reasoning prose for missing answers, or interpret partial acquisition as a completed primary result.

## Offline execution

Run `python3 research/fourth_call_control/analyze.py fourth_calls.jsonl` from the project directory. The script uses Python and NumPy. Add `--scope all` to include a planned 206-question analysis, and `--out report.json` to save the result. JSONL inputs may be split across files. Use one terminal record per item and arm. Each record has `item_id`, `arm` (`uniform` or `change`) and `status`. A record with `status: "ok"` also has an integer `count` drawn from the question's four options. Record call attempts and retries separately from these terminal records.

The archived prefix counts and reference labels come from the submitted supplement's `data/action_count.csv`. Public options come from `data/raw/mlvu/MLVU/json/4_count.json`. The report identifies its source and result files by SHA-256.
