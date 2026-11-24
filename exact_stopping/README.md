# Exact stopping for a four-report count

The reference procedure requests four model responses in order. Each response is a listed numerical option or a blank denoting an unusable answer. With the original frozen domain, score every count from 1 through 5. With the offered-option domain, use the sorted unique numeric choices of the question, including values outside 1 through 5. For each candidate, sum the score `s(r,c) = -2*abs(r-c) + (r==c)` for each valid response `r`. A blank contributes zero. Return the smallest candidate with maximum score, or leave the item unanswered if all four responses are blank.

## Stopping condition

After observing the first `t` slots, let `S(c)` be the accumulated score and `h` the smallest maximizing count. For each candidate `c` different from `h`, let `d(h,c)` be the minimum of `s(u,h)-s(u,c)` over the listed options and a blank. The smallest attainable final lead of `h` over `c` is `M = S(h)-S(c)+(4-t)*d(h,c)`. Stop with `h` exactly when every smaller rival has `M>0` and every larger rival has `M>=0`. An all-blank prefix requires the next call until all four slots have been observed.

Any future response reduces the lead by at most the amount used in this calculation. Conversely, the completion that repeats an option attaining the minimum in each remaining slot realizes this final margin against the specific rival. An all-blank completion preserves the current leader. Thus the decision is fixed precisely when the test passes. Caching the minimum margins for an option set leaves a comparison against each rival after updating the prefix scores. Only the offered options and answers already received enter the decision.

Adaptive sampling for self-consistency was studied by [Aggarwal and colleagues at EMNLP 2023](https://aclanthology.org/2023.emnlp-main.761/). Here the objective is exact agreement with the four-slot additive count estimator on every valid completion. Offline replay measures how many calls a sequential implementation would need for recorded responses. Savings use four planned calls per video as the denominator. The 32-video greedy Qwen panel produced the same four-report estimate as the range-clipped first report on every video. Its certified 89 calls and its 32 first calls therefore gave identical recorded decisions on that panel.

## Replay and exhaustive verification

Run from the anonymous package root with Python and NumPy. The development replay uses the archived numeric reports and their item-specific options.

```sh
python exact_stopping/replay.py --dev-csv data/action_count.csv --dev-options fourth_call/options.json --out replay_stopping.json --items-out replay_stopping.csv
python exact_stopping/verify.py --dev-options fourth_call/options.json
```

To reproduce the development results and both models on the official test split, run the following commands. The scorer reports all 60 test questions and those with four usable responses separately.

```sh
python exact_stopping/replay.py --dev-csv data/action_count.csv --dev-options fourth_call/options.json --test-items external_eval/items.json --test-calls external_eval/calls.jsonl --open-weight-calls open_weight_check/predictions.jsonl --open-weight-test-cohort open_weight_test/cohort.json --open-weight-test-calls open_weight_test/predictions.jsonl --out replay_stopping.json --items-out replay_stopping.csv
cmp replay_stopping.json exact_stopping/results.json
cmp replay_stopping.csv exact_stopping/items.csv
python exact_stopping/verify.py --dev-options fourth_call/options.json --test-items open_weight_test/cohort.json
```

The verifier independently enumerates every four-slot sequence, computes its reference decision, and compares the set of possible decisions after each prefix with the margin-based stopping condition.

## Candidate sets at deployment

A system with listed answers can use those values as both the possible responses and the decision candidates. The same margin test then certifies its own four-slot estimator. For a question with six offered options that include 6, a count of 6 is admissible. To reproduce that registered option-set sensitivity, add `--candidate-domain offered` to the joint replay command above and compare its outputs with `results_offered.json` and `items_offered.csv`. The default `--candidate-domain fixed` reproduces the original range of 1 through 5. The independent verifier checks both candidate sets on every admissible prefix.

On the 60 official DeepSeek test questions, early stopping used 180 of 240 calls with the fixed range and 191 of 240 with the offered options. Both rules reproduced all 60 decisions of their own four-slot procedure, including the available-report fallback for unusable responses.

On all 60 Qwen test questions the fixed-range rule stopped after 175 of 240 calls and the offered-option rule after 180 of 240. Both returned the same answer as their own four-slot procedure on all 60 questions. A single range-clipped first response reproduced the fixed-range output in 60 calls. A single raw first response reproduced the offered-option output in 60 calls.

For direct use from Python, `aggregate(reports, candidates=offered)` computes the count, and `stop_when_fixed(prefix, choices=offered, candidates=offered)` returns the stopping flag and certified prediction. Duplicated or unsorted numeric options are accepted. A `None` response represents any blank or unusable final answer.

## On-demand model calls

`count_sequential(options, source, candidates=options)` accepts either a function that generates one response for a named slot or an iterator that generates responses as it advances. The call order is U1, U2, U3, C4. It returns `(count, calls_used)` as soon as the exact stopping condition holds. Pass the function itself so subsequent requests are created only when needed.

```python
from exact_stopping.replay import count_sequential

# fetch_count(slot) sends that single model request and parses its final answer.
count, calls_used = count_sequential(options, fetch_count, candidates=options)
```

The independent verifier computes the earliest safe prefix from the full set of completions. Its counting oracle raises an exception for any extra request and checks the returned count and number of calls. It checks the iterator entry point separately using a stream truncated at the earliest safe prefix. The checks run on every possible sequence for each distinct four- or six-option list from the two splits, under each candidate domain.
