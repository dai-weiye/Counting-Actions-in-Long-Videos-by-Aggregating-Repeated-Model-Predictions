"""Reconstruct the MLVU results from anonymized item-level numeric records."""
import argparse
from collections import Counter, defaultdict
import csv
from itertools import combinations, product
import json
import math
from pathlib import Path
import random

import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parent
PRIMARY = ["uniform", "random", "recency", "change"]
CANDIDATES = np.arange(1, 6)
GRID = [(a, b, g) for a, b, g in product(
    (0.5, 0.8, 1.0, 1.3), (0.8, 1.0, 1.5, 2.0), (0.0, 0.4, 0.8, 1.5))]


def read_csv(filename):
    with (ROOT / "data" / filename).open(newline="") as stream:
        return list(csv.DictReader(stream))


def scores(counts, alpha, beta, gamma):
    """Accumulate left to right. Integer coefficients give exact mathematical ties."""
    dtype = np.result_type(alpha, beta, gamma)
    totals = np.zeros((len(counts), len(CANDIDATES)), dtype=dtype)
    for column in counts.T:
        observation = column[:, None]
        terms = (-alpha * np.maximum(CANDIDATES - observation, 0)
                 - beta * np.maximum(observation - CANDIDATES, 0)
                 + gamma * (observation == CANDIDATES))
        totals += terms
    return totals


def aggregate(counts, parameters):
    return CANDIDATES[scores(counts, *parameters).argmax(axis=1)]


def accuracy(correct):
    return {"n": len(correct), "correct": int(np.sum(correct)),
            "accuracy_pct": 100 * float(np.mean(correct))}


def rates(pred, gold):
    return {**accuracy(pred == gold), "mae": float(np.abs(pred - gold).mean()),
            "under": int(np.sum(pred < gold)), "over": int(np.sum(pred > gold))}


def paired(first, reference):
    wins = int(np.sum(first & ~reference))
    losses = int(np.sum(reference & ~first))
    n = wins + losses
    if not n:
        p = 1.0
    elif n <= 25:
        tail = sum(math.comb(n, i) for i in range(min(wins, losses) + 1))
        p = min(1.0, 2 * tail / (2**n))
    else:
        p = math.erfc(abs(abs(wins-losses) - 1) / math.sqrt(2 * n))
    return {"n": len(first), "wins": wins, "losses": losses,
            "delta_pp": 100 * (wins - losses) / len(first), "mcnemar_p": p}


def bootstrap(values):
    """Archived question-resampling routine, 10,000 draws, RNG seed zero."""
    rng = random.Random(0)
    n = len(values)
    outcomes = list(map(int, values))
    draws = sorted(sum(outcomes[rng.randrange(n)] for _ in range(n)) / n for _ in range(10000))
    return {"percentile95": [draws[250], draws[9749]],
            "nonpositive_draws": sum(x <= 0 for x in draws)}


def fit_on_half(O, y):
    perm = np.random.default_rng(0).permutation(len(y))
    dev, test = np.sort(perm[:len(y)//2]), np.sort(perm[len(y)//2:])
    parameters = max(GRID, key=lambda p: int(np.sum(aggregate(O[dev], p) == y[dev])))
    return parameters, dev, test


def verify_score_structure():
    """Check the ordered-count characterization over the historical option range."""
    checked = {}
    for count in (3, 4):
        reports = np.array(list(product(range(7), repeat=count)))
        ordered = np.sort(reports, axis=1)
        low = np.clip(ordered[:, (count - 1) // 2], 1, 5)
        high = np.clip(ordered[:, count // 2], 1, 5)
        in_interval = (CANDIDATES >= low[:, None]) & (CANDIDATES <= high[:, None])
        matches = (reports[:, :, None] == CANDIDATES).sum(axis=1)
        best_match = np.where(in_interval, matches, -1).max(axis=1)
        predicted_maxima = in_interval & (matches == best_match[:, None])
        exact = scores(reports, 2, 2, 1)
        np.testing.assert_array_equal(exact == exact.max(axis=1, keepdims=True), predicted_maxima)
        checked[str(count)] = len(reports)
    return {"report_range": [0, 6], "decision_range": [1, 5], "ordered_inputs_checked": checked}


def random_control(rows, O, y, params):
    seeds = ["seed1", "seed2", "seed3", "seed4"]
    ix = [i for i, row in enumerate(rows) if all(row[k] != "" for k in seeds)]
    samples = np.array([[int(rows[i][k]) for k in seeds] for i in ix])
    truth = y[ix]

    def describe(inputs):
        calls = (inputs == truth[:, None]).mean(axis=0)
        correctness = aggregate(inputs, params) == truth
        agreement = np.mean([np.mean(inputs[:, i] == inputs[:, j]) for i, j in combinations(range(4), 2)])
        detail = {**accuracy(correctness), "individual_call_pct": (100*calls).tolist(),
                  "best_call_pct": 100*float(calls.max()), "mean_call_pct": 100*float(calls.mean()),
                  "gain_over_best_pp": 100*float(correctness.mean() - calls.max()),
                  "pairwise_agreement": float(agreement), "bootstrap95": bootstrap(correctness)["percentile95"]}
        return detail, correctness

    primary, ok1 = describe(O[ix])
    seeded, ok2 = describe(samples)
    delta_boot = bootstrap(ok1.astype(int) - ok2.astype(int))
    return {"primary": primary, "random_seeds": seeded, "paired": paired(ok1, ok2),
            "difference_interval": delta_boot["percentile95"],
            "paired_bootstrap_two_sided_tail": min(1., 2*delta_boot["nonpositive_draws"]/10000),
            "n_resamples": 10000, "missing_items": [r["item_id"] for r in rows if not r["seed1"]]}


def duration_audit(records):
    # The original duration tree sorts public IDs before stratified partitioning.
    records = sorted(records, key=lambda r: r["item_id"]+".mp4")
    d = np.array([float(r["duration_s"]) for r in records])
    y = np.array([int(r["gold"]) for r in records])
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    tree = DecisionTreeClassifier(max_depth=3, random_state=0)
    folds = list(cv.split(d[:, None], y))
    predictions = cross_val_predict(tree, d[:, None], y, cv=folds)
    fold_rates = [float(np.mean(predictions[test] == y[test])) for _, test in folds]
    bins = defaultdict(Counter)
    for seconds, label in zip(d, y):
        bins[round(seconds)][int(label)] += 1
    hits = sum(max(labels.values()) for labels in bins.values())
    return {"depth3_stratified5_pooled": accuracy(predictions == y),
            "depth3_stratified5_mean_fold_pct": 100 * float(np.mean(fold_rates)),
            "nearest_second_majority_in_sample": {"n": len(y), "n_bins": len(bins),
                                                "correct": hits, "accuracy_pct": 100*hits/len(y)}}


def cross_task():
    by_task = defaultdict(list)
    for row in read_csv("reasoning.csv"):
        by_task[row["task"]].append(row)
    results = {}
    for task, rows in by_task.items():
        gold = [int(r["gold"]) for r in rows]
        def matched(column):
            return np.array([r[column] != "" and int(r[column]) == correct for r, correct in zip(rows, gold)])
        on, off = matched("reasoning_on"), matched("reasoning_off")
        results[task] = {"on": accuracy(on), "off": accuracy(off), "paired": paired(on, off)}
    return results


def reproduce():
    annotation_order = read_csv("action_count.csv")
    rows = sorted(annotation_order, key=lambda r: r["item_id"]+".mp4")
    ids = [r["item_id"] for r in rows]
    O = np.array([[int(r[k]) for k in PRIMARY] for r in rows])
    y = np.array([int(r["gold"]) for r in rows])
    expected = np.array([int(r["archived_mamc"]) for r in rows])
    params, dev, test = fit_on_half(O, y)
    full, single = aggregate(O, params), O[:, 0]
    np.testing.assert_array_equal(full, expected, err_msg="Archived MAMC item differs")
    orig, base = full == y, single == y
    majority = np.array([Counter(row).most_common(1)[0][0] for row in O])
    median = np.rint(np.median(O, axis=1)).astype(int)
    rules = {"gamma_zero": aggregate(O, (params[0], params[1], 0.0)),
             "rounded_median": median, "majority_first_encountered": majority,
             "alpha_0.4": aggregate(O, (0.4, params[1], params[2])),
             "beta_1.6": aggregate(O, (params[0], 1.6, params[2]))}
    for k in (1, 2, 3):
        rules[f"first_{k}_calls"] = aggregate(O[:, :k], params)
    ablations = {k: accuracy(v == y) for k, v in rules.items()}
    for name in ["f64", "f32", "uniform_nothink"]:
        ix = [i for i, r in enumerate(rows) if r[name] != ""]
        p = np.array([int(rows[i][name]) for i in ix])
        ablations[name] = {**accuracy(p == y[ix]), "full_on_same_subset": accuracy(orig[ix]),
                           "paired": paired(p == y[ix], orig[ix])}
    exact = scores(O, 2, 2, 1)
    tied = exact == exact.max(axis=1, keepdims=True)
    lower = CANDIDATES[tied.argmax(axis=1)]
    upper = CANDIDATES[::-1][tied[:, ::-1].argmax(axis=1)]
    three = O[:, :3]
    three_median = np.clip(np.median(three, axis=1), 1, 5).astype(int)
    assert np.array_equal(aggregate(three, (2, 2, 1)), three_median)
    estimators = {
        "first_uniform_report": single,
        "three_uniform_clipped_median": three_median,
        "three_uniform_vote": np.array([Counter(row).most_common(1)[0][0] for row in three]),
        "four_report_vote": majority,
        "four_report_exact_lower": lower,
        "four_report_exact_upper": upper,
        "four_report_archived_float": full,
    }
    matched_table = {
        rule: {label: {**rates(pred[idx], y[idx]),
                       "vs_first": paired(pred[idx] == y[idx], single[idx] == y[idx])}
               for label, idx in [("held_out_103", test), ("development_206", np.arange(len(y)))]}
        for rule, pred in estimators.items()
    }
    range_matched = np.clip(single, CANDIDATES[0], CANDIDATES[-1])
    range_diagnostic = {
        label: {"changed_first_call_counts": int(np.sum(range_matched[idx] != single[idx])),
                "raw_first": rates(single[idx], y[idx]),
                "clipped_first": rates(range_matched[idx], y[idx]),
                "exact_lower": rates(lower[idx], y[idx]),
                "paired_exact_lower_vs_clipped_first": paired(lower[idx] == y[idx], range_matched[idx] == y[idx])}
        for label, idx in [("held_out_103", test), ("development_206", np.arange(len(y)))]
    }
    ties = [i for i in range(len(y)) if lower[i] != upper[i]]
    assert np.all(tied[np.arange(len(full)), full - 1])
    exact_results = {"n_items_with_ties": len(ties), "lower": accuracy(lower == y),
                     "upper": accuracy(upper == y), "lower_changed_outputs": int(np.sum(lower != full)),
                     "tie_items": [{"item_id": ids[i], "choices": CANDIDATES[tied[i]].tolist(),
                                    "archived": int(full[i]), "gold": int(y[i])} for i in ties]}
    view_metrics = {"n_disagree_first_three": int(np.sum(np.ptp(O[:,:3],axis=1) != 0)),
                    "n_unanimous_four": int(np.sum(np.ptp(O,axis=1) == 0)),
                    "undercounts_with_a_correct_call": int(np.sum((full<y) & np.any(O==y[:,None],axis=1)))}
    transitions = Counter((int(np.sign(a-g)), int(np.sign(b-g))) for a,b,g in zip(single,full,y))
    errors = {"from_single_to_full": {f"{a:+d} to {b:+d}": n for (a,b), n in sorted(transitions.items())},
              "absolute_errors_nonzero": {int(k): v for k,v in sorted(Counter(np.abs(full[~orig]-y[~orig])).items())}}
    duration = np.array([float(r["duration_s"]) for r in rows])
    groups = {f"gold_{k}": y==k for k in CANDIDATES}
    groups.update(under_600s=duration<600, at_least_600s=duration>=600)
    strata = {s: {"single": rates(single[m],y[m]), "full": rates(full[m],y[m])} for s,m in groups.items()}
    with (ROOT / "provenance/archived_prefix_flags.csv").open() as f:
        prefix_flags = list(csv.DictReader(f))
    assert {(r["item_id"], r["report"]) for r in prefix_flags} == set(product(ids, PRIMARY))
    without_digit = [r for r in prefix_flags if r["digit_in_saved_prefix"] == "0"]
    affected = {r["item_id"] for r in without_digit}
    not_verified = {r["item_id"] for r in prefix_flags if r["prefix_is_reported_option"] == "0"}
    direct = np.array([item not in not_verified for item in ids])
    direct_test = test[direct[test]]
    prefix_audit = {
        "total_reports":len(prefix_flags), "prefix_without_digit":len(without_digit),
        "also_at_output_limit":sum(r["at_token_limit"] == "1" for r in without_digit),
        "questions_with_a_digit_free_prefix":len(affected),
        "net_correct_gain_on_these_questions":sum(int(a)-int(b) for i,a,b in zip(ids,orig,base) if i in affected),
        "all_four_short_answers_verifiable":{
            label:{"archived":accuracy(orig[idx]),"first":accuracy(base[idx]),"comparison":paired(orig[idx],base[idx])}
            for label, idx in [("development",np.flatnonzero(direct)), ("held_out",direct_test)]},
    }
    n, hits = len(y), int(orig.sum())
    hist_p = sum(math.comb(n, k) * 0.463**k * (1-0.463)**(n-k) for k in range(hits, n+1))
    usage = {k: float(np.mean([float(r["uniform_"+k]) for r in rows]))
             for k in ("prompt_tokens", "completion_tokens", "latency_s")}
    one_cost = (usage["prompt_tokens"]*0.28 + usage["completion_tokens"]*0.42)/1e6
    ctrl = random_control(rows, O, y, params)
    compare = paired(orig, base)
    example = ids.index("count_218")
    reports = O[example, :, None]
    mechanism = {"item_id": ids[example], "report_order": PRIMARY, "reports": O[example].tolist(),
                 "candidates": CANDIDATES.tolist(),
                 "total_absolute_deviation": np.abs(reports - CANDIDATES).sum(axis=0).tolist(),
                 "exact_matches": (reports == CANDIDATES).sum(axis=0).tolist(),
                 "scores": scores(O[example:example + 1], *params)[0].round(1).tolist(),
                 "estimate": int(full[example]), "gold": int(y[example])}
    plots = {"mechanism_example": mechanism,
             "sampling_control": {"primary_best": ctrl["primary"]["best_call_pct"],
                                  "primary_aggregate": ctrl["primary"]["accuracy_pct"],
                                  "random_best": ctrl["random_seeds"]["best_call_pct"],
                                  "random_aggregate": ctrl["random_seeds"]["accuracy_pct"]},
             "count_errors": {"corrected": compare["wins"], "introduced": compare["losses"],
                             "single_under": int(np.sum(single<y)), "single_over": int(np.sum(single>y)),
                             "full_under": int(np.sum(full<y)), "full_over": int(np.sum(full>y))}}
    return {"primary": {"full": rates(full,y), "single": rates(single,y),
                        "paired_full_vs_single": compare, "bootstrap95": bootstrap(orig)["percentile95"],
                        "historical_fixed_rate": 0.463, "one_sided_binomial_p": hist_p,
                        "archive_item_predictions_verified": n},
            "selection": {"parameters": list(params), "dev": accuracy(orig[dev]),
                          "holdout": accuracy(orig[test]), "holdout_single": accuracy(base[test]),
                          "holdout_majority": accuracy(majority[test]==y[test]),
                          "holdout_vs_single": paired(orig[test],base[test]),
                          "holdout_vs_majority": paired(orig[test], majority[test]==y[test])},
            "matched_development_table": matched_table,
            "integer_score_structure": verify_score_structure(),
            "matched_range_diagnostic": range_diagnostic,
            "archived_short_answer_audit": prefix_audit,
            "ablations": ablations, "tie_sensitivity": exact_results, "strata": strata,
            "sampling_control": ctrl, "errors": errors, "agreement": view_metrics, "figure_data": plots,
            "duration_audit": duration_audit(annotation_order),
            "reasoning_count": {"on_four_calls": accuracy(orig), "on_one_call": accuracy(base),
                                "off_one_call": accuracy(np.array([int(r["uniform_nothink"]) for r in rows]) == y),
                                "paired_four_on_vs_one_off": paired(orig, np.array(
                                    [int(r["uniform_nothink"]) for r in rows]) == y)},
            "cross_task": cross_task(),
            "inference_cost_estimates": {"observed_uniform_means": usage, "uniform_usd": one_cost,
                        "four_call_usd": 4*one_cost, "ideal_parallel_seconds": usage["latency_s"],
                        "four_serial_seconds": 4*usage["latency_s"]}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, help="Write the complete JSON report (default: stdout)")
    args = parser.parse_args()
    report = json.dumps(reproduce(), indent=2, allow_nan=False) + "\n"
    if args.out:
        args.out.write_text(report, encoding="utf-8")
    else:
        print(report, end="")
