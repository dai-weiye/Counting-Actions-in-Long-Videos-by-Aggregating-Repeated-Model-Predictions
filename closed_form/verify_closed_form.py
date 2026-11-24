"""Check the offered-option stopping formula against all legal four-slot completions."""

import argparse
import hashlib
import importlib.util
import json
from collections import defaultdict
from itertools import permutations, product
from pathlib import Path


def utility(report, candidate):
    if report is None:
        return 0
    return int(report == candidate) - 2 * abs(report - candidate)


def offered_stop(prefix, options):
    """Exact only for this score, offered candidates, and smaller-count ties."""
    if all(r is None for r in prefix):
        return len(prefix) == 4, None
    totals = {c: sum(utility(r, c) for r in prefix) for c in options}
    leader = min(options, key=lambda c: (-totals[c], c))
    remaining = 4 - len(prefix)
    for rival in options:
        if rival == leader:
            continue
        margin = totals[leader] - totals[rival] - remaining * (2 * abs(leader - rival) + 1)
        if margin < 0 or (margin == 0 and rival < leader):
            return False, None
    return True, leader


def decision(sequence, options):
    observed = [r for r in sequence if r is not None]
    if not observed:
        return None
    # Independent reference: minimize an integer penalty, breaking ties by count.
    return min(options, key=lambda c: (2 * sum(abs(r - c) for r in observed) - observed.count(c), c))


def read_options(path, field):
    rows = json.loads(path.read_text(encoding="utf-8"))
    return {tuple(sorted(set(map(int, row[field])))) for row in rows}


def verify(package):
    dev = package / "fourth_call" / "options.json"
    test = package / "external_eval" / "items.json"
    script = package / "exact_stopping" / "replay.py"
    module = importlib.util.spec_from_file_location("reference_replay", script)
    replay = importlib.util.module_from_spec(module)
    module.loader.exec_module(replay)
    domains = read_options(dev, "candidates") | read_options(test, "options")
    pair_count = reports_checked = sequence_count = prefix_count = 0
    for options in sorted(domains):
        outcomes = (*options, None)
        for leader, rival in permutations(options, 2):
            differences = [utility(r, leader) - utility(r, rival) for r in outcomes]
            bound = -2 * abs(leader - rival) - 1
            if min(differences) != bound or utility(rival, leader) - utility(rival, rival) != bound:
                raise ValueError(f"Pair margin differs from the claimed bound for {options}, {leader}, {rival}")
            pair_count += 1
            reports_checked += len(outcomes)

        possible = defaultdict(set)
        for sequence in product(outcomes, repeat=4):
            result = decision(sequence, options)
            for n in range(5):
                possible[sequence[:n]].add(result)
            sequence_count += 1

        stopping_lengths = {}
        for prefix, possible_outputs in possible.items():
            unique = len(possible_outputs) == 1
            expected = (True, next(iter(possible_outputs))) if unique else (False, None)
            explicit = offered_stop(prefix, options)
            generic = replay.stop_when_fixed(prefix, options, candidates=options)
            if explicit != generic or explicit != expected:
                raise ValueError(f"Different stopping results on {options}, {prefix}: {explicit}, {generic}, {expected}")
            stopping_lengths[prefix] = unique
            prefix_count += 1
        for sequence in product(outcomes, repeat=4):
            n = next(k for k in range(5) if stopping_lengths[sequence[:k]])
            answer, used = replay.count_sequential(options, iter(sequence), candidates=options)
            if n != used or offered_stop(sequence[:n], options) != (True, answer):
                raise ValueError(f"Different first stop on sequence {sequence}")

    return {
        "score": "Negative twice the absolute deviation, plus one for an exact match. A blank contributes zero.",
        "candidate_domain": "Offered numeric options, with the smaller numeric count winning a tie.",
        "single_report_margin": "For distinct h and c, the minimum over listed responses and a blank is -2*abs(h-c)-1, attained at response c.",
        "option_sets": len(domains),
        "ordered_distinct_pairs_across_sets": pair_count,
        "pair_by_legal_report_checks_including_blank": reports_checked,
        "four_slot_sequences": sequence_count,
        "nonempty_prefixes": prefix_count - len(domains),
        "prefixes_including_empty": prefix_count,
        "earliest_stops_compared": sequence_count,
        "differences_from_general_rule_or_completion_oracle": 0,
        "input_sha256": {
            str(path.relative_to(package)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (dev, test, script)
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True, help="Root of the anonymous reproducibility package")
    parser.add_argument("--out", type=Path, required=True, help="Destination JSON file for the exhaustive check")
    args = parser.parse_args()
    args.out.write_text(json.dumps(verify(args.package), indent=2) + "\n", encoding="utf-8")
