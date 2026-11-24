"""Independently verify the stopping decision on every admissible prefix."""

import argparse
import json
from collections import defaultdict
from itertools import product
from pathlib import Path

from replay import SLOTS, count_sequential, stop_when_fixed


def independent_decision(values, candidates):
    observed = [v for v in values if v is not None]
    if not observed:
        return None
    return min(
        candidates,
        key=lambda c: (2 * sum(abs(v - c) for v in observed) - observed.count(c), c),
    )


def check_runtime(values, choices, candidates, expected, first_fixed):
    requested = []

    def request(slot):
        k = len(requested)
        if k >= first_fixed or slot != SLOTS[k]:
            raise AssertionError(
                "The runtime requested an unnecessary or out-of-order slot"
            )
        requested.append(slot)
        return values[k]

    decision, used = count_sequential(choices, request, candidates)
    if (decision, used, len(requested)) != (expected, first_fixed, first_fixed):
        raise AssertionError((values, expected, first_fixed, decision, used))
    if count_sequential(choices, iter(values[:first_fixed]), candidates) != (
        expected,
        first_fixed,
    ):
        raise AssertionError("The iterator runtime did not stop at the same prefix")


def verify(options, domain):
    all_sequences = 0
    all_prefixes = 0
    for choices in sorted(options):
        candidates = choices if domain == "offered" else tuple(range(1, 6))
        by_prefix = defaultdict(set)
        for values in product((*choices, None), repeat=4):
            decision = independent_decision(values, candidates)
            for length in range(1, 5):
                by_prefix[values[:length]].add(decision)
            all_sequences += 1
        for prefix, possible in by_prefix.items():
            fixed, decision = stop_when_fixed(prefix, choices, candidates)
            if fixed != (len(possible) == 1) or (fixed and possible != {decision}):
                raise AssertionError((prefix, choices, possible, fixed, decision))
        # Compute the earliest safe stop from the independent completion sets.
        # The oracle refuses to generate any slot after this prefix.
        for values in product((*choices, None), repeat=4):
            first_fixed = next(
                t for t in range(1, 5) if len(by_prefix[values[:t]]) == 1
            )
            expected = next(iter(by_prefix[values[:first_fixed]]))
            check_runtime(values, choices, candidates, expected, first_fixed)
        all_prefixes += len(by_prefix)
    return len(options), all_sequences, all_prefixes


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dev-options", type=Path, required=True)
    p.add_argument("--test-items", type=Path)
    args = p.parse_args()
    options = {
        tuple(sorted(map(int, r["candidates"])))
        for r in json.loads(args.dev_options.read_text())
    }
    if args.test_items:
        test = json.loads(args.test_items.read_text())
        options.update(
            tuple(sorted(map(int, r.get("options") or r["candidates"]))) for r in test
        )
    for domain in ("fixed", "offered"):
        print(
            domain,
            "option sets, legal full sequences, legal nonempty prefixes:",
            *verify(options, domain),
        )
    print("Callable and bounded iterator confirmed the earliest safe stop for every full sequence.")
