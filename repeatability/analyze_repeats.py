"""Summarize repeated requests from anonymous question-level records."""
import argparse
from collections import Counter
import csv
import json
from math import sqrt
from pathlib import Path
from statistics import NormalDist


def wilson(events, total):
    if total == 0:
        return None
    z = NormalDist().inv_cdf(0.975)
    rate = events / total
    divisor = 1 + z*z/total
    mid = (rate + z*z/(2*total)) / divisor
    half = z * sqrt(rate*(1-rate)/total + z*z/(4*total*total)) / divisor
    return [max(0., mid-half), min(1., mid+half)]


def summarize(rows):
    if len(rows) != 103 or len({r['item_id'] for r in rows}) != 103:
        raise ValueError('Expected 103 distinct held-out questions')
    first_status, second_status, status_pairs, fingerprints = Counter(), Counter(), Counter(), Counter()
    pairs = same_output = changed = missing = same_fingerprint_pairs = same_fingerprint_changes = 0
    for row in rows:
        a, b = row['first_status'], row['repeat_status']
        first_status[a] += 1
        second_status[b] += 1
        status_pairs[a + '/' + b] += 1
        fp = row['fingerprint_comparison']
        fingerprints[fp] += 1
        if row['first_payload_sha256'] != row['repeat_payload_sha256']:
            raise ValueError('Digest mismatch at ' + row['item_id'])
        for status, value in ((a,row['first_count']), (b,row['repeat_count'])):
            if (status == 'ok') != bool(value):
                raise ValueError('Missing or unexpected count at ' + row['item_id'])
        if a == 'ok' and b == 'ok':
            pairs += 1
            unequal = int(row['first_count']) != int(row['repeat_count'])
            changed += unequal
            same_fingerprint_pairs += fp == 'same'
            same_fingerprint_changes += fp == 'same' and unequal
            same_output += not unequal
        else:
            same_output += a == b
            missing += 1
    return {'planned':len(rows), 'identical_request_digests':len(rows), 'first_status':first_status,
            'repeat_status':second_status, 'paired_status':status_pairs, 'response_fingerprints':fingerprints,
            'both_valid':pairs, 'different_valid_counts':changed, 'at_least_one_unusable_final':missing,
            'disagreement_conditional':changed/pairs if pairs else None,
            'disagreement_wilson_95':wilson(changed,pairs),'same_numeric_or_same_failure_status':same_output,
            'both_valid_same_fingerprint':same_fingerprint_pairs,
            'different_valid_counts_same_fingerprint':same_fingerprint_changes}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('records', type=Path)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    with args.records.open() as f:
        result = summarize(list(csv.DictReader(f)))
    text = json.dumps(result, indent=2, allow_nan=False) + '\n'
    if args.out:
        args.out.write_text(text)
    else:
        print(text, end='')
