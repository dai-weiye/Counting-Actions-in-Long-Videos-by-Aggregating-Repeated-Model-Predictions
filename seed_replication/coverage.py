"""Reconstruct status coverage without scoring an interrupted primary panel."""
import argparse
from collections import Counter
import json
from pathlib import Path


def coverage(design, records):
    def key(s):
        return s["item_id"],s["group"],s["call"]
    slots={key(s) for s in design["slots"]}
    observed={key(r):r for r in records}
    if len(observed)!=len(records) or not set(observed)<=slots:raise ValueError("Unknown or duplicate record")
    def count(rs):
        return dict(sorted(Counter(r["status"] for r in rs).items()))
    missing=[{k:s[k] for k in ("item_id","group","call")} for s in design["slots"] if key(s) not in observed]
    return {"planned_questions":len(design["filenames"]),"planned_slots":len(slots),
            "terminal_slots":len(observed),"not_recorded_slots":len(missing),"terminal_status_counts":count(records),
            "by_group":{str(g):{"terminal":count(r for r in records if r["group"]==g),
                                "not_recorded":sum(s["group"]==g for s in missing)}
                        for g in sorted({s["group"] for s in design["slots"]})},
            "not_recorded":missing}


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--design",type=Path,required=True);p.add_argument("--counts",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True);a=p.parse_args()
    result=coverage(json.loads(a.design.read_text()),[json.loads(s) for s in a.counts.read_text().splitlines()])
    a.out.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
