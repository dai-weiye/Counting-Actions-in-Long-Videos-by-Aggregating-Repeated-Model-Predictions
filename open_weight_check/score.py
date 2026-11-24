"""Score a completed, frozen open-weight panel with the original development labels."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from math import comb
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
SLOTS = ("U1","U2","U3","C4")


def aggregate(reports):
    if not reports:
        return None
    return max(range(1,6), key=lambda c:sum(-2*abs(x-c)+int(x==c) for x in reports))


def contrast(first,second,gold):
    n=len(gold)
    if not n:
        return None
    a,b,y=map(np.asarray,(first,second,gold))
    wins=int(np.sum((a==y)&(b!=y)))
    losses=int(np.sum((a!=y)&(b==y)))
    m=wins+losses
    p=min(1.0,2*sum(comb(m,j) for j in range(min(wins,losses)+1))/2**m)
    within=np.random.default_rng(0).multinomial(n,[wins/n,losses/n,1-(wins+losses)/n],20_000)
    ci=np.percentile((within[:,0]-within[:,1])*100/n,[2.5,97.5])
    return {"n":n,"aggregate_correct":int(np.sum(a==y)),"reference_correct":int(np.sum(b==y)),
            "aggregate_mae":float(np.mean(abs(a-y))),"reference_mae":float(np.mean(abs(b-y))),
            "wins":wins,"losses":losses,"delta_pp":100*(wins-losses)/n,
            "paired_95pct_delta_pp":ci.tolist(),"exact_mcnemar_p":p}


def evaluate(records,labels):
    cohort=json.loads((ROOT/"cohort.json").read_text())
    planned={Path(r["video"]).stem for r in cohort}
    if len(planned)!=32:raise ValueError("Invalid panel")
    ledger=defaultdict(dict)
    for raw in records:
        item, slot=raw["item_id"],raw["slot"]
        if item not in planned or slot not in SLOTS or slot in ledger[item]:
            raise ValueError(f"Unexpected or duplicate slot {(item,slot)}")
        if raw["status"]=="ok":
            if raw["count"] is None or str(raw["count"])!=raw["final"].strip():
                raise ValueError("Invalid accepted response")
        elif raw["count"] is not None:
            raise ValueError("Unexpected answer on unsuccessful response")
        ledger[item][slot]=raw
    if len(ledger)!=len(planned) or any(set(v)!=set(SLOTS) for v in ledger.values()):
        raise ValueError("All 128 terminal records are required")
    status={s:Counter() for s in SLOTS}
    available=Counter()
    yield_correct=Counter()
    gold,primary,single,clipped=[],[],[],[]
    count_disagree=0
    uniform_valid=0
    change_vs_first_usable=0
    change_vs_first_different=0
    same_as_range_matched_base=0
    items=[]
    for item in sorted(ledger):
        y=int(labels[item]["gold"])
        calls=ledger[item]
        rgb={r["rgb_sha256"] for s,r in calls.items() if s!="C4"}
        templates={r["formatted_prompt_sha256"] for r in calls.values()}
        if len(rgb)!=1 or len(templates)!=1:
            raise ValueError("Unexpected mismatch between repeated uniform inputs or formatted prompts")
        nums=[calls[s]["count"] for s in SLOTS]
        for s in SLOTS:status[s][calls[s]["status"]]+=1
        if None not in nums[:3]:
            uniform_valid+=1
            count_disagree+=len(set(nums[:3]))>1
        base=nums[0]
        supported=None if base is None else min(5,max(1,base))
        valid=[v for v in nums if v is not None]
        ag=aggregate(nums) if len(valid)==4 else None
        if base is not None and nums[3] is not None:
            change_vs_first_usable+=1
            change_vs_first_different+=nums[3]!=base
        if ag is not None:
            same_as_range_matched_base+=ag==supported
            gold.append(y);primary.append(ag);single.append(base);clipped.append(supported)
        decisions={"all_four":ag,"raw_u1":base,"range_matched_u1":supported,
                   "any_valid_fallback":aggregate(valid)}
        for k,v in decisions.items():
            available[k]+=v is not None;yield_correct[k]+=v==y
        items.append({"item_id":item,"gold":y,**{s:v for s,v in zip(SLOTS,nums)},**decisions})
    return {"planned":len(planned),"terminal_slots":len(records),"status_by_slot":status,
            "heldout_complete_pairs":len(gold),"same_input_all_three_usable":uniform_valid,
            "same_input_disagreements_among_usable":count_disagree,
            "fourth_and_first_usable":change_vs_first_usable,
            "fourth_vs_first_disagreements":change_vs_first_different,
            "four_report_output_equals_range_matched_u1":same_as_range_matched_base,
            "primary_vs_raw_u1":contrast(primary,single,gold),
            "sensitivity_vs_range_matched_u1":contrast(primary,clipped,gold),
            "available_over_planned":available,"correct_yield_over_planned":yield_correct},items


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions",type=Path,default=ROOT/"predictions.jsonl")
    parser.add_argument("--labels",type=Path,required=True,help="Bundled data/action_count.csv")
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--items-out",type=Path,required=True)
    args=parser.parse_args()
    records=[json.loads(l) for l in args.predictions.read_text().splitlines()]
    with args.labels.open() as f:labels={r["item_id"]:r for r in csv.DictReader(f)}
    report,items=evaluate(records,labels)
    report["input_hashes"]={key:hashlib.sha256(path.read_bytes()).hexdigest()
                            for key,path in [("predictions",args.predictions),("labels",args.labels)]}
    args.out.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    with args.items_out.open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=items[0]);writer.writeheader();writer.writerows(items)
