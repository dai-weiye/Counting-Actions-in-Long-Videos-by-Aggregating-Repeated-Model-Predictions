"""Recalculate group-level seed estimates from anonymous individual final counts."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import numpy as np


def aggregates(values):
    if not values:
        return {k: None for k in ("lower","upper","historical_float","majority")}
    candidates = range(1,6)
    exact = [sum(-2*abs(v-c)+int(v==c) for v in values) for c in candidates]
    raw = []
    for c in candidates:
        s = 0.
        for v in values:
            s += -0.8*max(c-v,0)-0.8*max(v-c,0)+0.4*int(v==c)
        raw.append(s)
    tied = [c for c,s in zip(candidates,exact) if s==max(exact)]
    return {"lower":tied[0], "upper":tied[-1], "historical_float":1+raw.index(max(raw)),
            "majority":Counter(values).most_common(1)[0][0]}


def per_cohort(ids, group, slots, gold):
    if not ids: return None
    decisions = []
    for i in ids:
        values = [slots[i,group,k]["count"] for k in range(1,5)]
        valid = [x for x in values if x is not None]
        pred = aggregates(valid)
        singles = [v == gold[i] for v in values]
        decisions.append({"id":i,"singles":singles,"four":len(valid)==4,"n":len(valid),
                          "all_agree":len(valid)==4 and len(set(valid))==1,
                          "strict":len(valid)==4 and pred["lower"]==gold[i],
                          "fallback":pred["lower"]==gold[i],
                          "upper":len(valid)==4 and pred["upper"]==gold[i],
                          "historical_float":len(valid)==4 and pred["historical_float"]==gold[i],
                          "majority":len(valid)==4 and pred["majority"]==gold[i]})
    n=len(ids)
    singles=np.array([r["singles"] for r in decisions],dtype=int)
    out={"n":n,"valid_reports":sum(r["n"] for r in decisions),"four_valid_questions":sum(r["four"] for r in decisions),
         "at_least_one_valid":sum(r["n"]>0 for r in decisions),
         "all_four_agree":sum(r["all_agree"] for r in decisions),"single_correct":singles.sum(axis=0).tolist(),
         "first_single_pct":float(100*singles[:,0].mean()),"best_single_pct":float(100*singles.mean(axis=0).max()),
         "mean_single_pct":float(100*singles.mean())}
    for k in ["strict","fallback","upper","historical_float","majority"]:
        correct=sum(int(d[k]) for d in decisions)
        out[k+"_correct"]=correct
        out[k+"_pct"]=100*correct/n
        out[k+"_gain_vs_mean_single_pp"]=out[k+"_pct"]-out["mean_single_pct"]
    out["strict_gain_vs_first_pp"]=out["strict_pct"]-out["first_single_pct"]
    out["strict_gain_vs_best_pp"]=out["strict_pct"]-out["best_single_pct"]
    return out


def dispersion(measures):
    if not measures or any(g is None for g in measures): return None
    out={}
    for endpoint in ("strict","fallback"):
        vals=[g[endpoint+"_gain_vs_mean_single_pp"] for g in measures]
        out[endpoint]={"gain_by_group_pp":vals,"mean_gain_pp":float(np.mean(vals)),
                       "group_sample_sd_pp":float(np.std(vals,ddof=1)),"range_pp":[min(vals),max(vals)]}
    return out


def analyze(records, design, gold):
    ids=[Path(v).stem for v in design["filenames"]]
    slots={(r["item_id"],r["group"],r["call"]):r for r in records}
    planned={(r["item_id"],r["group"],r["call"]) for r in design["slots"]}
    if slots.keys()!=planned or len(slots)!=len(records):
        raise ValueError("Analysis requires one terminal record for every scheduled slot")
    for row in records:
        if row["status"]=="ok" and type(row["count"]) is not int: raise ValueError("Missing numeric answer")
        if row["status"]!="ok" and row["count"] is not None: raise ValueError("Non-usable call carries a count")
    groups=sorted({r["group"] for r in design["slots"]})
    ok={g:[i for i in ids if all(slots[i,g,c]["status"]=="ok" for c in range(1,5))] for g in groups}
    common=[i for i in ids if all(i in ok[g] for g in groups)]
    all_m=[per_cohort(ids,g,slots,gold) for g in groups]
    common_m=[per_cohort(common,g,slots,gold) for g in groups]
    out={"planned_questions":len(ids),"planned_calls":len(records),"groups":groups,"common_four_valid_ids":common,
         "common_four_valid_n":len(common),"terminal_status":dict(Counter(r["status"] for r in records)),
         "status_by_group":{g:dict(Counter(r["status"] for r in records if r["group"]==g)) for g in groups},
         "common_by_group":common_m,"group_specific_complete_case":[per_cohort(ok[g],g,slots,gold) for g in groups],
         "all_planned_by_group":all_m}
    if len(groups)>1:
        out["common_dispersion"]=dispersion(common_m)
        out["all_planned_dispersion"]=dispersion(all_m)
        out["common_mean_gain_percentile_ci95_pp"]=None
        if common:
            item_gain=[]
            for i in common:
                gs=[per_cohort([i],g,slots,gold)["strict_gain_vs_mean_single_pp"] for g in groups]
                item_gain.append(np.mean(gs))
            draws=np.random.default_rng(0).choice(item_gain,size=(20000,len(common)),replace=True)
            out["common_mean_gain_percentile_ci95_pp"]=np.percentile(draws.mean(axis=1),[2.5,97.5]).tolist()
    return out


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--design",required=True,type=Path)
    p.add_argument("--counts",required=True,type=Path,help="released terminal_counts.jsonl")
    p.add_argument("--csv",required=True,type=Path,help="archived action_count.csv containing public gold labels")
    p.add_argument("--out",type=Path)
    a=p.parse_args()
    gold={r["item_id"]:int(r["gold"]) for r in csv.DictReader(a.csv.open())}
    report=analyze([json.loads(l) for l in a.counts.read_text().splitlines()],json.loads(a.design.read_text()),gold)
    text=json.dumps(report,indent=2,allow_nan=False)+"\n"
    if a.out: a.out.write_text(text)
    else: print(text,end="")
