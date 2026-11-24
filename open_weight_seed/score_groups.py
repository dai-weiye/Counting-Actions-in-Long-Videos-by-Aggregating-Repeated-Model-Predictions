"""Analyze the local sampling groups once the complete prediction ledger is available."""
import argparse
from collections import Counter
import csv
import hashlib
import importlib.util
from itertools import combinations
import json
from pathlib import Path


def score(design,records,gold,core):
    quant=core.analyze(records,design,gold)
    by={(r["item_id"],r["group"],r["call"]):r for r in records}
    common=quant["common_four_valid_ids"]
    any_report=any_estimate=0; same=total=0
    for filename in design["filenames"]:
        item=Path(filename).stem
        answers=[[by[item,g+1,c+1]["count"] for c in range(4)] for g in range(6)]
        valid=[v for gr in answers for v in gr if v is not None]
        for x,y in combinations(valid,2):
            total+=1;same+=int(x==y)
        if item in common:
            any_report+=int(len(set(valid))>1)
            decisions=[core.aggregates(gr)["lower"] for gr in answers]
            any_estimate+=int(len(set(decisions))>1)
    quant["across_selections"]={"common_questions":len(common),"common_with_varying_raw_count":any_report,
                                "common_with_varying_group_estimate":any_estimate,"all_valid_report_pairs":total,
                                "different_report_pairs":total-same,"different_report_pair_fraction":(total-same)/total if total else None}
    return quant


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--design",type=Path,required=True);p.add_argument("--counts",type=Path,required=True)
    p.add_argument("--csv",type=Path,required=True);p.add_argument("--out",type=Path,required=True)
    p.add_argument("--analysis-script",type=Path,default=Path(__file__).with_name("seed_analysis.py"));a=p.parse_args()
    sp=importlib.util.spec_from_file_location("shared_seed_analysis",a.analysis_script)
    shared=importlib.util.module_from_spec(sp);sp.loader.exec_module(shared)
    d=json.loads(a.design.read_text());pred=[json.loads(l) for l in a.counts.read_text().splitlines()]
    gold={r["item_id"]:int(r["gold"]) for r in csv.DictReader(a.csv.open())}
    result=score(d,pred,gold,shared);a.out.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
