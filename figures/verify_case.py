from pathlib import Path
from collections import Counter
from itertools import product
import argparse,csv,json,hashlib
parser=argparse.ArgumentParser(description="Verify the illustrated official-test prefix against every admissible completion")
parser.add_argument("--package",type=Path,required=True)
parser.add_argument("--out",type=Path,required=True)
args=parser.parse_args()
root=args.package
args.out.parent.mkdir(parents=True,exist_ok=True)
items_file=root/'external_eval/items.json';calls_file=root/'external_eval/calls.jsonl';replay_file=root/'exact_stopping/items.csv'
item=next(i for i in json.loads(items_file.read_text()) if i['id']=='AC_8')
calls={r['slot']:r['count'] for l in calls_file.read_text().splitlines() if (r:=json.loads(l))['id']=='AC_8'}
record=next(r for r in csv.DictReader(replay_file.open()) if r['split']=='test' and r['item_id']=='AC_8')
observations=[calls[s] for s in ('U1','U2','U3','C4')];possible=[*item['options'],None]
assert observations==[3,3,4,3] and item['gold']==3 and record['used_calls']=='3' and record['four_slot_count']=='3'
def score(o,c):return 0 if o is None else -2*abs(o-c)+int(o==c)
def scores(os):return {c:sum(score(o,c) for o in os) for c in range(1,6)}
def winner(os):return min(scores(os),key=lambda c:(-scores(os)[c],c))
checks=[]
for t in (2,3):
    prefix=observations[:t];prefix_scores=scores(prefix);h=winner(prefix)
    margins={r:prefix_scores[h]-prefix_scores[r]+(4-t)*min(score(o,h)-score(o,r) for o in possible) for r in range(1,6) if r!=h}
    completions=Counter(winner(prefix+list(s)) for s in product(possible,repeat=4-t))
    safe=all(m>=0 if r>h else m>0 for r,m in margins.items())
    assert sorted(completions)==([1,3] if t==2 else [3])
    assert safe==(t==3) and sum(completions.values())==len(possible)**(4-t)
    checks.append(dict(observed_prefix=prefix,leader=h,prefix_scores=prefix_scores,minimum_final_lead_by_rival=margins,possible_final_counts=sorted(completions),completions_by_final_count=dict(sorted(completions.items())),stop=safe))
assert winner([3,3,1,1])==1 and winner([3,3,None,None])==3
files=[items_file,calls_file,replay_file]
result=dict(item_id=item['id'],video=item['video'],offered_options_in_prompt_order=item['options'],candidate_counts=[1,2,3,4,5],admissible_blank=True,acquired_four_slot_reports=observations,gold=item['gold'],observed_decision=record,fourth_slot_shown_as_skipped_in_replay=True,checkpoints=checks,source_sha256={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
args.out.write_text(json.dumps(result,indent=2)+'\n')
