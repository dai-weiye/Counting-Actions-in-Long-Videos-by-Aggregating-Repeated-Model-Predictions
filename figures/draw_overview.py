"""Build MAMC's ordered-evidence architecture from released records (standard library)."""
import argparse
import itertools
import json
from pathlib import Path
import xml.etree.ElementTree as ET

W, H = 511.92, 162
AMETHYST, BLUE, CHAMPAGNE, SIGNAL_RED = "#6E5BA3", "#478BCA", "#FED479", "#EB2124"
BLUE_INK, RED_INK = "#35658F", "#C91C20"
SLATE, LIGHT_SLATE = "#75828B", "#8E989F"
INK, MUTED, FRAME, PALE, WHITE = "#252932", "#535B65", "#D0D4D8", "#EEF5FB", "#FFFFFF"
A = argparse.ArgumentParser()
A.add_argument("archive", type=Path, help="Root of the anonymous reproducibility archive")
A.add_argument("out", type=Path, help="Output directory")
args = A.parse_args()
args.out.mkdir(parents=True, exist_ok=True)
source = args.archive / "external_eval"
def jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
def one_item(records):
    return [r for r in records if r["id"] == "AC_8"]
item, = one_item(json.loads((source/"items.json").read_text()))
calls = {r["slot"]:r["count"] for r in one_item(jsonl(source/"calls.jsonl"))}
reports = [calls[s] for s in ("U1","U2","U3","C4")]
selected = {r["arm"]:r["frame_indices"] for r in one_item(jsonl(source/"selected_frames.jsonl"))}
locked = {r["arm"]:r["frame_indices"] for r in one_item(jsonl(source/"planned_frame_inputs.jsonl"))}
assert selected == locked and all(len(a)==128 for a in selected.values())
assert all(a == sorted(a) for a in selected.values())
last = selected["uniform"][-1]
targets = list(range(1,6)); admissible = [*item["options"],None]
def s(o,c):
    return 0 if o is None else int(o==c)-2*abs(o-c)
def summed(os):
    return [sum(s(o,c) for o in os) for c in targets]
def winner(os):
    if all(x is None for x in os): return None
    scores=summed(os)
    return min(targets, key=lambda c:(-scores[c-1],c))
def checkpoint(t):
    obs=reports[:t]; val=summed(obs); h=winner(obs); others=[c for c in targets if c!=h]
    leads=[val[h-1]-val[c-1] for c in others]
    margins=[v+(4-t)*min(s(u,h)-s(u,c) for u in admissible) for c,v in zip(others,leads)]
    possible={winner(obs+list(tail)) for tail in itertools.product(admissible,repeat=4-t)}
    safe=all(m>0 if c<h else m>=0 for c,m in zip(others,margins))
    assert safe==(len(possible)==1)
    return dict(prefix=obs,leader=h,rivals=others,lead=leads,worst_final_lead=margins,safe=safe)
prefix_scores=[summed(reports[:i]) for i in (1,2,3)]
per_call=[summed([o]) for o in reports[:3]]
pre,post=[checkpoint(t) for t in (2,3)]
assert reports==[3,3,4,3] and not pre["safe"] and post["safe"]
assert post["lead"]==[14,8,3,10] and post["worst_final_lead"]==[9,6,0,5]
assert pre["worst_final_lead"]==[0,2,0,0]
def fmt(x):
    return f'{x:.4f}'.rstrip('0').rstrip('.') if isinstance(x,float) else str(x)
def attr(a):return {k.replace('_','-'):fmt(v) for k,v in a.items() if v is not None}
root=ET.Element("svg",attr(dict(xmlns="http://www.w3.org/2000/svg",width=f"{W}pt",height=f"{H}pt",viewBox=f"0 0 {W} {H}",font_family="Arimo, Arial, Helvetica, sans-serif",font_size=8)))
def group(i,kind="node"):
    return ET.SubElement(root,"g",{"id":("n:" if kind=="node" else "e:")+i,"data-kind":kind})
def put(g,tag,**p):
    return ET.SubElement(g,tag,attr(p))
def text(g,x,y,s,size=8,col=INK,anch="start",bold=False):
    n=put(g,"text",x=x,y=y,font_size=size,text_anchor=anch,fill=col,font_weight="bold" if bold else "normal")
    n.text=str(s).replace("-","−") if isinstance(s,int) else str(s)
def line(g,x,y,X,Y,col=FRAME,width=.65,dash=False):
    put(g,"line",x1=x,y1=y,x2=X,y2=Y,stroke=col,stroke_width=width,stroke_dasharray="2.6 2.2" if dash else None)
def rect(g,x,y,w,h,fill=WHITE,stroke=None,lw=.6):
    put(g,"rect",x=x,y=y,width=w,height=h,fill=fill,stroke=stroke,stroke_width=lw if stroke else None)
def poly(g,ps,col,stroke=None,lw=.65):
    put(g,"polygon" if col!="none" else "polyline",points=" ".join(fmt(x)+","+fmt(y) for x,y in ps),fill=col,stroke=stroke,stroke_width=lw if stroke else None)
def circ(g,x,y,r,fill,stroke=None,lw=.8):
    put(g,"circle",cx=x,cy=y,r=r,fill=fill,stroke=stroke,stroke_width=lw if stroke else None)
def arrow(g,x,y,X,col=INK,w=.8):
    line(g,x,y,X-3.2,y,col,w);poly(g,[(X,y),(X-3.7,y-1.5),(X-3.7,y+1.5)],col)
def blend(a,b,t):
    v=[int(round(int(a[i:i+2],16)*(1-t)+int(b[i:i+2],16)*t)) for i in (1,3,5)]
    return "#"+"".join(f"{x:02X}" for x in v)
def title(g,x,label):
    text(g,x,14,label,9.2,AMETHYST,bold=True)
def strip(g,x,y,w,idx,optional):
    # Each equal-duration bin contains observed sample indices from the released trace.
    counts=[0]*16
    for i in idx:
        counts[min(15,(i*16)//(last+1))]+=1
    assert sum(counts)==128
    dx=w/16
    for j,count in enumerate(counts):
        height=0.5+7.5*count/15
        xx=x+j*dx+.10
        rect(g,xx,y+3.6-8,dx-.36,8,"#F0F2F3")
        if count:
            rect(g,xx,y+3.6-height,dx-.36,height,"#99A3AA" if optional else BLUE)
G=group("sampled-video")
title(G,5,"Frame sampling");text(G,5,32,"Frames per time bin",7.1,MUTED);arrow(G,68,29,76,MUTED,.65)
rY=[47,65,83,102]
for j,(name,y) in enumerate(zip(["U1","U2","U3","C4"],rY)):
    optional=(j==3);text(G,5,y+2.4,name,7.6,MUTED if optional else BLUE_INK,bold=not optional)
    strip(G,20,y,55,selected["change" if optional else "uniform"],optional)
text(G,5,123,"U   Uniform, 128 frames",7.1,MUTED);text(G,5,135,"C   Change, 128 of 256",7.1,MUTED)
G=group("shared-model")
for i,y in enumerate(rY):
    if i<3:line(G,76,y,87,y,BLUE,.7)
    else:line(G,76,y,87,y,SLATE,.7,True)
rect(G,87,38,31,72,PALE,BLUE,.9)
text(G,102.5,65,"Frozen",7.8,BLUE_INK,"middle",True);text(G,102.5,75.5,"VLM",9.0,BLUE_INK,"middle",True)
line(G,102.5,111,102.5,119,SLATE,.7)
text(G,102.5,129,"Question",7.1,MUTED,"middle");text(G,102.5,138,"+ options",7.1,MUTED,"middle")
G=group("count-reports")
text(G,124,32,"Report",7.3,MUTED)
for y,o in zip(rY[:3],reports[:3]):
    arrow(G,118.5,y,126,BLUE,.72)
    text(G,135.5,y+3.2,o,9,BLUE_INK,"middle",True)
    arrow(G,143,y,153,BLUE,.72)
line(G,118.5,rY[3],127,rY[3],SLATE,.7,True)
line(G,127,99,127,105,SLATE,.75)
G=group("ordered-evidence")
title(G,155,"Per-report evidence")
cols=[165.6+17.7*j for j in range(5)]
for x,c in zip(cols,targets): text(G,x,32,c,7.6,AMETHYST,"middle",True)
for y,ss in zip(rY[:3],per_call):
    for x,score in zip(cols,ss):
        fill={1:CHAMPAGNE,-2:"#DCEAF5",-4:"#EAF2F8",-6:"#F4F7FA"}[score]
        rect(G,x-8.3,y-6.6,16.6,13.2,fill)
        text(G,x,y+2.6,"+1" if score==1 else score,7.6,INK,"middle",score==1)
text(G,155,106,"Candidate counts 1 to 5",7.3,MUTED)
text(G,155,125,"s(o,c) = −2|o−c| + 1[o=c]",7.6,INK)
G=group("accumulation")
title(G,258,"Cumulative score");arrow(G,245,11,254,SLATE,.75)
xpath=[268,300,332]
for x,name in zip(xpath,("U1","U2","U3")):text(G,x,32,name,7.2,MUTED,"middle")
def score_y(v):
    return 44+(2-v)*4.05
for tick in [0,-5,-10,-15]:
    yy=score_y(tick);line(G,264,yy,336,yy,FRAME,.45);text(G,261.5,yy+2.1,tick,7,MUTED,"end")
for c in [1,5,2,4,3]:
    ps=[(x,score_y(v[c-1])) for x,v in zip(xpath,prefix_scores)]
    col=AMETHYST if c==3 else (SLATE if c in (2,5) else LIGHT_SLATE)
    poly(G,ps,'none',col,1.65 if c==3 else 1.0)
    for x,y in ps:
        circ(G,x,y,1.95 if c==3 else 1.65,AMETHYST if c==3 else WHITE,col,.8)
    x,y=ps[-1]
    text(G,337.5,y+2.15,'c'+str(c),7.1,AMETHYST if c==3 else MUTED,bold=(c==3))
text(G,265,125,"S(c) = Σ s(o,c)",7.7,AMETHYST);text(G,265,139,"Ties favor smaller counts",7.15,MUTED);arrow(G,346,11,363,SLATE,.75)
G=group("stopping-margin")
title(G,368,"Lead after U3");text(G,436,14,"One call left",7.3,MUTED)
x0=390;unit=6.35
for v in (0,7,14):
    x=x0+v*unit;text(G,x,32,v,7.0,MUTED,'middle')
    line(G,x,37,x,103,'#DADDE0',.45)
text(G,369,32,"Rival",7,MUTED)
for y,r,L,m in zip([47,64,81,98],post["rivals"],post["lead"],post["worst_final_lead"]):
    text(G,376,y+2.6,r,7.6,INK,'middle')
    line(G,x0,y,x0+L*unit,y,BLUE,1.05)
    if m:
        line(G,x0,y,x0+m*unit,y,AMETHYST,2)
    circ(G,x0+L*unit,y,2,WHITE,BLUE,.9)
    circ(G,x0+m*unit,y,2.15,AMETHYST,AMETHYST,.45)
    text(G,487,y+2.6,m,7.5,AMETHYST,'start',True)
line(G,371,114,379,114,AMETHYST,1.7);circ(G,379,114,1.7,AMETHYST)
text(G,384,116.5,"Guaranteed",7.2,INK)
line(G,439,114,447,114,BLUE,1.05);circ(G,447,114,1.7,WHITE,BLUE,.75)
text(G,451,116.5,"Current",7.2,INK)
# Every rival contributes to the readout, including the zero margin versus larger count 4.
for yy in [47,98]:line(G,494.5,yy,499.5,yy,AMETHYST,.7)
line(G,499.5,47,499.5,98,AMETHYST,.7)
line(G,499.5,73,505.5,73,AMETHYST,.75);line(G,505.5,73,505.5,132,AMETHYST,.75)
line(G,505.5,132,495,132,AMETHYST,.75);poly(G,[(491.5,132),(495.2,130.45),(495.2,133.55)],AMETHYST)
text(G,445,135.4,"Count 3",10.5,AMETHYST,bold=True);text(G,445,147.5,"C4 skipped",7.6,MUTED)
G=group("request-another-uniform-read","edge")
text(G,368,133,"After U2",7.7,RED_INK,bold=True);text(G,368,144,"Rival 1 can tie",7.4,RED_INK)
line(G,364,140,356,140,SIGNAL_RED,.85);line(G,356,140,356,155,SIGNAL_RED,.85)
line(G,356,155,245,155,SIGNAL_RED,.85);text(G,222,157.4,"Request U3",7.6,RED_INK,"middle",True)
line(G,198.7,155,80.5,155,SIGNAL_RED,.85);line(G,80.5,155,80.5,83,SIGNAL_RED,.85)
arrow(G,80.5,83,87,SIGNAL_RED,.85)

ET.indent(root)
ET.ElementTree(root).write(args.out/"overview.svg",encoding='utf-8',xml_declaration=True)
def bins(xs):return [sum(min(15,i*16//(last+1))==b for i in xs) for b in range(16)]
evidence=dict(frame_counts_per_temporal_bin={k:bins(v) for k,v in selected.items()},item="AC_8",options=item["options"],frame_source="external_eval/selected_frames.jsonl",last_uniform_index=last,per_report_scores=per_call,cumulative_scores=prefix_scores,checkpoints=[pre,post])
(args.out/"figure_values.json").write_text(json.dumps(evidence,indent=2)+"\n")
