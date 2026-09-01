from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import html
import cairosvg

OUT = Path(__file__).resolve().parent / "s2_flowcharts_top_level"
OUT.mkdir(parents=True, exist_ok=True)

COL = {
    "blue":"#0C447C", "green":"#0F6E56", "purple":"#3C3489", "red":"#993C1D",
    "decision":"#854F0B", "neutral":"#5F5E5A", "white":"#FFFFFF", "text":"#111111",
    "hil":"#D85A30", "gate":"#7A5C12", "loop":"#3D8FD9", "panel":"#F8FAFC",
    "panel_border":"#D9E1E8", "detail_border":"#8A887F",
}
TARGET_GAP = 2.0

def esc(v): return html.escape(str(v))

@dataclass(frozen=True)
class R:
    x:float; y:float; w:float; h:float
    @property
    def left(self): return self.x
    @property
    def right(self): return self.x+self.w
    @property
    def top(self): return self.y
    @property
    def bottom(self): return self.y+self.h
    @property
    def cx(self): return self.x+self.w/2
    @property
    def cy(self): return self.y+self.h/2
    def a(self,side,gap=0.0):
        if side=="top": return (self.cx,self.top-gap)
        if side=="bottom": return (self.cx,self.bottom+gap)
        if side=="left": return (self.left-gap,self.cy)
        if side=="right": return (self.right+gap,self.cy)
        raise ValueError(side)

@dataclass(frozen=True)
class D:
    cx:float; cy:float; w:float; h:float
    @property
    def left(self): return self.cx-self.w/2
    @property
    def right(self): return self.cx+self.w/2
    @property
    def top(self): return self.cy-self.h/2
    @property
    def bottom(self): return self.cy+self.h/2
    def a(self,side,gap=0.0):
        if side=="top": return (self.cx,self.top-gap)
        if side=="bottom": return (self.cx,self.bottom+gap)
        if side=="left": return (self.left-gap,self.cy)
        if side=="right": return (self.right+gap,self.cy)
        raise ValueError(side)

@dataclass(frozen=True)
class C:
    cx:float; cy:float; r:float
    @property
    def left(self): return self.cx-self.r
    @property
    def right(self): return self.cx+self.r
    @property
    def top(self): return self.cy-self.r
    @property
    def bottom(self): return self.cy+self.r
    def a(self,side,gap=0.0):
        if side=="top": return (self.cx,self.top-gap)
        if side=="bottom": return (self.cx,self.bottom+gap)
        if side=="left": return (self.left-gap,self.cy)
        if side=="right": return (self.right+gap,self.cy)
        raise ValueError(side)

def rect_svg(g,fill,rx=14,stroke="none",sw=0):
    return f'<rect x="{g.x}" y="{g.y}" width="{g.w}" height="{g.h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>'
def diamond_svg(g,fill): return f'<polygon points="{g.cx},{g.top} {g.right},{g.cy} {g.cx},{g.bottom} {g.left},{g.cy}" fill="{fill}"/>'
def circle_svg(g,fill): return f'<circle cx="{g.cx}" cy="{g.cy}" r="{g.r}" fill="{fill}"/>'
def label(x,y,text,fill,size=14,weight=700,anchor="middle"):
    return f'<text x="{x}" y="{y}" text-anchor="{anchor}" fill="{fill}" font-size="{size}" font-weight="{weight}">{esc(text)}</text>'
def text_lines(x,y,title,subs=(),title_size=17,sub_size=13,line_gap=20,fill="#FFFFFF",anchor="middle"):
    o=[f'<text x="{x}" y="{y}" text-anchor="{anchor}" class="node-title" font-size="{title_size}" fill="{fill}">{esc(title)}</text>']; yy=y+line_gap
    for s in subs:
        o.append(f'<text x="{x}" y="{yy}" text-anchor="{anchor}" class="node-sub" font-size="{sub_size}" fill="{fill}" opacity="0.90">{esc(s)}</text>'); yy += line_gap-2
    return "\n".join(o)

def _clean_points(points):
    c=[]
    for p in points:
        p=(float(p[0]),float(p[1]))
        if not c or p!=c[-1]: c.append(p)
    assert len(c)>=2
    for a,b in zip(c,c[1:]):
        dx=abs(a[0]-b[0]); dy=abs(a[1]-b[1])
        assert dx<1e-9 or dy<1e-9, f"non-orthogonal segment: {a}->{b}"
        assert dx+dy>0.5, f"zero/near-zero segment: {a}->{b}"
    return c

def path(points,cls="edge-dark",marker=True):
    points=_clean_points(points)
    markers={"edge-dark":"arrowDark","edge-hil":"arrowHil","edge-gate":"arrowGate","edge-loop":"arrowLoop","edge-assoc":"arrowAssoc"}
    d="M"+" L".join(f"{x:g} {y:g}" for x,y in points); m=f' marker-end="url(#{markers[cls]})"' if marker else ""
    return f'<path d="{d}" class="{cls}"{m}/>'
def direct(src,ss,dst,ds,cls="edge-dark",gap=TARGET_GAP): return path([src.a(ss),dst.a(ds,gap)],cls)
def ortho_vh(src,ss,dst,ds,cls="edge-dark",gap=TARGET_GAP,bend_y=None):
    s=src.a(ss); t=dst.a(ds,gap); by=t[1] if bend_y is None else bend_y; return path([s,(s[0],by),(t[0],by),t],cls)
def ortho_hv(src,ss,dst,ds,cls="edge-dark",gap=TARGET_GAP,bend_x=None):
    s=src.a(ss); t=dst.a(ds,gap); bx=t[0] if bend_x is None else bend_x; return path([s,(bx,s[1]),(bx,t[1]),t],cls)

def header(w,h,title):
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}"><title>{esc(title)}</title>
<defs>
<marker id="arrowDark" markerWidth="6" markerHeight="6" refX="8" refY="4.5" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L9,4.5 L0,9 z" fill="#263238"/></marker>
<marker id="arrowHil" markerWidth="6" markerHeight="6" refX="8" refY="4.5" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L9,4.5 L0,9 z" fill="#D85A30"/></marker>
<marker id="arrowGate" markerWidth="6" markerHeight="6" refX="8" refY="4.5" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L9,4.5 L0,9 z" fill="#7A5C12"/></marker>
<marker id="arrowLoop" markerWidth="6" markerHeight="6" refX="8" refY="4.5" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L9,4.5 L0,9 z" fill="#3D8FD9"/></marker>
<marker id="arrowAssoc" markerWidth="6" markerHeight="6" refX="8" refY="4.5" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L9,4.5 L0,9 z" fill="#8A887F"/></marker>
</defs><style>
text {{font-family:Helvetica,Arial,sans-serif}} .node-title{{font-weight:700}} .node-sub{{font-weight:400}}
.edge-dark{{fill:none;stroke:#263238;stroke-width:2.8;stroke-linejoin:round;stroke-linecap:round}}
.edge-hil{{fill:none;stroke:#D85A30;stroke-width:3;stroke-linejoin:round;stroke-linecap:round}}
.edge-gate{{fill:none;stroke:#7A5C12;stroke-width:2.8;stroke-linejoin:round;stroke-linecap:round}}
.edge-loop{{fill:none;stroke:#3D8FD9;stroke-width:2.8;stroke-dasharray:7 5;stroke-linejoin:round;stroke-linecap:round}}
.edge-assoc{{fill:none;stroke:#8A887F;stroke-width:2;stroke-dasharray:6 4;stroke-linejoin:round;stroke-linecap:round}}
</style>'''

def write(name,w,h,title,body):
    svg=header(w,h,title)+"\n"+body+"\n</svg>"; sp=OUT/f"{name}.svg"; pp=OUT/f"{name}.pdf"; pn=OUT/f"{name}.png"
    sp.write_text(svg,encoding="utf-8"); cairosvg.svg2pdf(bytestring=svg.encode(),write_to=str(pp)); cairosvg.svg2png(bytestring=svg.encode(),write_to=str(pn),output_width=w*2,output_height=h*2)
    return sp,pp,pn

def draw_node(out,g,fill,title,subs=(),ts=16,ss=12,rx=14):
    out.extend([rect_svg(g,fill,rx),text_lines(g.cx,g.y+(28 if subs else g.h/2+6),title,subs,ts,ss,19)])
def draw_decision(out,g,title,subs=(),ts=15,ss=11):
    out.extend([diamond_svg(g,COL["decision"]),text_lines(g.cx,g.cy+(5 if not subs else -3),title,subs,ts,ss,15)])
def audit_rect_bounds(name,g,W,H,m=0): assert g.left>=m and g.top>=m and g.right<=W-m and g.bottom<=H-m, f"{name}: bounds"
def audit_diamond_bounds(name,g,W,H,m=0): assert g.left>=m and g.right<=W-m and g.top>=m and g.bottom<=H-m, f"{name}: bounds"

def build_ieee():
    W,H=840,1360
    out=[rect_svg(R(28,24,W-56,H-48),COL["panel"],20,COL["panel_border"],2)]
    out.append(label(W/2,58,"Scenario 2 - Top-Level OLTC Flow",COL["text"],22,700))
    cx=430; mw=430; x=cx-mw/2; skip_rail=78

    run=R(x,86,mw,64)
    prep=R(x,180,mw,82)
    ckpt=D(cx,326,230,82)
    reset=R(x,410,mw,66)
    setup=R(x,508,mw,112)
    calib=R(x,654,mw,100)
    loop=R(x,792,mw,80)
    merge=C(cx,936,14)
    result=R(x,982,mw,78)
    summary=R(x,1092,mw,66)
    pubend=R(x,1190,mw,62)
    ret=R(x,1280,mw,42)

    draw_node(out,run,COL["blue"],"run_scenario_2()",("network, profiles, voltage limits, tap options",),18,12.5)
    draw_node(out,prep,COL["blue"],"Prepare simulation",("adapt_profiles(); on_scenario_start()", "get_resume_records() -> start_t"),17,12)
    draw_decision(out,ckpt,"checkpoint covers all T?",ts=15)
    draw_node(out,reset,COL["blue"],"Reset network state",("drop stale controllers; pp.reset_results()",),16,11.5)
    draw_node(out,setup,COL["green"],"Prepare OLTC group",("select trafos; print original metadata", "optional default completion + user override", "print final metadata; validate ganged tap data"),15,11)
    draw_node(out,calib,COL["green"],"Calibrate + initialize tap state",("derive gang limits / neutral + controlled buses", "deep-copy probe PFs -> tap sign", "resume last tap_pos or start at neutral"),15,11)
    draw_node(out,loop,COL["neutral"],"Per-timestep OLTC loop",("pre-PF -> tap decision -> optional post-PF / rollback", "record / publish; see Diagram 2"),15,11)
    out.append(circle_svg(merge,COL["neutral"]))
    draw_node(out,result,COL["purple"],"ScenarioResult.from_records()",("aggregate resumed + newly simulated records",),16,11.5)
    draw_node(out,summary,COL["blue"],"Tap activity summary",("moves; blocked actions; min / max tap seen",),15,11)
    draw_node(out,pubend,COL["purple"],"on_scenario_end()",("close checkpoint; final live event",),15,11)
    draw_node(out,ret,COL["blue"],"return ScenarioResult",(),15)

    out.append(direct(run,"bottom",prep,"top")); out.append(direct(prep,"bottom",ckpt,"top"))
    out.append(direct(ckpt,"bottom",reset,"top")); out.append(label(cx+14,ckpt.bottom+19,"no / resume",COL["text"],11,700,"start"))
    out.append(direct(reset,"bottom",setup,"top")); out.append(direct(setup,"bottom",calib,"top")); out.append(direct(calib,"bottom",loop,"top")); out.append(direct(loop,"bottom",merge,"top"))

    # Complete checkpoint skips OLTC setup, calibration and simulation entirely.
    out.append(path([ckpt.a("left"),(skip_rail,ckpt.cy),(skip_rail,merge.cy),merge.a("left",TARGET_GAP)],"edge-gate"))
    out.append(label(ckpt.left-12,ckpt.cy-10,"yes - skip simulation",COL["gate"],11,700,"end"))

    out.append(direct(merge,"bottom",result,"top")); out.append(direct(result,"bottom",summary,"top")); out.append(direct(summary,"bottom",pubend,"top")); out.append(direct(pubend,"bottom",ret,"top"))

    for n,g in {"run":run,"prep":prep,"reset":reset,"setup":setup,"calib":calib,"loop":loop,"result":result,"summary":summary,"pubend":pubend,"ret":ret}.items(): audit_rect_bounds(n,g,W,H,24)
    audit_diamond_bounds("ckpt",ckpt,W,H,24)
    assert skip_rail > 28 and skip_rail < ckpt.left-28
    assert prep.top-run.bottom>=28; assert ckpt.top-prep.bottom>=23; assert reset.top-ckpt.bottom>=43
    assert setup.top-reset.bottom>=30; assert calib.top-setup.bottom>=30; assert loop.top-calib.bottom>=30
    assert merge.top-loop.bottom>=48; assert result.top-merge.bottom>=28; assert summary.top-result.bottom>=30
    assert pubend.top-summary.bottom>=30; assert ret.top-pubend.bottom>=28

    write("flow_s2_top_ieee_final",W,H,"Scenario 2 top-level OLTC flow - IEEE","\n".join(out)); return W,H


def build_presentation():
    W,H=1920,1080
    out=[rect_svg(R(0,0,W,H),COL["panel"],0)]
    main=R(28,24,1060,1030); out.append(rect_svg(main,"#F4F8FB",24,COL["panel_border"],1.5))
    cx=500; mw=700; x=cx-mw/2; skip_rail=74
    run=R(x,36,mw,58); prep=R(x,120,mw,66); ckpt=D(cx,244,290,76)
    reset=R(x,314,mw,58); setup=R(x,400,mw,86); calib=R(x,514,mw,84); loop=R(x,630,mw,68)
    merge=C(cx,758,15); result=R(x,792,mw,62); summary=R(x,876,mw,50); pubend=R(x,946,mw,48); ret=R(x,1012,mw,40)

    draw_node(out,run,COL["blue"],"run_scenario_2()",("network, profiles, OLTC options",),23,15)
    draw_node(out,prep,COL["blue"],"Prepare simulation",("adapt_profiles(); publisher start; checkpoint resume",),21,14)
    draw_decision(out,ckpt,"checkpoint covers all T?",ts=20)
    draw_node(out,reset,COL["blue"],"Reset network state",("drop controllers; reset results",),19,13)
    draw_node(out,setup,COL["green"],"Prepare OLTC group",("select ganged trafos; complete / override / validate tap metadata",),19,13)
    draw_node(out,calib,COL["green"],"Calibrate + initialize tap",("derive gang limits / control buses; two probe PFs -> sign", "resume checkpoint tap or use neutral"),18,12.5)
    draw_node(out,loop,COL["neutral"],"Per-timestep OLTC loop",("tap decision + post-PF / rollback; see detailed figure",),19,13)
    out.append(circle_svg(merge,COL["neutral"]))
    draw_node(out,result,COL["purple"],"ScenarioResult.from_records()",("aggregate resumed + new records",),19,13)
    draw_node(out,summary,COL["blue"],"Tap movement summary",("moves / blocks / observed range",),18,12)
    draw_node(out,pubend,COL["purple"],"on_scenario_end()",("checkpoint + final live event",),18,12)
    draw_node(out,ret,COL["blue"],"return ScenarioResult",(),18)

    out.append(direct(run,"bottom",prep,"top")); out.append(direct(prep,"bottom",ckpt,"top"))
    out.append(direct(ckpt,"bottom",reset,"top")); out.append(label(cx+18,ckpt.bottom+21,"no / resume",COL["text"],14,700,"start"))
    out.append(direct(reset,"bottom",setup,"top")); out.append(direct(setup,"bottom",calib,"top")); out.append(direct(calib,"bottom",loop,"top")); out.append(direct(loop,"bottom",merge,"top"))
    out.append(path([ckpt.a("left"),(skip_rail,ckpt.cy),(skip_rail,merge.cy),merge.a("left",TARGET_GAP)],"edge-gate"))
    out.append(label(ckpt.left-16,ckpt.cy-11,"yes - skip simulation",COL["gate"],14,700,"end"))
    out.append(direct(merge,"bottom",result,"top")); out.append(direct(result,"bottom",summary,"top")); out.append(direct(summary,"bottom",pubend,"top")); out.append(direct(pubend,"bottom",ret,"top"))

    # Explanatory panels only.
    px=1125; pw=760
    p1=R(px,28,pw,300); p2=R(px,350,pw,300); p3=R(px,672,pw,360)
    for p in (p1,p2,p3): out.append(rect_svg(p,COL["white"],18,COL["detail_border"],1.2))

    out.append(label(px+22,60,"OLTC group + metadata setup",COL["text"],18,700,"start"))
    lines=[
        "Selection Tier 1: in-service trafos with vn_hv_kv >= 66 kV.",
        "Tier 2: trafos whose hv_bus is an ext_grid slack bus.",
        "Tier 3: all trafos at the highest available HV voltage level.",
        "Optional completion fills only missing tap fields from benchmark defaults.",
        "User override is then applied explicitly and may overwrite existing/default values.",
        "Validation enforces shared neutral / side / step %, Ratio type and feasible ganged range.",
    ]
    for i,t in enumerate(lines): out.append(label(px+22,96+34*i,t,COL["text"],12.2,600,"start"))

    out.append(label(px+22,382,"Tap-sign calibration + resume state",COL["text"],18,700,"start"))
    lines=[
        "Controlled signal = mean vm_pu across all selected LV-side busbars.",
        "Calibration runs on deepcopy(net), so working-net result tables are untouched.",
        "Probe 1: tap_neutral; Probe 2: neutral +/-1 inside the ganged range.",
        "Observed delta-v determines which tap direction lowers controlled voltage.",
        "abs(delta-v) < 1e-5 pu is treated as an inactive / invalid tap response.",
        "Partial resume restores current_tap from the last checkpoint record when available.",
    ]
    for i,t in enumerate(lines): out.append(label(px+22,418+34*i,t,COL["text"],12.2,600,"start"))

    out.append(label(px+22,704,"Checkpoint + lifecycle semantics",COL["text"],18,700,"start"))
    lines=[
        "A complete checkpoint returns from resumed records before any OLTC setup or calibration.",
        "A partial checkpoint repeats OLTC metadata validation + sign calibration, then restores tap_pos.",
        "There is no S3-style teardown: the controlled transformer group remains at its final tap position.",
        "Setup/calibration exceptions propagate before the timestep loop.",
        "Pre/post/rollback runpp exceptions inside the loop are handled locally as shown in Diagram 2.",
        "Finalization builds ScenarioResult, logs tap activity, calls on_scenario_end(), then returns.",
    ]
    for i,t in enumerate(lines): out.append(label(px+22,742+39*i,t,COL["text"],12.2,600,"start"))

    for n,g in {"run":run,"prep":prep,"reset":reset,"setup":setup,"calib":calib,"loop":loop,"result":result,"summary":summary,"pubend":pubend,"ret":ret}.items(): audit_rect_bounds(n,g,W,H,10)
    audit_diamond_bounds("ckpt",ckpt,W,H,10)
    assert main.right < p1.left; assert skip_rail>28 and skip_rail<ckpt.left-40
    assert reset.top-ckpt.bottom>=30; assert setup.top-reset.bottom>=28; assert calib.top-setup.bottom>=28; assert loop.top-calib.bottom>=30
    assert merge.top-loop.bottom>=45; assert result.top-merge.bottom>=18; assert summary.top-result.bottom>=20; assert pubend.top-summary.bottom>=18; assert ret.top-pubend.bottom>=16

    write("flow_s2_top_presentation_final",W,H,"Scenario 2 top-level OLTC flow - presentation","\n".join(out)); return W,H

if __name__=="__main__":
    iw,ih=build_ieee(); pw,ph=build_presentation()
    print(f"IEEE top-level audited: {iw} x {ih}")
    print(f"Presentation top-level audited: {pw} x {ph}")
    print(f"Outputs: {OUT}")
