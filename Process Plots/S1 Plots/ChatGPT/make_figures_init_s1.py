from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import html
import cairosvg

OUT = Path(__file__).resolve().parent / "s1_flowcharts_top_level"
OUT.mkdir(parents=True, exist_ok=True)

COL = {
    "blue":"#0C447C", "green":"#0F6E56", "purple":"#3C3489", "red":"#993C1D",
    "dry":"#3B6D11", "decision":"#854F0B", "neutral":"#5F5E5A", "white":"#FFFFFF",
    "text":"#111111", "edge":"#263238", "hil":"#D85A30", "gate":"#7A5C12",
    "loop":"#3D8FD9", "panel":"#F8FAFC", "panel_border":"#D9E1E8", "detail_border":"#8A887F",
}
TARGET_GAP = 2.0


def esc(v): return html.escape(str(v))

@dataclass(frozen=True)
class R:
    x: float; y: float; w: float; h: float
    @property
    def left(self): return self.x
    @property
    def right(self): return self.x + self.w
    @property
    def top(self): return self.y
    @property
    def bottom(self): return self.y + self.h
    @property
    def cx(self): return self.x + self.w/2
    @property
    def cy(self): return self.y + self.h/2
    def a(self, side, gap=0.0):
        if side == "top": return (self.cx, self.top-gap)
        if side == "bottom": return (self.cx, self.bottom+gap)
        if side == "left": return (self.left-gap, self.cy)
        if side == "right": return (self.right+gap, self.cy)
        raise ValueError(side)

@dataclass(frozen=True)
class D:
    cx: float; cy: float; w: float; h: float
    @property
    def left(self): return self.cx-self.w/2
    @property
    def right(self): return self.cx+self.w/2
    @property
    def top(self): return self.cy-self.h/2
    @property
    def bottom(self): return self.cy+self.h/2
    def a(self, side, gap=0.0):
        if side == "top": return (self.cx, self.top-gap)
        if side == "bottom": return (self.cx, self.bottom+gap)
        if side == "left": return (self.left-gap, self.cy)
        if side == "right": return (self.right+gap, self.cy)
        raise ValueError(side)


def rect_svg(g, fill, rx=14, stroke="none", sw=0):
    return f'<rect x="{g.x}" y="{g.y}" width="{g.w}" height="{g.h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>'

def diamond_svg(g, fill):
    return f'<polygon points="{g.cx},{g.top} {g.right},{g.cy} {g.cx},{g.bottom} {g.left},{g.cy}" fill="{fill}"/>'

def label(x,y,text,fill,size=14,weight=700,anchor="middle"):
    return f'<text x="{x}" y="{y}" text-anchor="{anchor}" fill="{fill}" font-size="{size}" font-weight="{weight}">{esc(text)}</text>'

def text_lines(x,y,title,subs=(),title_size=17,sub_size=13,line_gap=20,fill="#FFFFFF",anchor="middle"):
    out=[f'<text x="{x}" y="{y}" text-anchor="{anchor}" class="node-title" font-size="{title_size}" fill="{fill}">{esc(title)}</text>']
    yy=y+line_gap
    for s in subs:
        out.append(f'<text x="{x}" y="{yy}" text-anchor="{anchor}" class="node-sub" font-size="{sub_size}" fill="{fill}" opacity="0.90">{esc(s)}</text>')
        yy += line_gap-2
    return "\n".join(out)

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

def path(points, cls="edge-dark", marker=True):
    points=_clean_points(points)
    markers={"edge-dark":"arrowDark","edge-hil":"arrowHil","edge-gate":"arrowGate","edge-loop":"arrowLoop","edge-assoc":"arrowAssoc"}
    d="M"+" L".join(f"{x:g} {y:g}" for x,y in points)
    m=f' marker-end="url(#{markers[cls]})"' if marker else ""
    return f'<path d="{d}" class="{cls}"{m}/>'

def direct(src,ss,dst,ds,cls="edge-dark",gap=TARGET_GAP): return path([src.a(ss),dst.a(ds,gap)],cls)

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
    svg=header(w,h,title)+"\n"+body+"\n</svg>"
    sp=OUT/f"{name}.svg"; pp=OUT/f"{name}.pdf"; pn=OUT/f"{name}.png"
    sp.write_text(svg,encoding="utf-8")
    cairosvg.svg2pdf(bytestring=svg.encode(),write_to=str(pp))
    cairosvg.svg2png(bytestring=svg.encode(),write_to=str(pn),output_width=w*2,output_height=h*2)
    return sp,pp,pn

def draw_node(out,g,fill,title,subs=(),ts=16,ss=12,rx=14):
    out.append(rect_svg(g,fill,rx)); base=g.y+(28 if subs else g.h/2+6)
    out.append(text_lines(g.cx,base,title,subs,ts,ss,19))

def audit_rect_bounds(name,g,W,H,m=0):
    assert g.left>=m and g.top>=m and g.right<=W-m and g.bottom<=H-m, f"{name}: bounds"


def build_ieee():
    W,H=820,1120
    out=[rect_svg(R(34,26,752,1060),COL["panel"],20,COL["panel_border"],2)]
    out.append(label(W/2,60,"Scenario 1 - Top-Level Baseline Flow",COL["text"],22,700))
    cx=410; bw=430; x=cx-bw/2
    run=R(x,92,bw,64)
    prep=R(x,190,bw,90)
    reset=R(x,316,bw,66)
    controls=R(x,418,bw,88)
    writer=R(x,542,bw,82)
    execphase=R(x,664,bw,94)
    result=R(x,800,bw,76)
    pubend=R(x,916,bw,66)
    ret=R(x,1020,bw,48)

    draw_node(out,run,COL["blue"],"run_scenario_1()",("network, profiles, voltage limits",),18,13)
    draw_node(out,prep,COL["blue"],"Prepare time axis + profiles",("adapt_profiles(); on_scenario_start()", "integer-index DER/load tables; validate all time steps"),16.5,12)
    draw_node(out,reset,COL["blue"],"Reset network state",("drop stale controllers; pp.reset_results()",),16,12)
    draw_node(out,controls,COL["green"],"Attach profile-driven ConstControl",("sgen p_mw if DER profiles exist", "load p_mw + q_mvar if load profiles exist"),16,11.5)
    draw_node(out,writer,COL["green"],"Configure OutputWriter",("log V, line/trafo loading, losses, ext_grid P",),16,11.5)
    draw_node(out,execphase,COL["neutral"],"Annual sweep + post-processing pass",("run_timeseries() owns the PF sweep", "then convert logged tables to records; see Diagram 2"),16,11.5)
    draw_node(out,result,COL["purple"],"ScenarioResult.from_records()",("aggregate all baseline records",),16,12)
    draw_node(out,pubend,COL["purple"],"on_scenario_end()",("close publisher handle; final live event",),16,12)
    draw_node(out,ret,COL["blue"],"return ScenarioResult",(),16)

    chain=[run,prep,reset,controls,writer,execphase,result,pubend,ret]
    for a,b in zip(chain,chain[1:]): out.append(direct(a,"bottom",b,"top"))

    for n,g in {"run":run,"prep":prep,"reset":reset,"controls":controls,"writer":writer,"execphase":execphase,"result":result,"pubend":pubend,"ret":ret}.items(): audit_rect_bounds(n,g,W,H,30)
    assert prep.top-run.bottom>=30; assert reset.top-prep.bottom>=30; assert controls.top-reset.bottom>=30
    assert writer.top-controls.bottom>=30; assert execphase.top-writer.bottom>=38; assert result.top-execphase.bottom>=38
    assert pubend.top-result.bottom>=36; assert ret.top-pubend.bottom>=36

    write("flow_s1_top_ieee_final",W,H,"Scenario 1 top-level baseline flow - IEEE","\n".join(out))
    return W,H


def build_presentation():
    W,H=1920,1080
    out=[rect_svg(R(0,0,W,H),COL["panel"],0)]
    main=R(28,24,1040,1032); out.append(rect_svg(main,"#F4F8FB",24,COL["panel_border"],1.5))
    cx=500; bw=720; x=cx-bw/2
    run=R(x,46,bw,58); prep=R(x,134,bw,74); reset=R(x,238,bw,58)
    controls=R(x,326,bw,76); writer=R(x,432,bw,70); execphase=R(x,536,bw,82)
    result=R(x,652,bw,68); pubend=R(x,754,bw,62); ret=R(x,850,bw,48)

    draw_node(out,run,COL["blue"],"run_scenario_1()",("network, profiles, voltage limits",),24,16)
    draw_node(out,prep,COL["blue"],"Prepare profiles + integer timestep axis",("adapt_profiles(); on_scenario_start()", "reindex DER/load tables to 0...T-1 and validate coverage"),21,14)
    draw_node(out,reset,COL["blue"],"Reset stale controllers / results",(),21)
    draw_node(out,controls,COL["green"],"Attach ConstControl profile sources",("DER active power; load active + reactive power",),21,14)
    draw_node(out,writer,COL["green"],"Configure OutputWriter",("V/loading + losses + ext_grid P",),21,14)
    draw_node(out,execphase,COL["neutral"],"run_timeseries sweep + logged-result post-processing",("two distinct execution phases; see detailed figure",),21,14)
    draw_node(out,result,COL["purple"],"ScenarioResult.from_records()",("baseline summary from complete record list",),20,14)
    draw_node(out,pubend,COL["purple"],"on_scenario_end()",("final publisher event / handle close",),20,14)
    draw_node(out,ret,COL["blue"],"return ScenarioResult",(),20)
    chain=[run,prep,reset,controls,writer,execphase,result,pubend,ret]
    for a,b in zip(chain,chain[1:]): out.append(direct(a,"bottom",b,"top"))

    px=1120; pw=760
    p1=R(px,28,pw,280); p2=R(px,330,pw,300); p3=R(px,652,pw,340)
    for p in (p1,p2,p3): out.append(rect_svg(p,COL["white"],16,COL["detail_border"],1.2))

    out.append(label(px+22,60,"Baseline execution contract",COL["text"],18,700,"start"))
    lines=[
        "No manual control loop, OLTC, SVC, Volt-Var, OPF, curtailment, or hardware path.",
        "run_timeseries() owns the annual power-flow sweep through pandapower controllers.",
        "Violation detection is deliberately deferred until after the entire sweep.",
        "continue_on_divergence=True allows later timesteps to run after an individual PF failure.",
        "The runner controls DER p_mw only; it does not explicitly overwrite sgen q_mvar.",
    ]
    for i,t in enumerate(lines): out.append(label(px+22,96+36*i,t,COL["text"],12.2,600,"start"))

    out.append(label(px+22,362,"Publisher / checkpoint semantics",COL["text"],18,700,"start"))
    lines=[
        "enable_checkpointing and live_csv_rewrite_fn are accepted but not used for resume/progress logic.",
        "Scenario 1 never calls get_resume_records(), so a prior checkpoint cannot skip or resume the sweep.",
        "on_timestep() is called only while post-processing converged logged timesteps.",
        "A non-converged logged timestep is appended directly and therefore is not sent to on_timestep().",
        "PublishHandle may still write checkpoint lines during post-processing, but this runner will not consume them on restart.",
    ]
    for i,t in enumerate(lines): out.append(label(px+22,400+39*i,t,COL["text"],12.0,600,"start"))

    out.append(label(px+22,684,"Data and timing semantics",COL["text"],18,700,"start"))
    lines=[
        "DFData/ConstControl use integer timestep rows and actual element indices as profile columns.",
        "OutputWriter logs vm_pu, line/trafo loading, line/trafo losses, and ext_grid p_mw.",
        "_timed_runpp() records runpp wall time keyed by OutputWriter.time_step.",
        "Source comment says the timing map is cleared per call, but the supplied runner does not call _RUNPP_TIMING.clear().",
        "That discrepancy is an implementation note, not silently changed by the diagram.",
    ]
    for i,t in enumerate(lines): out.append(label(px+22,722+41*i,t,COL["text"],12.0,600,"start"))

    for n,g in {"run":run,"prep":prep,"reset":reset,"controls":controls,"writer":writer,"execphase":execphase,"result":result,"pubend":pubend,"ret":ret}.items(): audit_rect_bounds(n,g,W,H,18)
    assert main.right < p1.left
    assert prep.top-run.bottom>=30; assert reset.top-prep.bottom>=28; assert controls.top-reset.bottom>=28
    assert writer.top-controls.bottom>=28; assert execphase.top-writer.bottom>=32; assert result.top-execphase.bottom>=30
    assert pubend.top-result.bottom>=30; assert ret.top-pubend.bottom>=30

    write("flow_s1_top_presentation_final",W,H,"Scenario 1 top-level baseline flow - presentation","\n".join(out))
    return W,H


if __name__ == "__main__":
    iw,ih=build_ieee(); pw,ph=build_presentation()
    print(f"IEEE top-level audited: {iw} x {ih}")
    print(f"Presentation top-level audited: {pw} x {ph}")
    print(f"Outputs: {OUT}")
