from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import html
import cairosvg

OUT = Path(__file__).resolve().parent / "s1_flowcharts_per_timestep"
OUT.mkdir(parents=True, exist_ok=True)

COL = {
    "blue":"#0C447C", "green":"#0F6E56", "purple":"#3C3489", "red":"#993C1D",
    "decision":"#854F0B", "neutral":"#5F5E5A", "white":"#FFFFFF", "text":"#111111",
    "hil":"#D85A30", "gate":"#7A5C12", "loop":"#3D8FD9", "panel":"#F8FAFC",
    "panel_border":"#D9E1E8", "detail_border":"#8A887F",
}
TARGET_GAP=2.0

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

def rect_svg(g,fill,rx=12,stroke="none",sw=0): return f'<rect x="{g.x}" y="{g.y}" width="{g.w}" height="{g.h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>'
def diamond_svg(g,fill): return f'<polygon points="{g.cx},{g.top} {g.right},{g.cy} {g.cx},{g.bottom} {g.left},{g.cy}" fill="{fill}"/>'
def circle_svg(g,fill): return f'<circle cx="{g.cx}" cy="{g.cy}" r="{g.r}" fill="{fill}"/>'
def label(x,y,text,fill,size=13,weight=700,anchor="middle"): return f'<text x="{x}" y="{y}" text-anchor="{anchor}" fill="{fill}" font-size="{size}" font-weight="{weight}">{esc(text)}</text>'

def text_lines(x,y,title,subs=(),title_size=15,sub_size=11,line_gap=16,fill="#FFFFFF",anchor="middle"):
    o=[f'<text x="{x}" y="{y}" text-anchor="{anchor}" class="node-title" font-size="{title_size}" fill="{fill}">{esc(title)}</text>']; yy=y+line_gap
    for s in subs:
        o.append(f'<text x="{x}" y="{yy}" text-anchor="{anchor}" class="node-sub" font-size="{sub_size}" fill="{fill}" opacity="0.90">{esc(s)}</text>'); yy+=line_gap-2
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
.edge-dark{{fill:none;stroke:#263238;stroke-width:2.6;stroke-linejoin:round;stroke-linecap:round}}
.edge-hil{{fill:none;stroke:#D85A30;stroke-width:2.8;stroke-linejoin:round;stroke-linecap:round}}
.edge-gate{{fill:none;stroke:#7A5C12;stroke-width:2.6;stroke-linejoin:round;stroke-linecap:round}}
.edge-loop{{fill:none;stroke:#3D8FD9;stroke-width:2.6;stroke-dasharray:7 5;stroke-linejoin:round;stroke-linecap:round}}
.edge-assoc{{fill:none;stroke:#8A887F;stroke-width:2;stroke-dasharray:5 4;stroke-linejoin:round;stroke-linecap:round}}
</style>'''

def write(name,w,h,title,body):
    svg=header(w,h,title)+"\n"+body+"\n</svg>"; sp=OUT/f"{name}.svg"; pp=OUT/f"{name}.pdf"; pn=OUT/f"{name}.png"
    sp.write_text(svg,encoding="utf-8"); cairosvg.svg2pdf(bytestring=svg.encode(),write_to=str(pp)); cairosvg.svg2png(bytestring=svg.encode(),write_to=str(pn),output_width=w*2,output_height=h*2)
    return sp,pp,pn

def draw_node(out,g,fill,title,subs=(),ts=14,ss=10,rx=12): out.extend([rect_svg(g,fill,rx),text_lines(g.cx,g.y+(20 if subs else g.h/2+5),title,subs,ts,ss,15)])
def draw_decision(out,g,title,subs=(),ts=13,ss=10): out.extend([diamond_svg(g,COL["decision"]),text_lines(g.cx,g.cy+(4 if not subs else -4),title,subs,ts,ss,14)])
def audit_rect_bounds(name,g,W,H,m=0): assert g.left>=m and g.top>=m and g.right<=W-m and g.bottom<=H-m, f"{name}: bounds"
def audit_diamond_bounds(name,g,W,H,m=0): assert g.left>=m and g.right<=W-m and g.top>=m and g.bottom<=H-m, f"{name}: bounds"


def build_ieee():
    W,H=920,1430
    out=[rect_svg(R(14,14,W-28,H-28),COL["panel"],16,COL["panel_border"],1.5)]
    rects={}; diamonds={}
    def Rg(n,x,y,w,h): g=R(x,y,w,h); rects[n]=g; return g
    def Dg(n,x,y,w,h): g=D(x,y,w,h); diamonds[n]=g; return g
    cx=460; engine_rail=42; failure_rail=58; post_loop_rail=34; mr=12
    out.append(label(W/2,36,"Scenario 1 - Annual Sweep + Post-Processing",COL["text"],17,700))

    # Phase A: run_timeseries-owned sweep
    start=Rg("start",230,66,460,58); draw_node(out,start,COL["green"],"[4] ts.run_timeseries()",("continue_on_divergence=True; run=_timed_runpp",),13,9)
    partA=Rg("partA",90,150,740,440); out.append(rect_svg(partA,"none",16,COL["detail_border"],1.0)); out.append(label(partA.x+16,partA.y-6,"Phase A - pandapower run_timeseries sweep",COL["text"],10.5,700,"start"))
    eloop=Dg("eloop",cx,196,170,48); out.append(diamond_svg(eloop,COL["neutral"])); out.append(text_lines(cx,192,"for each engine t",(),13,9,14)); out.append(direct(start,"bottom",eloop,"top"))
    control=Rg("control",240,244,440,60); draw_node(out,control,COL["green"],"ConstControl writes profile values",("sgen p_mw; load p_mw / q_mvar from DFData[t]",),12.2,8.7); out.append(direct(eloop,"bottom",control,"top"))
    runpp=Rg("runpp",240,334,440,60); draw_node(out,runpp,COL["green"],"_timed_runpp() -> pp.runpp()",("record elapsed ms keyed by OutputWriter.time_step",),12.2,8.7); out.append(direct(control,"bottom",runpp,"top"))
    writer=Rg("writer",240,424,440,72); draw_node(out,writer,COL["green"],"OutputWriter logs settled result tables",("vm_pu; line/trafo loading; line/trafo losses", "ext_grid p_mw"),12,8.5); out.append(direct(runpp,"bottom",writer,"top"))
    emore=Dg("emore",cx,548,190,48); draw_decision(out,emore,"more engine timesteps?",ts=11.8); out.append(direct(writer,"bottom",emore,"top"))
    out.append(path([emore.a("left"),(engine_rail,emore.cy),(engine_rail,eloop.cy),eloop.a("left",TARGET_GAP)],"edge-loop")); out.append(label(emore.left-8,emore.cy-7,"yes",COL["loop"],10,700,"end"))

    # Handoff after opaque sweep
    logs=Rg("logs",230,628,460,72); draw_node(out,logs,COL["blue"],"Read OutputWriter logs",("vm_log / loading / losses / grid import tables",),12.5,9); out.append(path([emore.a("bottom"),(emore.cx,604),(logs.cx,604),logs.a("top",TARGET_GAP)],"edge-dark")); out.append(label(emore.cx+10,emore.bottom+17,"no - sweep complete",COL["text"],10,700,"start"))

    # Phase B: runner-owned post-processing
    partB=Rg("partB",90,738,740,664); out.append(rect_svg(partB,"none",16,COL["detail_border"],1.0)); out.append(label(partB.x+16,partB.y-6,"Phase B - run_scenario_1() post-processing loop",COL["text"],10.5,700,"start"))
    ploop=Dg("ploop",cx,790,170,48); out.append(diamond_svg(ploop,COL["neutral"])); out.append(text_lines(cx,786,"for each logged t",(),13,9,14)); out.append(direct(logs,"bottom",ploop,"top"))
    dconv=Dg("dconv",cx,882,220,60); draw_decision(out,dconv,"logged timestep converged?",ts=12); out.append(direct(ploop,"bottom",dconv,"top"))

    failed=Rg("failed",88,844,220,76); draw_node(out,failed,COL["red"],"Non-converged logged row",("empty V/loading; losses/import=None", "append failed record; no publish"),10.5,8.2,10); out.append(direct(dconv,"left",failed,"right","edge-hil")); out.append(label(dconv.left-8,dconv.cy-8,"no",COL["hil"],10,700,"end"))
    extract=Rg("extract",210,948,500,78); draw_node(out,extract,COL["blue"],"Read converged row + threshold violations",("drop NaNs; V outside limits; line/trafo loading above limits", "derive losses, grid import, DER generation, load"),11.2,8.5); out.append(direct(dconv,"bottom",extract,"top")); out.append(label(dconv.cx+10,dconv.bottom+18,"yes",COL["text"],10,700,"start"))
    record=Rg("record",210,1060,500,74); draw_node(out,record,COL["purple"],"Build converged TimestepRecord",("publish_fn.on_timestep(rec) -> records.append(rec)", "t_total_ms from _RUNPP_TIMING[t]"),11.3,8.6); out.append(direct(extract,"bottom",record,"top"))

    mnext=C(cx,1182,mr); out.append(direct(record,"bottom",mnext,"top"))
    out.append(path([failed.a("left"),(failure_rail,failed.cy),(failure_rail,mnext.cy),mnext.a("left",TARGET_GAP)],"edge-hil")); out.append(circle_svg(mnext,COL["neutral"]))
    pmore=Dg("pmore",cx,1250,190,48); draw_decision(out,pmore,"more logged timesteps?",ts=11.8); out.append(direct(mnext,"bottom",pmore,"top"))
    out.append(path([pmore.a("left"),(post_loop_rail,pmore.cy),(post_loop_rail,ploop.cy),ploop.a("left",TARGET_GAP)],"edge-loop")); out.append(label(pmore.left-8,pmore.cy-7,"yes",COL["loop"],10,700,"end"))
    done=Rg("done",580,1228,250,44); draw_node(out,done,COL["purple"],"post-processing complete",("return records to top-level finalization",),10.5,8.2,9); out.append(direct(pmore,"right",done,"left")); out.append(label(pmore.right+8,pmore.cy-7,"no",COL["text"],10,700,"start"))

    for n,g in rects.items(): audit_rect_bounds(n,g,W,H,8)
    for n,g in diamonds.items(): audit_diamond_bounds(n,g,W,H,8)
    assert partA.top < eloop.top and partA.bottom > emore.bottom
    assert partB.top < ploop.top and partB.bottom > done.bottom
    assert control.top-eloop.bottom>=24; assert runpp.top-control.bottom>=28; assert writer.top-runpp.bottom>=28; assert emore.top-writer.bottom>=28
    assert ploop.top-logs.bottom>=66; assert dconv.top-ploop.bottom>=36; assert extract.top-dconv.bottom>=36; assert record.top-extract.bottom>=30
    assert mnext.top-record.bottom>=36; assert pmore.top-mnext.bottom>=30; assert engine_rail < partA.left-20; assert post_loop_rail < failure_rail < failed.left-20

    write("flow_s1_exec_ieee_final",W,H,"Scenario 1 annual sweep and post-processing - IEEE","\n".join(out)); return W,H


def build_presentation():
    W,H=1920,1080
    out=[rect_svg(R(0,0,W,H),COL["panel"],0)]; rects={}; diamonds={}
    def Rg(n,x,y,w,h): g=R(x,y,w,h); rects[n]=g; return g
    def Dg(n,x,y,w,h): g=D(x,y,w,h); diamonds[n]=g; return g
    cx=480; engine_rail=38; failure_rail=56; post_loop_rail=30; mr=12

    # Main execution area, no title inside slide figure.
    partA=Rg("partA",76,36,900,390); out.append(rect_svg(partA,"#FFFFFF",18,COL["detail_border"],1.2)); out.append(label(partA.x+18,partA.y+24,"Phase A - run_timeseries owns the power-flow sweep",COL["text"],14,700,"start"))
    start=Rg("start",210,72,540,42); draw_node(out,start,COL["green"],"ts.run_timeseries(... continue_on_divergence=True, run=_timed_runpp)",(),12.5)
    eloop=Dg("eloop",cx,150,180,34); out.append(diamond_svg(eloop,COL["neutral"])); out.append(text_lines(cx,147,"for each engine t",(),12,8.5,12)); out.append(direct(start,"bottom",eloop,"top"))
    control=Rg("control",210,190,540,42); draw_node(out,control,COL["green"],"ConstControl: sgen P + load P/Q <- DFData[t]",(),12.3); out.append(direct(eloop,"bottom",control,"top"))
    runpp=Rg("runpp",210,258,540,44); draw_node(out,runpp,COL["green"],"_timed_runpp() -> pp.runpp()",("store t_total_ms by OutputWriter time_step",),11.8,8.2); out.append(direct(control,"bottom",runpp,"top"))
    writer=Rg("writer",210,328,540,48); draw_node(out,writer,COL["green"],"OutputWriter logs V / loading / losses / ext_grid P",(),12.0); out.append(direct(runpp,"bottom",writer,"top"))
    emore=Dg("emore",cx,398,190,34); draw_decision(out,emore,"more engine timesteps?",ts=11.3); out.append(direct(writer,"bottom",emore,"top")); out.append(path([emore.a("left"),(engine_rail,emore.cy),(engine_rail,eloop.cy),eloop.a("left",TARGET_GAP)],"edge-loop")); out.append(label(emore.left-8,emore.cy-6,"yes",COL["loop"],9.5,700,"end"))

    handoff=Rg("handoff",210,454,540,44); draw_node(out,handoff,COL["blue"],"Sweep complete -> read OutputWriter tables",(),12.2); out.append(direct(emore,"bottom",handoff,"top")); out.append(label(emore.cx+10,emore.bottom+14,"no",COL["text"],9.5,700,"start"))

    partB=Rg("partB",76,528,900,494); out.append(rect_svg(partB,"#FFFFFF",18,COL["detail_border"],1.2)); out.append(label(partB.x+18,partB.y+24,"Phase B - run_scenario_1() converts logged rows to records",COL["text"],14,700,"start"))
    ploop=Dg("ploop",cx,574,180,34); out.append(diamond_svg(ploop,COL["neutral"])); out.append(text_lines(cx,571,"for each logged t",(),12,8.5,12)); out.append(direct(handoff,"bottom",ploop,"top"))
    dconv=Dg("dconv",cx,650,220,40); draw_decision(out,dconv,"logged timestep converged?",ts=11.4); out.append(direct(ploop,"bottom",dconv,"top"))
    failed=Rg("failed",88,624,230,52); draw_node(out,failed,COL["red"],"Non-converged row",("append failed record; no on_timestep()",),10.2,7.8,10); out.append(direct(dconv,"left",failed,"right","edge-hil")); out.append(label(dconv.left-8,dconv.cy-7,"no",COL["hil"],9.5,700,"end"))
    extract=Rg("extract",210,704,540,54); draw_node(out,extract,COL["blue"],"Read row + threshold V / line / trafo violations",("derive losses, import, DER generation, load",),11.4,8.2); out.append(direct(dconv,"bottom",extract,"top")); out.append(label(dconv.cx+10,dconv.bottom+15,"yes",COL["text"],9.5,700,"start"))
    record=Rg("record",210,786,540,52); draw_node(out,record,COL["purple"],"Build record -> publish -> append",("t_total_ms from _RUNPP_TIMING[t]",),11.3,8.2); out.append(direct(extract,"bottom",record,"top"))
    mnext=C(cx,874,mr); out.append(direct(record,"bottom",mnext,"top")); out.append(path([failed.a("left"),(failure_rail,failed.cy),(failure_rail,mnext.cy),mnext.a("left",TARGET_GAP)],"edge-hil")); out.append(circle_svg(mnext,COL["neutral"]))
    pmore=Dg("pmore",cx,930,190,36); draw_decision(out,pmore,"more logged timesteps?",ts=11.2); out.append(direct(mnext,"bottom",pmore,"top"))
    done=Rg("done",630,912,250,36); draw_node(out,done,COL["purple"],"post-process complete",(),10.8); out.append(direct(pmore,"right",done,"left")); out.append(label(pmore.right+8,pmore.cy-6,"no",COL["text"],9.5,700,"start"))
    loopback_y=988; out.append(path([pmore.a("bottom"),(pmore.cx,loopback_y),(post_loop_rail,loopback_y),(post_loop_rail,ploop.cy),ploop.a("left",TARGET_GAP)],"edge-loop")); out.append(label(pmore.cx+12,pmore.bottom+14,"yes - next row",COL["loop"],9.5,700,"start"))

    # Explanatory panels
    px=1040; pw=840
    p1=R(px,34,pw,270); p2=R(px,326,pw,286); p3=R(px,634,pw,352)
    for p in (p1,p2,p3): out.append(rect_svg(p,COL["white"],16,COL["detail_border"],1.2))
    out.append(label(px+22,64,"Phase A semantics",COL["text"],17,700,"start"))
    lines=[
        "ConstControl is the only per-step state writer in the annual sweep.",
        "run_timeseries forwards voltage_depend_loads=False to the internal run function.",
        "continue_on_divergence=True keeps the annual sweep moving after a failed PF timestep.",
        "_timed_runpp measures only runpp wall time; it does not include post-processing cost.",
        "OutputWriter is the handoff boundary: violation decisions are not made during this phase.",
    ]
    for i,t in enumerate(lines): out.append(label(px+22,100+34*i,t,COL["text"],12.0,600,"start"))

    out.append(label(px+22,358,"Phase B record semantics",COL["text"],17,700,"start"))
    lines=[
        "Convergence is inferred from the logged vm_pu row: present and not all-NaN.",
        "Converged rows are re-thresholded against V/load limits during post-processing.",
        "Losses require both line and trafo loss logs for that timestep; otherwise losses_mw=None.",
        "Grid import is taken from res_ext_grid.p_mw when the row is available.",
        "Failed rows get empty result series and are appended immediately without publisher.on_timestep().",
    ]
    for i,t in enumerate(lines): out.append(label(px+22,394+36*i,t,COL["text"],12.0,600,"start"))

    out.append(label(px+22,666,"Publisher / implementation notes",COL["text"],17,700,"start"))
    lines=[
        "No get_resume_records() call exists in Scenario 1, so the runner cannot resume from its checkpoint.",
        "live_csv_rewrite_fn is accepted but never called; there is no t%96 progress branch.",
        "on_timestep() occurs only after the annual sweep, while converged rows are being post-processed.",
        "The runner does not explicitly force sgen q_mvar to zero; it only drives DER p_mw via ConstControl.",
        "The source comment says _RUNPP_TIMING is cleared per run, but no clear() call appears in run_scenario_1().",
        "These are implementation-faithful notes, not corrections introduced by the diagram.",
    ]
    for i,t in enumerate(lines): out.append(label(px+22,704+39*i,t,COL["text"],11.7,600,"start"))

    for n,g in rects.items(): audit_rect_bounds(n,g,W,H,4)
    for n,g in diamonds.items(): audit_diamond_bounds(n,g,W,H,4)
    assert partA.bottom < handoff.top; assert handoff.bottom < partB.top
    assert engine_rail < partA.left-20; assert post_loop_rail < failure_rail < failed.left-20
    assert loopback_y < H-40; assert p1.left > partA.right+40

    write("flow_s1_exec_presentation_final",W,H,"Scenario 1 annual sweep and post-processing - presentation","\n".join(out)); return W,H


if __name__ == "__main__":
    iw,ih=build_ieee(); pw,ph=build_presentation()
    print(f"IEEE execution audited: {iw} x {ih}")
    print(f"Presentation execution audited: {pw} x {ph}")
    print(f"Outputs: {OUT}")
