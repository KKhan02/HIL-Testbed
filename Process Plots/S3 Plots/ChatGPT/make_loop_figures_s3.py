from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import html
import cairosvg

OUT = Path(__file__).resolve().parent / "s3_flowcharts_per_timestep"
OUT.mkdir(parents=True, exist_ok=True)

COL = {
    "blue":"#0C447C", "green":"#0F6E56", "purple":"#3C3489", "red":"#993C1D",
    "dry":"#3B6D11", "decision":"#854F0B", "neutral":"#5F5E5A", "white":"#FFFFFF",
    "text":"#111111", "edge":"#263238", "hil":"#D85A30", "gate":"#7A5C12",
    "loop":"#3D8FD9", "panel":"#F8FAFC", "panel_border":"#D9E1E8", "detail_border":"#8A887F",
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
def overlaps(a,b,pad=0): return not(a.right+pad<=b.left or b.right+pad<=a.left or a.bottom+pad<=b.top or b.bottom+pad<=a.top)

def build_ieee():
    W,H=900,1580; out=[rect_svg(R(14,14,W-28,H-28),COL["panel"],16,COL["panel_border"],1.5)]
    rects={}; diamonds={}
    def Rg(n,x,y,w,h): g=R(x,y,w,h); rects[n]=g; return g
    def Dg(n,x,y,w,h): g=D(x,y,w,h); diamonds[n]=g; return g
    cx=420; failure_rail=58; loop_rail=42; mr=12
    out.append(label(W/2,36,"Scenario 3 - Per-Timestep SVC Execution",COL["text"],17,700))

    loop=Dg("loop",cx,82,150,50); out.append(diamond_svg(loop,COL["neutral"])); out.append(text_lines(cx,78,"for each t",("in time_steps",),14,10,14))
    state=Rg("state",220,124,400,66); draw_node(out,state,COL["blue"],"[1] write timestep state",("load P/Q; DER P=profile, DER Q=0", "SVC q_mvar=0 before pre-PF"),12.5,9); out.append(direct(loop,"bottom",state,"top"))
    pre=Rg("pre",220,220,400,54); draw_node(out,pre,COL["blue"],"[2] pre-control PF",("pp.runpp(voltage_depend_loads=False)",),13,9); out.append(direct(state,"bottom",pre,"top"))
    dpre=Dg("dpre",cx,330,214,64); draw_decision(out,dpre,"pre-PF converged?",ts=13); out.append(direct(pre,"bottom",dpre,"top"))
    prefail=Rg("prefail",72,291,170,78)
    draw_node(out,prefail,COL["red"],"Pre-PF failure",
            ("append failed record", "q=0; no publish; continue"),11,8.5,10)
    out.append(direct(dpre,"left",prefail,"right","edge-hil"))
    out.append(label(dpre.left-8,dpre.cy-10,"no",COL["hil"],10,700,"end"))

    read=Rg("read",220,398,400,48); draw_node(out,read,COL["blue"],"[3] read vm_pu at SVC bus",(),13); out.append(direct(dpre,"bottom",read,"top")); out.append(label(dpre.cx+10,dpre.bottom+18,"yes",COL["text"],10,700,"start"))
    droop=Rg("droop",220,474,400,68); draw_node(out,droop,COL["green"],"_droop_q(vm_svc, Q_MAX, k_q)",("deadbanded droop; clip +/-Q_MAX", "returns q_cmd + saturated flag"),12,9); out.append(direct(read,"bottom",droop,"top"))
    dq=Dg("dq",cx,600,210,62); draw_decision(out,dq,"|q_cmd| > 0?",ts=13); out.append(direct(droop,"bottom",dq,"top"))

    reuse=Rg("reuse",102,650,220,64); post=Rg("post",520,650,300,64)
    draw_node(out,reuse,COL["neutral"],"Deadband path",("reuse pre-PF results", "converged_post=True"),11,9,10)
    draw_node(out,post,COL["blue"],"[4] post-SVC PF",("write SVC q=q_cmd; try runpp()", "exception -> converged_post=False"),11.5,9,10)
    out.append(ortho_hv(dq,"left",reuse,"top","edge-gate")); out.append(label(dq.left-8,dq.cy-8,"no",COL["gate"],10,700,"end"))
    out.append(ortho_hv(dq,"right",post,"top")); out.append(label(dq.right+8,dq.cy-8,"yes",COL["text"],10,700,"start"))
    mpost=C(cx,752,mr); out.append(ortho_vh(reuse,"bottom",mpost,"left","edge-gate")); out.append(ortho_vh(post,"bottom",mpost,"right")); out.append(circle_svg(mpost,COL["neutral"]))
    dconv=Dg("dconv",cx,820,210,60); draw_decision(out,dconv,"converged_post?",ts=13); out.append(direct(mpost,"bottom",dconv,"top"))

    empty=Rg("empty",92,868,230,68)
    extract=Rg("extract",480,868,340,78)
    draw_node(out,empty,COL["red"],"Non-converged state",("empty V/loading series", "losses/import=None"),11,9,10)
    draw_node(out,extract,COL["blue"],"Extract final network state",("copy V / line / trafo results", "threshold violations; losses + grid import"),11.5,9,10)
    out.append(ortho_hv(dconv,"left",empty,"top","edge-hil")); out.append(label(dconv.left-8,dconv.cy-8,"no",COL["hil"],10,700,"end"))
    out.append(ortho_hv(dconv,"right",extract,"top")); out.append(label(dconv.right+8,dconv.cy-8,"yes",COL["text"],10,700,"start"))
    mrec=C(cx,982,mr); out.append(ortho_vh(empty,"bottom",mrec,"left","edge-hil")); out.append(ortho_vh(extract,"bottom",mrec,"right")); out.append(circle_svg(mrec,COL["neutral"]))

    record=Rg("record",190,1016,460,76); draw_node(out,record,COL["purple"],"[5] build normal TimestepRecord",("svc_q_mvar=q_cmd; svc_saturated=saturated", "publish_fn.on_timestep(rec) -> records.append(rec)"),12,9); out.append(direct(mrec,"bottom",record,"top"))
    periodic=Dg("periodic",cx,1148,198,54); draw_decision(out,periodic,"t % 96 == 0?",ts=12); out.append(direct(record,"bottom",periodic,"top"))
    progress=Rg("progress",640,1117,240,62); draw_node(out,progress,COL["purple"],"Periodic progress",("optional partial result -> live CSV", "log vm_svc / q_cmd / violations"),10.5,8.3,10); out.append(direct(periodic,"right",progress,"left","edge-gate")); out.append(label(periodic.right+8,periodic.cy-8,"yes",COL["gate"],10,700,"start"))
    mnext=C(cx,1222,mr); out.append(direct(periodic,"bottom",mnext,"top")); out.append(label(periodic.cx+10,periodic.bottom+16,"no",COL["text"],10,700,"start")); out.append(ortho_vh(progress,"bottom",mnext,"right","edge-gate",bend_y=mnext.cy))

    # Pre-PF failure bypasses normal publisher and periodic work, entering only the iteration-complete merge.
    out.append(path([prefail.a("left"),(failure_rail,prefail.cy)],"edge-hil",marker=False))
    out.append(path([(failure_rail,prefail.cy),(failure_rail,mnext.cy)],"edge-hil",marker=False))
    out.append(path([(failure_rail,mnext.cy),mnext.a("left",TARGET_GAP)],"edge-hil"))
    out.append(circle_svg(mnext,COL["neutral"]))

    more=Dg("more",cx,1300,174,48); draw_decision(out,more,"more timesteps?",ts=12); out.append(direct(mnext,"bottom",more,"top"))
    result=Rg("result",540,1278,220,44); draw_node(out,result,COL["purple"],"loop complete",("return to top-level cleanup",),11,8.5,9); out.append(direct(more,"right",result,"left")); out.append(label(more.right+8,more.cy-7,"no",COL["text"],10,700,"start"))
    loopback_y=1370; out.append(path([more.a("bottom"),(more.cx,loopback_y),(loop_rail,loopback_y),(loop_rail,loop.cy),loop.a("left",TARGET_GAP)],"edge-loop")); out.append(label(more.cx+12,more.bottom+16,"yes - next t",COL["loop"],10,700,"start"))

    for n,g in rects.items(): audit_rect_bounds(n,g,W,H,8)
    for n,g in diamonds.items(): audit_diamond_bounds(n,g,W,H,8)
    assert state.top-loop.bottom>=17; assert pre.top-state.bottom>=28; assert dpre.top-pre.bottom>=24; assert read.top-dpre.bottom>=36
    assert droop.top-read.bottom>=28; assert dq.top-droop.bottom>=27; assert mpost.top-max(reuse.bottom,post.bottom)>=26
    assert dconv.top-mpost.bottom>=24; assert mrec.top-max(empty.bottom,extract.bottom)>=24; assert record.top-mrec.bottom>=22
    assert periodic.top-record.bottom>=29; assert mnext.top-periodic.bottom>=35; assert more.top-mnext.bottom>=42
    assert loop_rail < failure_rail < prefail.left-8; assert result.bottom < loopback_y-20; assert loopback_y < H-30
    assert abs(prefail.cy - dpre.cy) < 1e-9

    write("flow_s3_loop_ieee_final",W,H,"Scenario 3 per-timestep SVC flow - IEEE","\n".join(out)); return W,H


def build_presentation():
    W,H=1920,1080; out=[rect_svg(R(0,0,W,H),COL["panel"],0)]; rects={}; diamonds={}
    def Rg(n,x,y,w,h): g=R(x,y,w,h); rects[n]=g; return g
    def Dg(n,x,y,w,h): g=D(x,y,w,h); diamonds[n]=g; return g
    cx=470; mr=13; failure_rail=42; loop_rail=18

    loop=Dg("loop",cx,24,180,36); out.append(diamond_svg(loop,COL["neutral"])); out.append(text_lines(cx,21,"for each t",("in time_steps",),14,9,12))
    state=Rg("state",180,50,580,46); draw_node(out,state,COL["blue"],"[1] write load / DER profiles; reset DER Q and SVC Q to 0",(),14); out.append(direct(loop,"bottom",state,"top"))
    pre=Rg("pre",180,120,580,46); draw_node(out,pre,COL["blue"],"[2] pre-control PF",("pp.runpp(voltage_depend_loads=False)",),14,9); out.append(direct(state,"bottom",pre,"top"))
    dpre=Dg("dpre",cx,216,240,48); draw_decision(out,dpre,"pre-PF converged?",ts=13); out.append(direct(pre,"bottom",dpre,"top"))
    prefail=Rg("prefail",60,189,220,54)
    draw_node(out,prefail,COL["red"],"Pre-PF failure",
            ("append failed record; no publish; continue",),11.5,8.5,10)
    out.append(direct(dpre,"left",prefail,"right","edge-hil"))
    out.append(label(dpre.left-10,dpre.cy-8,"no",COL["hil"],11,700,"end"))    
    read=Rg("read",180,272,580,40); draw_node(out,read,COL["blue"],"[3] read vm_svc at fixed SVC bus",(),14); out.append(direct(dpre,"bottom",read,"top")); out.append(label(dpre.cx+12,dpre.bottom+16,"yes",COL["text"],11,700,"start"))
    droop=Rg("droop",180,336,580,50); draw_node(out,droop,COL["green"],"_droop_q() -> q_cmd, saturated",("deadbanded droop clipped to +/-Q_MAX",),13.5,9); out.append(direct(read,"bottom",droop,"top"))
    dq=Dg("dq",cx,438,220,48); draw_decision(out,dq,"|q_cmd| > 0?",ts=13); out.append(direct(droop,"bottom",dq,"top"))
    reuse=Rg("reuse",86,476,260,48); post=Rg("post",590,476,330,48); draw_node(out,reuse,COL["neutral"],"Deadband: reuse pre-PF",("converged_post=True",),11.5,8.5,10); draw_node(out,post,COL["blue"],"[4] write q_cmd + post-PF",("exception -> converged_post=False",),12,8.5,10)
    out.append(ortho_hv(dq,"left",reuse,"top","edge-gate")); out.append(label(dq.left-10,dq.cy-8,"no",COL["gate"],11,700,"end")); out.append(ortho_hv(dq,"right",post,"top")); out.append(label(dq.right+10,dq.cy-8,"yes",COL["text"],11,700,"start"))
    mpost=C(cx,552,mr); out.append(ortho_vh(reuse,"bottom",mpost,"left","edge-gate")); out.append(ortho_vh(post,"bottom",mpost,"right")); out.append(circle_svg(mpost,COL["neutral"]))
    dconv=Dg("dconv",cx,610,220,46); draw_decision(out,dconv,"converged_post?",ts=13); out.append(direct(mpost,"bottom",dconv,"top"))
    empty=Rg("empty",78,646,270,52); extract=Rg("extract",570,642,360,60); draw_node(out,empty,COL["red"],"Non-converged result",("empty V/loading; no losses/import",),11.5,8.5,10); draw_node(out,extract,COL["blue"],"Extract final state",("V/loading; violations; losses + grid import",),12,8.5,10)
    out.append(ortho_hv(dconv,"left",empty,"top","edge-hil")); out.append(label(dconv.left-10,dconv.cy-8,"no",COL["hil"],11,700,"end")); out.append(ortho_hv(dconv,"right",extract,"top")); out.append(label(dconv.right+10,dconv.cy-8,"yes",COL["text"],11,700,"start"))
    mrec=C(cx,736,mr); out.append(ortho_vh(empty,"bottom",mrec,"left","edge-hil")); out.append(ortho_vh(extract,"bottom",mrec,"right")); out.append(circle_svg(mrec,COL["neutral"]))
    record=Rg("record",150,764,640,52); draw_node(out,record,COL["purple"],"[5] build record -> publish -> append",("publish_fn.on_timestep(rec) occurs before records.append(rec)",),12.5,9); out.append(direct(mrec,"bottom",record,"top"))
    periodic=Dg("periodic",cx,866,220,46); draw_decision(out,periodic,"t % 96 == 0?",ts=13); out.append(direct(record,"bottom",periodic,"top"))
    progress=Rg("progress",690,839,340,54); draw_node(out,progress,COL["purple"],"Periodic progress",("optional partial result -> live CSV; logger.info",),11.5,8.5,10); out.append(direct(periodic,"right",progress,"left","edge-gate")); out.append(label(periodic.right+10,periodic.cy-8,"yes",COL["gate"],11,700,"start"))
    mnext=C(cx,930,mr); out.append(direct(periodic,"bottom",mnext,"top")); out.append(label(periodic.cx+12,periodic.bottom+17,"no",COL["text"],11,700,"start")); out.append(ortho_vh(progress,"bottom",mnext,"right","edge-gate",bend_y=mnext.cy))
    out.append(path([prefail.a("left"),(failure_rail,prefail.cy)],"edge-hil",marker=False)); out.append(path([(failure_rail,prefail.cy),(failure_rail,mnext.cy)],"edge-hil",marker=False)); out.append(path([(failure_rail,mnext.cy),mnext.a("left",TARGET_GAP)],"edge-hil")); out.append(circle_svg(mnext,COL["neutral"]))
    more=Dg("more",760,984,200,44); draw_decision(out,more,"more timesteps?",ts=13); out.append(path([mnext.a("bottom"),(mnext.cx,more.cy),more.a("left",TARGET_GAP)],"edge-dark"))
    done=Rg("done",900,962,180,44); draw_node(out,done,COL["purple"],"loop complete",(),12); out.append(direct(more,"right",done,"left")); out.append(label(more.right+10,more.cy-8,"no",COL["text"],11,700,"start"))
    loopback_y=1045; out.append(path([more.a("bottom"),(more.cx,loopback_y),(loop_rail,loopback_y),(loop_rail,loop.cy),loop.a("left",TARGET_GAP)],"edge-loop")); out.append(label(more.cx+14,more.bottom+18,"yes - next t",COL["loop"],11,700,"start"))

    # Right-side implementation panels.
    px=1120; pw=760; panels=[R(px,24,pw,276),R(px,322,pw,286),R(px,630,pw,392)]
    for p in panels: out.append(rect_svg(p,COL["white"],16,COL["detail_border"],1.2))
    out.append(label(px+22,54,"Deadbanded droop law",COL["text"],17,700,"start"))
    droop_lines=[
        "error = 1.00 - vm_svc; deadband = 0.01 pu.",
        "|error| <= 0.01 -> q_cmd=0, saturated=False.",
        "Undervoltage: q_raw = k_q*(error - 0.01), positive Q injection.",
        "Overvoltage: q_raw = k_q*(error + 0.01), negative Q absorption.",
        "q_cmd = clip(q_raw, -Q_MAX, +Q_MAX); saturated when |q_raw| >= Q_MAX.",
        "Q_MAX = 0.20*sum(trafo sn_mva); k_q = Q_MAX/0.03.",
    ]
    for i,t in enumerate(droop_lines): out.append(label(px+22,88+32*i,t,COL["text"],12.2,600,"start"))
    out.append(label(px+22,354,"PF and result semantics",COL["text"],17,700,"start"))
    pf_lines=[
        "Pre-PF always starts with DER Q=0 and SVC Q=0.",
        "q_cmd=0 skips the second runpp and reuses the valid pre-PF result tables.",
        "A post-PF exception does not abort the scenario; the timestep is recorded converged=False.",
        "Converged records threshold V, line loading, and trafo loading directly from result tables.",
        "There is no active-P curtailment and no DER reactive-power control in Scenario 3.",
    ]
    for i,t in enumerate(pf_lines): out.append(label(px+22,390+37*i,t,COL["text"],12.2,600,"start"))
    out.append(label(px+22,662,"Publisher / failure semantics and legend",COL["text"],17,700,"start"))
    sem=[
        "Normal path: build rec -> publish_fn.on_timestep(rec) -> records.append(rec).",
        "PublishHandle checkpoints every normal-path timestep; dashboard live frames use update_every_k.",
        "Pre-PF failure path is different: records.append(failed_record) then continue.",
        "Therefore pre-PF failures skip on_timestep(), checkpoint/live publishing, and the t%96 branch.",
        "The red failure bypass rail and the blue next-timestep loopback rail are separate.",
        "After the loop, top-level finally removes the temporary SVC before final aggregation.",
    ]
    for i,t in enumerate(sem): out.append(label(px+22,700+36*i,t,COL["text"],12.1,600,"start"))
    ly=940
    for i,(cls,txt,col) in enumerate([("edge-dark","normal execution",COL["text"]),("edge-hil","failure / non-converged path",COL["hil"]),("edge-gate","optional / deadband branch",COL["gate"]),("edge-loop","next-timestep return",COL["loop"])]):
        yy=ly+22*i; out.append(path([(px+30,yy),(px+92,yy)],cls)); out.append(label(px+108,yy+4,txt,col,10.5,600,"start"))

    for n,g in rects.items(): audit_rect_bounds(n,g,W,H,4)
    for n,g in diamonds.items(): audit_diamond_bounds(n,g,W,H,4)
    assert pre.top-state.bottom>=20; assert dpre.top-pre.bottom>=24; assert read.top-dpre.bottom>=32; assert droop.top-read.bottom>=24
    assert dq.top-droop.bottom>=28; assert mpost.top-max(reuse.bottom,post.bottom)>=15; assert dconv.top-mpost.bottom>=22
    assert mrec.top-max(empty.bottom,extract.bottom)>=21; assert record.top-mrec.bottom>=15; assert periodic.top-record.bottom>=27; assert mnext.top-periodic.bottom>=18
    assert loop_rail < failure_rail < prefail.left-6; assert done.bottom < loopback_y-15; assert loopback_y < H-20; assert panels[0].left>1080
    assert abs(prefail.cy - dpre.cy) < 1e-9

    write("flow_s3_loop_presentation_final",W,H,"Scenario 3 per-timestep SVC flow - presentation","\n".join(out)); return W,H

if __name__=="__main__":
    iw,ih=build_ieee(); pw,ph=build_presentation(); print(f"IEEE loop audited: {iw} x {ih}"); print(f"Presentation loop audited: {pw} x {ph}"); print(f"Outputs: {OUT}")
