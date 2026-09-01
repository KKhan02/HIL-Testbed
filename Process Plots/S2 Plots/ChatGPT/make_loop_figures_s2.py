from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import html
import cairosvg

OUT = Path(__file__).resolve().parent / "s2_flowcharts_per_timestep"
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
    W,H=980,1900
    out=[rect_svg(R(14,14,W-28,H-28),COL["panel"],16,COL["panel_border"],1.5)]
    rects={}; diamonds={}
    def Rg(n,x,y,w,h): g=R(x,y,w,h); rects[n]=g; return g
    def Dg(n,x,y,w,h): g=D(x,y,w,h); diamonds[n]=g; return g
    cx=500; loop_rail=42; failure_rail=58; mr=12
    out.append(label(W/2,36,"Scenario 2 - Per-Timestep OLTC Execution",COL["text"],17,700))

    loop=Dg("loop",cx,80,160,50); out.append(diamond_svg(loop,COL["neutral"])); out.append(text_lines(cx,76,"for each t",("in time_steps",),14,10,14))
    state=Rg("state",280,120,440,62); draw_node(out,state,COL["blue"],"[1] write timestep profiles",("load P/Q; DER P=profile; DER Q=0",),13,9); out.append(direct(loop,"bottom",state,"top"))
    pre=Rg("pre",280,210,440,54); draw_node(out,pre,COL["blue"],"[2] pre-action PF at current tap",("pp.runpp(voltage_depend_loads=False)",),13,9); out.append(direct(state,"bottom",pre,"top"))
    dpre=Dg("dpre",cx,322,220,64); draw_decision(out,dpre,"pre-PF converged?",ts=13); out.append(direct(pre,"bottom",dpre,"top"))
    prefail=Rg("prefail",90,284,190,76); draw_node(out,prefail,COL["red"],"Pre-PF failure",("append failed record", "hold tap; no publish; continue"),11,8.5,10); out.append(direct(dpre,"left",prefail,"right","edge-hil")); out.append(label(dpre.left-8,dpre.cy-9,"no",COL["hil"],10,700,"end"))

    read=Rg("read",280,398,440,58); draw_node(out,read,COL["blue"],"[3] read controlled-bus voltage",("vm_ctrl = mean(vm_pu); min/max are diagnostics",),12.5,9); out.append(direct(dpre,"bottom",read,"top")); out.append(label(dpre.cx+10,dpre.bottom+18,"yes",COL["text"],10,700,"start"))
    dctrl=Dg("dctrl",cx,520,270,78); draw_decision(out,dctrl,"vm_ctrl vs [0.98, 1.02]?",ts=12)
    out.append(direct(read,"bottom",dctrl,"top"))

    high=Rg("high",92,484,245,72); low=Rg("low",662,484,245,72)
    draw_node(out,high,COL["green"],"Overvoltage candidate",("current_tap + sign*1", "clip to ganged min/max"),11.5,9,10)
    draw_node(out,low,COL["green"],"Undervoltage candidate",("current_tap - sign*1", "clip to ganged min/max"),11.5,9,10)
    out.append(direct(dctrl,"left",high,"right","edge-gate"))
    out.append(label(
        dctrl.left+4,
        dctrl.cy-12,
        "> 1.02",
        COL["gate"],
        10,
        700,
        "end",
    ))

    out.append(direct(dctrl,"right",low,"left","edge-gate"))
    out.append(label(
        dctrl.right-4,
        dctrl.cy-12,
        "< 0.98",
        COL["gate"],
        10,
        700,
        "start",
    ))

    hold=Rg("hold",350,590,300,58); draw_node(out,hold,COL["neutral"],"In deadband: hold tap",("tap_attempted=False; reuse pre-PF",),11.5,9,10)
    out.append(direct(dctrl,"bottom",hold,"top","edge-gate"))
    out.append(label(dctrl.cx+12,dctrl.bottom+18,"0.98-1.02",COL["gate"],10,700,"start"))
    mtry=C(cx,686,mr); out.append(ortho_vh(high,"bottom",mtry,"left","edge-gate",bend_y=mtry.cy)); out.append(ortho_vh(low,"bottom",mtry,"right","edge-gate",bend_y=mtry.cy)); out.append(circle_svg(mtry,COL["neutral"]))
    dmove=Dg("dmove",cx,752,240,62); draw_decision(out,dmove,"candidate != current_tap?",ts=12); out.append(direct(mtry,"bottom",dmove,"top"))

    limit=Rg("limit",722,718,220,68); draw_node(out,limit,COL["neutral"],"Tap limit reached",("blocked_reason=tap_limit_reached", "reuse pre-PF; converged_post=True"),10.5,8.3,10); out.append(direct(dmove,"right",limit,"left","edge-gate")); out.append(label(dmove.right+8,dmove.cy-8,"no",COL["gate"],10,700,"start"))
    apply=Rg("apply",300,824,400,62); draw_node(out,apply,COL["blue"],"[5] apply candidate tap to ganged trafos",("save prev_tap; set tap_pos=candidate",),12.5,9); out.append(direct(dmove,"bottom",apply,"top")); out.append(label(dmove.cx+10,dmove.bottom+18,"yes",COL["text"],10,700,"start"))
    post=Rg("post",300,916,400,56); draw_node(out,post,COL["blue"],"[5a] post-tap PF",("try pp.runpp() at candidate tap",),12.5,9); out.append(direct(apply,"bottom",post,"top"))
    dpost=Dg("dpost",cx,1038,220,62); draw_decision(out,dpost,"post-tap PF converged?",ts=12); out.append(direct(post,"bottom",dpost,"top"))

    accept=Rg("accept",120,1082,255,68); rollback=Rg("rollback",625,1072,290,88)
    draw_node(out,accept,COL["green"],"Accept tap move",("current_tap=candidate", "tap_changed=True"),11,8.5,10)
    draw_node(out,rollback,COL["red"],"Rollback candidate tap",("restore prev_tap; blocked=post_pf_non_convergence", "runpp() again at previous tap"),10.5,8.2,10)
    out.append(ortho_hv(dpost,"left",accept,"top","edge-gate")); out.append(label(dpost.left-8,dpost.cy-8,"yes",COL["gate"],10,700,"end"))
    out.append(ortho_hv(dpost,"right",rollback,"top","edge-hil")); out.append(label(dpost.right+8,dpost.cy-8,"no",COL["hil"],10,700,"start"))

    mpost=C(cx,1206,mr); out.append(ortho_vh(accept,"bottom",mpost,"left","edge-gate",bend_y=mpost.cy)); out.append(ortho_vh(rollback,"bottom",mpost,"right","edge-hil",bend_y=mpost.cy)); out.append(circle_svg(mpost,COL["neutral"]))

    # Deadband and tap-limit branches both skip candidate/post-PF execution.
    out.append(path([hold.a("left"),(100,hold.cy),(100,1244),(cx-14,1244)],"edge-gate",marker=False))
    limit_rail = 955
    out.append(path([
        limit.a("right"),
        (limit_rail, limit.cy),
        (limit_rail, 1244),
        (cx+14, 1244),
    ], "edge-gate", marker=False))
    msettle=C(cx,1244,mr)
    out.append(direct(mpost,"bottom",msettle,"top"))
    out.append(path([(100,1244),msettle.a("left",TARGET_GAP)],"edge-gate"))
    out.append(path([(limit_rail,1244),msettle.a("right",TARGET_GAP)],"edge-gate"))
    out.append(circle_svg(msettle,COL["neutral"]))

    dconv=Dg("dconv",cx,1312,220,60); draw_decision(out,dconv,"settled PF converged?",ts=12); out.append(direct(msettle,"bottom",dconv,"top"))
    empty=Rg("empty",105,1356,245,72); extract=Rg("extract",610,1352,315,80)
    draw_node(out,empty,COL["red"],"Non-converged settled state",("empty V/loading; losses/import=None", "possible only if rollback PF also fails"),10.5,8.3,10)
    draw_node(out,extract,COL["blue"],"Extract settled network state",("V / line / trafo results; violations", "losses + grid import + DER/load totals"),11,8.5,10)
    out.append(ortho_hv(dconv,"left",empty,"top","edge-hil")); out.append(label(dconv.left-8,dconv.cy-8,"no",COL["hil"],10,700,"end"))
    out.append(ortho_hv(dconv,"right",extract,"top")); out.append(label(dconv.right+8,dconv.cy-8,"yes",COL["text"],10,700,"start"))
    mrec=C(cx,1470,mr); out.append(ortho_vh(empty,"bottom",mrec,"left","edge-hil",bend_y=mrec.cy)); out.append(ortho_vh(extract,"bottom",mrec,"right",bend_y=mrec.cy)); out.append(circle_svg(mrec,COL["neutral"]))

    record=Rg("record",260,1504,480,76); draw_node(out,record,COL["purple"],"[6] build settled TimestepRecord",("tap flags / blocked_reason; final network state", "publish_fn.on_timestep(rec) -> records.append(rec)"),11.5,8.8); out.append(direct(mrec,"bottom",record,"top"))
    periodic=Dg("periodic",cx,1634,200,54); draw_decision(out,periodic,"t % 96 == 0?",ts=12); out.append(direct(record,"bottom",periodic,"top"))
    progress=Rg("progress",720,1603,240,62); draw_node(out,progress,COL["purple"],"Periodic progress",("optional partial result -> live CSV", "log tap / vm_ctrl / violation counts"),10.2,8.2,10); out.append(direct(periodic,"right",progress,"left","edge-gate")); out.append(label(periodic.right+8,periodic.cy-8,"yes",COL["gate"],10,700,"start"))
    mnext=C(cx,1708,mr); out.append(direct(periodic,"bottom",mnext,"top")); out.append(label(periodic.cx+10,periodic.bottom+16,"no",COL["text"],10,700,"start")); out.append(ortho_vh(progress,"bottom",mnext,"right","edge-gate",bend_y=mnext.cy))

    # Pre-PF failure skips publisher + periodic work and enters iteration-complete merge only.
    out.append(path([prefail.a("left"),(failure_rail,prefail.cy)],"edge-hil",marker=False)); out.append(path([(failure_rail,prefail.cy),(failure_rail,mnext.cy)],"edge-hil",marker=False)); out.append(path([(failure_rail,mnext.cy),mnext.a("left",TARGET_GAP)],"edge-hil")); out.append(circle_svg(mnext,COL["neutral"]))

    more=Dg("more",cx,1782,180,48); draw_decision(out,more,"more timesteps?",ts=12); out.append(direct(mnext,"bottom",more,"top"))
    done=Rg("done",650,1760,230,44); draw_node(out,done,COL["purple"],"loop complete",("return to top-level finalization",),10.5,8.3,9); out.append(direct(more,"right",done,"left")); out.append(label(more.right+8,more.cy-7,"no",COL["text"],10,700,"start"))
    loopback_y=1840; out.append(path([more.a("bottom"),(more.cx,loopback_y),(loop_rail,loopback_y),(loop_rail,loop.cy),loop.a("left",TARGET_GAP)],"edge-loop")); out.append(label(more.cx+12,more.bottom+16,"yes - next t",COL["loop"],10,700,"start"))

    for n,g in rects.items(): audit_rect_bounds(n,g,W,H,8)
    for n,g in diamonds.items(): audit_diamond_bounds(n,g,W,H,8)
    assert abs(prefail.cy-dpre.cy)<1e-9
    assert state.top-loop.bottom>=15; assert pre.top-state.bottom>=25; assert dpre.top-pre.bottom>=25; assert read.top-dpre.bottom>=35
    assert dctrl.top-read.bottom>=25; assert mtry.top-max(high.bottom,low.bottom)>=25; assert dmove.top-mtry.bottom>=20
    assert apply.top-dmove.bottom>=40; assert post.top-apply.bottom>=25; assert dpost.top-post.bottom>=30
    assert mpost.top-max(accept.bottom,rollback.bottom)>=30; assert msettle.top-mpost.bottom>=14; assert dconv.top-msettle.bottom>=25
    assert mrec.top-max(empty.bottom,extract.bottom)>=25; assert record.top-mrec.bottom>=20; assert periodic.top-record.bottom>=25; assert mnext.top-periodic.bottom>=35
    assert loop_rail<failure_rail<prefail.left-10; assert done.bottom<loopback_y-20; assert loopback_y<H-30
    assert limit_rail > limit.right + 10

    write("flow_s2_loop_ieee_final",W,H,"Scenario 2 per-timestep OLTC flow - IEEE","\n".join(out)); return W,H


def build_presentation():
    W,H=1920,1080
    out=[rect_svg(R(0,0,W,H),COL["panel"],0)]; rects={}; diamonds={}
    def Rg(n,x,y,w,h): g=R(x,y,w,h); rects[n]=g; return g
    def Dg(n,x,y,w,h): g=D(x,y,w,h); diamonds[n]=g; return g
    cx=480; loop_rail=18; failure_rail=42; mr=12

    loop=Dg("loop",cx,20,180,32); out.append(diamond_svg(loop,COL["neutral"])); out.append(text_lines(cx,17,"for each t",("in time_steps",),14,9,12))
    state=Rg("state",170,42,620,38); draw_node(out,state,COL["blue"],"[1] write load / DER profiles; force DER Q=0",(),13.5); out.append(direct(loop,"bottom",state,"top"))
    pre=Rg("pre",170,96,620,38); draw_node(out,pre,COL["blue"],"[2] pre-action PF at current tap",("pp.runpp(voltage_depend_loads=False)",),13.2,8.8); out.append(direct(state,"bottom",pre,"top"))
    dpre=Dg("dpre",cx,170,240,42); draw_decision(out,dpre,"pre-PF converged?",ts=12.8); out.append(direct(pre,"bottom",dpre,"top"))
    prefail=Rg("prefail",62,143,220,54); draw_node(out,prefail,COL["red"],"Pre-PF failure",("append failed record; hold tap; no publish; continue",),10.8,8.0,10); out.append(direct(dpre,"left",prefail,"right","edge-hil")); out.append(label(dpre.left-10,dpre.cy-8,"no",COL["hil"],10.5,700,"end"))

    read=Rg("read",170,214,620,36); draw_node(out,read,COL["blue"],"[3] vm_ctrl = mean(control-bus vm_pu); min/max diagnostic only",(),13.1); out.append(direct(dpre,"bottom",read,"top")); out.append(label(dpre.cx+12,dpre.bottom+15,"yes",COL["text"],10.5,700,"start"))
    dctrl=Dg("dctrl",cx,290,300,48); draw_decision(out,dctrl,"vm_ctrl vs [0.98, 1.02]?",ts=12.7); out.append(direct(read,"bottom",dctrl,"top"))
    high=Rg("high",70,266,250,48); low=Rg("low",690,266,250,48); hold=Rg("hold",320,330,320,44)
    draw_node(out,high,COL["green"],"High V: candidate=current+sign",("clip to gang range",),10.8,8.1,10)
    draw_node(out,low,COL["green"],"Low V: candidate=current-sign",("clip to gang range",),10.8,8.1,10)
    draw_node(out,hold,COL["neutral"],"In band: hold + reuse pre-PF",("tap_attempted=False",),10.8,8.1,10)

    out.append(direct(dctrl,"left",high,"right","edge-gate"))
    out.append(label(
        dctrl.left+20,
        dctrl.cy-8,
        ">1.02",
        COL["gate"],
        10,
        700,
        "end",
    ))

    out.append(direct(dctrl,"right",low,"left","edge-gate"))
    out.append(label(
        dctrl.right+12,
        dctrl.cy-8,
        "<0.98",
        COL["gate"],
        10,
        700,
        "start",
    ))

    out.append(direct(dctrl,"bottom",hold,"top","edge-gate"))
    out.append(label(dctrl.cx+14,dctrl.bottom+15,"in band",COL["gate"],10,700,"start"))



    mtry=C(cx,402,mr); out.append(ortho_vh(high,"bottom",mtry,"left","edge-gate",bend_y=mtry.cy)); out.append(ortho_vh(low,"bottom",mtry,"right","edge-gate",bend_y=mtry.cy)); out.append(circle_svg(mtry,COL["neutral"]))
    dmove=Dg("dmove",cx,450,250,40); draw_decision(out,dmove,"candidate != current?",ts=12.0); out.append(direct(mtry,"bottom",dmove,"top"))
    limit=Rg("limit",725,430,250,40); draw_node(out,limit,COL["neutral"],"Tap limit reached",("reuse pre-PF; blocked_reason set",),10.2,7.9,10); out.append(direct(dmove,"right",limit,"left","edge-gate")); out.append(label(dmove.right+10,dmove.cy-7,"no",COL["gate"],10,700,"start"))
    apply=Rg("apply",240,492,480,38); draw_node(out,apply,COL["blue"],"[5] write candidate tap to ganged transformer group",("save prev_tap",),12.0,8.1); out.append(direct(dmove,"bottom",apply,"top")); out.append(label(dmove.cx+12,dmove.bottom+15,"yes",COL["text"],10,700,"start"))
    post=Rg("post",240,548,480,38); draw_node(out,post,COL["blue"],"[5a] post-tap PF",(),12.5); out.append(direct(apply,"bottom",post,"top"))
    dpost=Dg("dpost",cx,624,230,42); draw_decision(out,dpost,"post-PF converged?",ts=12.0); out.append(direct(post,"bottom",dpost,"top"))
    accept=Rg("accept",80,655,250,46); rollback=Rg("rollback",650,647,310,62)
    draw_node(out,accept,COL["green"],"Accept candidate",("current_tap=candidate; changed=True",),10.5,8.0,10)
    draw_node(out,rollback,COL["red"],"Rollback candidate",("restore prev_tap; blocked_reason set", "runpp() again at previous tap"),10.0,7.8,10)
    out.append(ortho_hv(dpost,"left",accept,"top","edge-gate")); out.append(label(dpost.left-10,dpost.cy-8,"yes",COL["gate"],10,700,"end"))
    out.append(ortho_hv(dpost,"right",rollback,"top","edge-hil")); out.append(label(dpost.right+10,dpost.cy-8,"no",COL["hil"],10,700,"start"))
    mpost=C(cx,738,mr); out.append(ortho_vh(accept,"bottom",mpost,"left","edge-gate",bend_y=mpost.cy)); out.append(ortho_vh(rollback,"bottom",mpost,"right","edge-hil",bend_y=mpost.cy)); out.append(circle_svg(mpost,COL["neutral"]))
    msettle=C(cx,772,mr); out.append(direct(mpost,"bottom",msettle,"top"))
    out.append(path([hold.a("left"),(104,hold.cy),(104,msettle.cy),msettle.a("left",TARGET_GAP)],"edge-gate"))
    out.append(path([limit.a("right"),(1000,limit.cy),(1000,msettle.cy),msettle.a("right",TARGET_GAP)],"edge-gate")); out.append(circle_svg(msettle,COL["neutral"]))

    dconv=Dg("dconv",cx,822,230,40); draw_decision(out,dconv,"settled PF converged?",ts=12.0); out.append(direct(msettle,"bottom",dconv,"top"))
    empty=Rg("empty",70,800,250,44); extract=Rg("extract",660,796,330,52)
    draw_node(out,empty,COL["red"],"Non-converged settled state",("rollback PF also failed",),10.2,7.9,10)
    draw_node(out,extract,COL["blue"],"Extract final state",("V/loading + violations + losses/import",),10.7,8.0,10)
    out.append(direct(dconv,"left",empty,"right","edge-hil")); out.append(label(dconv.left-10,dconv.cy-7,"no",COL["hil"],10,700,"end"))
    out.append(direct(dconv,"right",extract,"left")); out.append(label(dconv.right+10,dconv.cy-7,"yes",COL["text"],10,700,"start"))
    mrec=C(cx,872,mr); out.append(ortho_vh(empty,"bottom",mrec,"left","edge-hil",bend_y=mrec.cy)); out.append(ortho_vh(extract,"bottom",mrec,"right",bend_y=mrec.cy)); out.append(circle_svg(mrec,COL["neutral"]))
    record=Rg("record",225,892,510,38); draw_node(out,record,COL["purple"],"[6] build record -> publish -> append",(),11.8); out.append(direct(mrec,"bottom",record,"top"))

    periodic=Dg("periodic",cx,966,190,40); draw_decision(out,periodic,"t % 96 == 0?",ts=11.3); out.append(direct(record,"bottom",periodic,"top"))
    progress=Rg("progress",730,936,300,44); draw_node(out,progress,COL["purple"],"Periodic progress",("optional partial CSV + logger.info",),10.3,8.0,10)
    out.append(ortho_hv(periodic,"right",progress,"left","edge-gate")); out.append(label(periodic.right+8,periodic.cy-7,"yes",COL["gate"],10,700,"start"))
    mnext=C(cx,1026,mr); out.append(direct(periodic,"bottom",mnext,"top")); out.append(label(periodic.cx+10,periodic.bottom+15,"no",COL["text"],10,700,"start")); out.append(path([progress.a("bottom"),(progress.cx,mnext.cy),(mnext.right+2,mnext.cy),mnext.a("right",TARGET_GAP)],"edge-gate")); out.append(circle_svg(mnext,COL["neutral"]))

    # Pre-PF failure bypass to iteration-complete merge.
    out.append(path([prefail.a("left"),(failure_rail,prefail.cy)],"edge-hil",marker=False)); out.append(path([(failure_rail,prefail.cy),(failure_rail,mnext.cy)],"edge-hil",marker=False)); out.append(path([(failure_rail,mnext.cy),mnext.a("left",TARGET_GAP)],"edge-hil"))
    more=Dg("more",760,1050,200,34); draw_decision(out,more,"more timesteps?",ts=11.5); out.append(path([mnext.a("bottom"),(mnext.cx,more.cy),more.a("left",TARGET_GAP)],"edge-dark"))
    done=Rg("done",880,1033,200,34); draw_node(out,done,COL["purple"],"loop complete",(),11.0); out.append(direct(more,"right",done,"left")); out.append(label(more.right+8,more.cy-6,"no",COL["text"],10,700,"start"))
    loopback_y=1072; out.append(path([more.a("bottom"),(more.cx,loopback_y),(loop_rail,loopback_y),(loop_rail,loop.cy),loop.a("left",TARGET_GAP)],"edge-loop")); out.append(label(more.cx+12,more.bottom+14,"yes - next t",COL["loop"],10,700,"start"))

    # Right-side explanatory panels.
    px=1120; pw=760
    p1=R(px,24,pw,280); p2=R(px,326,pw,290); p3=R(px,638,pw,350)
    for p in (p1,p2,p3): out.append(rect_svg(p,COL["white"],16,COL["detail_border"],1.2))
    out.append(label(px+22,54,"OLTC decision semantics",COL["text"],17,700,"start"))
    lines=[
        "Control signal is the mean voltage across all selected secondary busbars.",
        "vm_ctrl > 1.02: candidate = current_tap + calibrated sign.",
        "vm_ctrl < 0.98: candidate = current_tap - calibrated sign.",
        "Only one tap step is requested per timestep; candidate is clipped to the ganged range.",
        "Inside the deadband no post-PF is run; valid pre-PF results are reused.",
        "If clipping returns current_tap, the controller records tap_limit_reached and also reuses pre-PF.",
    ]
    for i,t in enumerate(lines): out.append(label(px+22,88+32*i,t,COL["text"],12.0,600,"start"))

    out.append(label(px+22,358,"Post-tap + rollback semantics",COL["text"],17,700,"start"))
    lines=[
        "Candidate PF converges: accept the move and set tap_changed=True.",
        "Candidate PF fails: restore prev_tap and set blocked_reason=post_pf_non_convergence.",
        "A second runpp() is attempted after rollback to rebuild valid result tables.",
        "Rollback success is recorded as converged=True but post_pf_reused=False.",
        "Rollback failure leaves converged_post=False and empty result tables.",
        "No reactive-power control or active-power curtailment exists in Scenario 2.",
    ]
    for i,t in enumerate(lines): out.append(label(px+22,394+34*i,t,COL["text"],12.0,600,"start"))

    out.append(label(px+22,670,"Publisher / failure semantics + implementation note",COL["text"],17,700,"start"))
    lines=[
        "Normal path: build record -> on_timestep(rec) -> append -> optional t%96 progress.",
        "Pre-PF failure appends its failed record directly, then continue: no on_timestep() and no t%96 branch.",
        "The pre-PF failure rail and blue next-timestep rail are intentionally separate.",
        "Current source edge case: if rollback PF also fails, der_gen_mw_t/load_mw_t are not assigned in that branch.",
        "If this is the first such occurrence record construction can raise; otherwise stale prior-loop values can be reused.",
        "This note reflects the supplied implementation and is not silently corrected by the diagram.",
    ]
    for i,t in enumerate(lines): out.append(label(px+22,706+37*i,t,COL["text"],11.7,600,"start"))

    # Geometry checks on main execution region only. Panels intentionally occupy the right half.
    for n,g in rects.items(): audit_rect_bounds(n,g,W,H,4)
    for n,g in diamonds.items(): audit_diamond_bounds(n,g,W,H,4)
    assert abs(prefail.cy-dpre.cy)<1e-9
    assert loop_rail<failure_rail<prefail.left-10
    assert p1.left>1080

    write("flow_s2_loop_presentation_final",W,H,"Scenario 2 per-timestep OLTC flow - presentation","\n".join(out)); return W,H

if __name__=="__main__":
    iw,ih=build_ieee(); pw,ph=build_presentation()
    print(f"IEEE loop audited: {iw} x {ih}")
    print(f"Presentation loop audited: {pw} x {ph}")
    print(f"Outputs: {OUT}")
