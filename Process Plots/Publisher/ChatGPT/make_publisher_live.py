from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import html
from urllib import parse
import cairosvg

OUT = Path(__file__).resolve().parent / "publisher_flowcharts_live"
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


def draw_node(out,g,fill,title,subs=(),ts=14,ss=10,rx=12):
    out.append(rect_svg(g,fill,rx))

    line_gap = 15
    n = 1 + len(subs)

    if n == 1:
        base = g.cy + 5
    else:
        total = line_gap + (n - 2) * (line_gap - 2)
        base = g.cy - total / 2 + 5

    out.append(text_lines(
        g.cx, base, title, subs,
        ts, ss, line_gap
    ))


def draw_decision(out,g,title,subs=(),ts=13,ss=10):
    out.append(diamond_svg(g,COL["decision"]))

    line_gap = 14
    n = 1 + len(subs)

    if n == 1:
        base = g.cy + 4
    else:
        total = line_gap + (n - 2) * (line_gap - 2)
        base = g.cy - total / 2 + 4

    out.append(text_lines(
        g.cx, base, title, subs,
        ts, ss, line_gap
    ))


def audit_rect_bounds(name,g,W,H,m=0): assert g.left>=m and g.top>=m and g.right<=W-m and g.bottom<=H-m, f"{name}: bounds"
def audit_diamond_bounds(name,g,W,H,m=0): assert g.left>=m and g.right<=W-m and g.top>=m and g.bottom<=H-m, f"{name}: bounds"


def build_ieee():
    W,H=980,1840
    out=[rect_svg(R(14,14,W-28,H-28),COL["panel"],16,COL["panel_border"],1.5)]
    rects={}; diamonds={}
    def Rg(n,x,y,w,h): g=R(x,y,w,h); rects[n]=g; return g
    def Dg(n,x,y,w,h): g=D(x,y,w,h); diamonds[n]=g; return g
    cx=500; mr=12; resume_rail=48; loop_rail=28
    out.append(label(W/2,36,"Publisher - Live Streaming + Crash Resume",COL["text"],18,700))

    start=Rg("start",260,70,480,58); draw_node(out,start,COL["blue"],"runner: on_scenario_start(sid, label, T)",(),14)
    closeold=Dg("closeold",cx,176,240,56); draw_decision(out,closeold,"old checkpoint handle open?",ts=12)
    closebox=Rg("closebox",720,148,220,56); draw_node(out,closebox,COL["purple"],"Close previous handle",("_checkpoint_fh=None",),11,8.5,10)
    init=Rg("init",250,232,500,82); draw_node(out,init,COL["purple"],"Initialize scenario publish context",("store sid/label/T; set live path", "mkdir live/; truncate live/<sid>.jsonl; start attempt timer"),12.5,9)
    dcp=Dg("dcp",cx,372,240,58); draw_decision(out,dcp,"checkpointing enabled?",ts=12)
    cpon=Rg("cpon",620,420,310,84); draw_node(out,cpon,COL["purple"],"Open resume checkpoint",("checkpoint/<sid>.jsonl in append mode", "read .elapsed if present; preserve existing JSONL"),11,8.5,10)
    cpoff=Rg("cpoff",70,420,300,84); draw_node(out,cpoff,COL["neutral"],"Checkpointing disabled",("clear checkpoint/elapsed state", "live dashboard stream still available"),11,8.5,10)
    mstart=C(cx,548,mr); out.append(ortho_vh(cpon,"bottom",mstart,"right","edge-gate",bend_y=mstart.cy)); out.append(ortho_vh(cpoff,"bottom",mstart,"left","edge-gate",bend_y=mstart.cy)); out.append(circle_svg(mstart,COL["neutral"]))

    resume=Rg("resume",250,582,500,66); draw_node(out,resume,COL["blue"],"runner may call get_resume_records(sid)",("S2-S5 use this after on_scenario_start; S1 does not",),12,9)
    dpath=Dg("dpath",cx,710,270,64); draw_decision(out,dpath,"resume checkpoint available?",("live path first; else .completed archive",),11,8.5)
    parse=Rg("parse",680,681,280,58)
    draw_node(
        out,parse,COL["green"],
        "Reconstruct records",
        ("parse JSONL -> TimestepRecord",
        "skip corrupt/partial lines"),
        11,8.3,10
    )

    fresh=Rg("fresh",80,683,240,54)
    draw_node(
        out,fresh,COL["neutral"],
        "Fresh scenario",
        ("return []",),
        11,8.5,10
    )
    mresume=C(cx,800,mr); out.append(ortho_vh(parse,"bottom",mresume,"right","edge-gate",bend_y=mresume.cy)); out.append(ortho_vh(fresh,"bottom",mresume,"left","edge-gate",bend_y=mresume.cy)); out.append(circle_svg(mresume,COL["neutral"]))

    loop=Dg("loop",cx,856,170,46); out.append(diamond_svg(loop,COL["neutral"])); out.append(text_lines(cx,852,"each runner call",("on_timestep(rec)",),13,9,13))
    dwrite=Dg("dwrite",cx,936,230,54); draw_decision(out,dwrite,"checkpoint file open?",ts=12)
    ckwrite=Rg("ckwrite",680,905,250,62); draw_node(out,ckwrite,COL["purple"],"Write full checkpoint row",("rec.to_checkpoint_dict(); flush",),10.8,8.2,10)
    mwrite=C(cx,1008,mr); out.append(direct(dwrite,"bottom",mwrite,"top")); out.append(ortho_vh(ckwrite,"bottom",mwrite,"right","edge-gate",bend_y=mwrite.cy)); out.append(circle_svg(mwrite,COL["neutral"]))
    dk=Dg("dk",cx,1080,220,54); draw_decision(out,dk,"rec.t % update_every_k == 0?",ts=11.5)
    dpathlive=Dg("dpathlive",cx,1170,220,54); draw_decision(out,dpathlive,"live path initialized?",ts=11.5)
    warn=Rg("warn",90,1143,260,54)
    draw_node(
        out,warn,COL["red"],
        "Warn + return",
        ("on_scenario_start was not called",),
        10.8,8.2,10
    )

    live=Rg("live",640,1136,300,68)
    draw_node(
        out,live,COL["purple"],
        "Emit dashboard sample",
        ("write cumulative elapsed sidecar if enabled",
        "build_live_frame() -> append live/<sid>.jsonl"),
        11,8.2,10
    )
    miter=C(cx,1256,mr)

    warn_route_y = miter.top - 18

    out.append(path([
        warn.a("bottom"),
        (warn.cx, warn_route_y),
        (miter.cx, warn_route_y),
        miter.a("top", TARGET_GAP),
    ], "edge-hil"))

    out.append(ortho_vh(
        live,"bottom",
        miter,"right",
        "edge-gate",
        bend_y=miter.cy
    ))

    out.append(circle_svg(miter,COL["neutral"]))
    
    out.append(path([dk.a("left"),(resume_rail,dk.cy),(resume_rail,miter.cy),miter.a("left",TARGET_GAP)],"edge-gate")); out.append(label(dk.left-8,dk.cy-8,"no -> return",COL["gate"],10,700,"end"))

    dend=Dg("dend",cx,1334,190,50); draw_decision(out,dend,"scenario finished?",ts=12)
    out.append(direct(miter,"bottom",dend,"top"))
    out.append(path([dend.a("left"),(loop_rail,dend.cy),(loop_rail,loop.cy),loop.a("left",TARGET_GAP)],"edge-loop")); out.append(label(dend.left-8,dend.cy-8,"no",COL["loop"],10,700,"end"))
    end=Rg("end",250,1380,500,76); draw_node(out,end,COL["blue"],"runner: on_scenario_end(result)",("close checkpoint handle; persist cumulative elapsed", "append scenario_complete to live JSONL if live path exists"),12,8.8)
    out.append(direct(dend,"bottom",end,"top")); out.append(label(dend.cx+10,dend.bottom+16,"yes",COL["text"],10,700,"start"))
    static=Rg("static",250,1492,500,70); draw_node(out,static,COL["purple"],"benchmark_runner: publish_scenario_result()",("write final scenarios/<sid>.json",),12,9)
    archive=Rg("archive",250,1598,500,80); draw_node(out,archive,COL["green"],"Archive completed checkpoint",("checkpoint/<sid>.jsonl -> .jsonl.completed", "keeps audit/resume fallback while marking scenario complete"),11.5,8.5)
    done=Rg("done",170,1712,660,44)
    draw_node(
        out,
        done,
        COL["neutral"],
        "Scenario publishing complete",
        ("next scenario may reuse the same PublishHandle",),
        12,
        9,
    )
    
    out.append(direct(end,"bottom",static,"top")); out.append(direct(static,"bottom",archive,"top")); out.append(direct(archive,"bottom",done,"top"))

    # Start branches.
    out.append(path([start.a("bottom"),closeold.a("top",TARGET_GAP)],"edge-dark"))
    out.append(path([closeold.a("right"),closebox.a("left",TARGET_GAP)],"edge-gate")); out.append(label(closeold.right+8,closeold.cy-8,"yes",COL["gate"],10,700,"start"))
    out.append(path([closebox.a("bottom"),(closebox.cx,init.cy),init.a("right",TARGET_GAP)],"edge-gate"))
    out.append(direct(closeold,"bottom",init,"top")); out.append(label(closeold.cx+10,closeold.bottom+16,"no",COL["text"],10,700,"start"))
    out.append(direct(init,"bottom",dcp,"top"))
    out.append(path([dcp.a("right"),(775,dcp.cy),cpon.a("top",TARGET_GAP)],"edge-gate")); out.append(label(dcp.right+8,dcp.cy-8,"yes",COL["gate"],10,700,"start"))
    out.append(path([dcp.a("left"),(220,dcp.cy),cpoff.a("top",TARGET_GAP)],"edge-gate")); out.append(label(dcp.left-8,dcp.cy-8,"no",COL["gate"],10,700,"end"))
    out.append(direct(mstart,"bottom",resume,"top")); out.append(direct(resume,"bottom",dpath,"top"))
    out.append(direct(
        dpath,"right",
        parse,"left",
        "edge-gate"
    ))
    out.append(label(
        dpath.right+8,dpath.cy-8,
        "yes",COL["gate"],10,700,"start"
    ))

    out.append(direct(
        dpath,"left",
        fresh,"right",
        "edge-gate"
    ))
    out.append(label(
        dpath.left-8,dpath.cy-8,
        "no",COL["gate"],10,700,"end"
    ))
    out.append(direct(mresume,"bottom",loop,"top")); out.append(direct(loop,"bottom",dwrite,"top"))
    out.append(direct(dwrite,"right",ckwrite,"left","edge-gate")); out.append(label(dwrite.right+8,dwrite.cy-8,"yes",COL["gate"],10,700,"start"))
    out.append(direct(dwrite,"bottom",mwrite,"top")); out.append(label(dwrite.cx+10,dwrite.bottom+16,"no",COL["text"],10,700,"start"))
    out.append(direct(mwrite,"bottom",dk,"top")); out.append(direct(dk,"bottom",dpathlive,"top")); out.append(label(dk.cx+10,dk.bottom+16,"yes",COL["text"],10,700,"start"))
    out.append(direct(
        dpathlive,"left",
        warn,"right",
        "edge-hil"
    ))
    out.append(label(
        dpathlive.left-8,dpathlive.cy-8,
        "no",COL["hil"],10,700,"end"
    ))

    out.append(direct(
        dpathlive,"right",
        live,"left",
        "edge-gate"
    ))
    out.append(label(
        dpathlive.right+8,dpathlive.cy-8,
        "yes",COL["gate"],10,700,"start"
    ))
    
    for n,g in rects.items(): audit_rect_bounds(n,g,W,H,8)
    for n,g in diamonds.items(): audit_diamond_bounds(n,g,W,H,8)
    assert closeold.top-start.bottom>=20
    assert init.top-closeold.bottom>=28
    assert dcp.top-init.bottom>=28
    assert mstart.top-max(cpon.bottom,cpoff.bottom)>=30
    assert resume.top-mstart.bottom>=22
    assert dpath.top-resume.bottom>=30
    assert mresume.top-max(parse.bottom,fresh.bottom)>=28
    assert loop.top-mresume.bottom>=20
    assert dwrite.top-loop.bottom>=30
    assert mwrite.top-dwrite.bottom>=30
    assert dk.top-mwrite.bottom>=30
    assert dpathlive.top-dk.bottom>=30
    assert miter.top-max(warn.bottom,live.bottom)>=30
    assert dend.top-miter.bottom>=30
    assert end.top-dend.bottom>=21
    assert static.top-end.bottom>=30
    assert archive.top-static.bottom>=30
    assert done.top-archive.bottom>=30
    assert loop_rail < resume_rail < fresh.left-10
    assert abs(fresh.cy-dpath.cy) < 1e-9
    assert abs(parse.cy-dpath.cy) < 1e-9
    assert abs(warn.cy-dpathlive.cy) < 1e-9
    assert abs(live.cy-dpathlive.cy) < 1e-9
    assert warn.bottom < warn_route_y
    assert warn_route_y < miter.top
    assert done.bottom < H-30

    write("flow_publisher_live_ieee_final",W,H,"Publisher live/checkpoint lifecycle - IEEE","\n".join(out))
    return W,H


def build_presentation():
    W,H=1920,1080
    out=[rect_svg(R(0,0,W,H),COL["panel"],0)]; rects={}; diamonds={}
    def Rg(n,x,y,w,h): g=R(x,y,w,h); rects[n]=g; return g
    def Dg(n,x,y,w,h): g=D(x,y,w,h); diamonds[n]=g; return g
    cx=500; mr=12; loop_rail=18; bypass_rail=46

    start=Rg("start",170,18,660,36)
    draw_node(out,start,COL["blue"],
            "on_scenario_start(sid, label, T)",(),14)

    closeold=Dg("closeold",cx,86,240,36)
    draw_decision(out,closeold,"old checkpoint handle open?",ts=11.5)

    closebox=Rg("closebox",720,66,250,40)
    draw_node(out,closebox,COL["purple"],
            "Close previous handle",
            ("_checkpoint_fh=None",),10.5,7.8,10)

    init=Rg("init",170,118,660,42)
    draw_node(out,init,COL["purple"],
            "Initialize scenario publish context",
            ("truncate live stream; store sid/label/T; start timer",),
            11.5,8.2)

    dcp=Dg("dcp",cx,202,250,40)
    draw_decision(out,dcp,"checkpointing enabled?",ts=12)

    cpoff=Rg("cpoff",70,178,270,48)
    cpon=Rg("cpon",660,176,320,52)

    draw_node(out,cpoff,COL["neutral"],
            "No checkpoint stream",
            ("clear checkpoint/elapsed state",),10.5,7.9,10)

    draw_node(out,cpon,COL["purple"],
            "Open checkpoint append stream",
            ("read .elapsed; preserve existing JSONL",),10.5,7.9,10)

    mstart=C(cx,256,mr)

    resume=Rg("resume",180,280,640,40)
    draw_node(out,resume,COL["blue"],
            "runner: get_resume_records(sid) where implemented",
            (),11.5)

    dpath=Dg("dpath",cx,366,260,44)
    draw_decision(out,dpath,
                "resume checkpoint available?",ts=11.5)

    fresh=Rg("fresh",80,342,240,48)
    parse=Rg("parse",680,338,300,56)

    draw_node(out,fresh,COL["neutral"],
            "Fresh scenario",
            ("return []",),10.5,7.9,10)

    draw_node(out,parse,COL["green"],
            "Reconstruct records",
            ("JSONL -> TimestepRecord",
            "skip corrupt/partial lines"),
            10.5,7.8,10)

    mresume=C(cx,422,mr)

    loop=Dg("loop",cx,462,190,34)
    out.append(diamond_svg(loop,COL["neutral"]))
    out.append(text_lines(
        loop.cx,loop.cy-2,
        "on_timestep(rec)",
        ("runner callback",),
        12.5,8.5,12
    ))

    dwrite=Dg("dwrite",cx,516,230,36)
    draw_decision(out,dwrite,"checkpoint file open?",ts=11.5)

    ck=Rg("ck",735,497,270,38)
    draw_node(out,ck,COL["purple"],
            "Checkpoint every callback",
            ("write full record + flush",),10.2,7.6,10)

    mwrite=C(cx,564,mr)

    dk=Dg("dk",cx,618,230,36)
    draw_decision(out,dk,
                "t % update_every_k == 0?",ts=11.2)

    dpathlive=Dg("dpathlive",cx,674,220,36)
    draw_decision(out,dpathlive,
                "live path initialized?",ts=11.2)

    warn=Rg("warn",90,651,260,46)
    live=Rg("live",680,647,300,54)

    draw_node(out,warn,COL["red"],
            "Warn + return",
            ("on_scenario_start not called",),
            10.2,7.7,10)

    draw_node(out,live,COL["purple"],
            "Dashboard sample",
            ("persist elapsed; build frame; append JSONL",),
            10.2,7.7,10)

    miter=C(cx,744,mr)

    out.append(direct(start,"bottom",closeold,"top"))

    out.append(direct(
        closeold,"right",
        closebox,"left",
        "edge-gate"
    ))
    out.append(label(
        closeold.right+8,closeold.cy-7,
        "yes",COL["gate"],10,700,"start"
    ))

    out.append(path([
        closebox.a("bottom"),
        (closebox.cx,init.cy),
        init.a("right",TARGET_GAP),
    ],"edge-gate"))

    out.append(direct(closeold,"bottom",init,"top"))
    out.append(label(
        closeold.cx+10,closeold.bottom+14,
        "no",COL["text"],10,700,"start"
    ))

    out.append(direct(init,"bottom",dcp,"top"))

    out.append(direct(dcp,"left",cpoff,"right","edge-gate"))
    out.append(label(
        dcp.left-8,dcp.cy-7,
        "no",COL["gate"],10,700,"end"
    ))

    out.append(direct(dcp,"right",cpon,"left","edge-gate"))
    out.append(label(
        dcp.right+8,dcp.cy-7,
        "yes",COL["gate"],10,700,"start"
    ))

    out.append(ortho_vh(
        cpoff,"bottom",
        mstart,"left",
        "edge-gate",
        bend_y=mstart.cy
    ))
    out.append(ortho_vh(
        cpon,"bottom",
        mstart,"right",
        "edge-gate",
        bend_y=mstart.cy
    ))
    out.append(circle_svg(mstart,COL["neutral"]))

    out.append(direct(mstart,"bottom",resume,"top"))
    out.append(direct(resume,"bottom",dpath,"top"))

    out.append(direct(dpath,"left",fresh,"right","edge-gate"))
    out.append(label(
        dpath.left-8,dpath.cy-7,
        "no",COL["gate"],10,700,"end"
    ))

    out.append(direct(dpath,"right",parse,"left","edge-gate"))
    out.append(label(
        dpath.right+8,dpath.cy-7,
        "yes",COL["gate"],10,700,"start"
    ))

    out.append(ortho_vh(
        fresh,"bottom",
        mresume,"left",
        "edge-gate",
        bend_y=mresume.cy
    ))
    out.append(ortho_vh(
        parse,"bottom",
        mresume,"right",
        "edge-gate",
        bend_y=mresume.cy
    ))
    out.append(circle_svg(mresume,COL["neutral"]))

    out.append(direct(mresume,"bottom",loop,"top"))
    out.append(direct(loop,"bottom",dwrite,"top"))

    out.append(direct(dwrite,"right",ck,"left","edge-gate"))
    out.append(label(
        dwrite.right+8,dwrite.cy-7,
        "yes",COL["gate"],10,700,"start"
    ))

    out.append(direct(dwrite,"bottom",mwrite,"top"))
    out.append(label(
        dwrite.cx+10,dwrite.bottom+14,
        "no",COL["text"],10,700,"start"
    ))

    out.append(ortho_vh(
        ck,"bottom",
        mwrite,"right",
        "edge-gate",
        bend_y=mwrite.cy
    ))
    out.append(circle_svg(mwrite,COL["neutral"]))

    out.append(direct(mwrite,"bottom",dk,"top"))

    out.append(direct(dk,"bottom",dpathlive,"top"))
    out.append(label(
        dk.cx+10,dk.bottom+14,
        "yes",COL["text"],10,700,"start"
    ))

    out.append(direct(
        dpathlive,"left",
        warn,"right",
        "edge-hil"
    ))
    out.append(label(
        dpathlive.left-8,dpathlive.cy-7,
        "no",COL["hil"],10,700,"end"
    ))

    out.append(direct(
        dpathlive,"right",
        live,"left",
        "edge-gate"
    ))
    out.append(label(
        dpathlive.right+8,dpathlive.cy-7,
        "yes",COL["gate"],10,700,"start"
    ))

    # update_every_k == no bypass
    out.append(path([
        dk.a("left"),
        (bypass_rail,dk.cy),
        (bypass_rail,miter.cy),
        miter.a("left",TARGET_GAP),
    ],"edge-gate"))
    out.append(label(
        dk.left-8,dk.cy-7,
        "no -> return",COL["gate"],10,700,"end"
    ))

    # warn branch enters merge from above, so it cannot overlap dk-no
    warn_route_y = miter.top - 15

    out.append(path([
        warn.a("bottom"),
        (warn.cx,warn_route_y),
        (miter.cx,warn_route_y),
        miter.a("top",TARGET_GAP),
    ],"edge-hil"))

    out.append(ortho_vh(
        live,"bottom",
        miter,"right",
        "edge-gate",
        bend_y=miter.cy
    ))

    out.append(circle_svg(miter,COL["neutral"]))
  

    dend=Dg("dend",cx,794,190,34); draw_decision(out,dend,"scenario finished?",ts=11.8); out.append(direct(miter,"bottom",dend,"top"))
    out.append(path([dend.a("left"),(loop_rail,dend.cy),(loop_rail,loop.cy),loop.a("left",TARGET_GAP)],"edge-loop")); out.append(label(dend.left-8,dend.cy-7,"no",COL["loop"],10,700,"end"))
    end=Rg("end",170,824,660,44); draw_node(out,end,COL["blue"],"on_scenario_end(result)",("close checkpoint; persist elapsed; append scenario_complete live event",),11.5,8.2); out.append(direct(dend,"bottom",end,"top")); out.append(label(dend.cx+10,dend.bottom+14,"yes",COL["text"],10,700,"start"))
    static=Rg("static",170,894,660,40); draw_node(out,static,COL["purple"],"benchmark_runner writes scenarios/<sid>.json",("publish_scenario_result(result)",),11.5,8.2); out.append(direct(end,"bottom",static,"top"))
    archive=Rg("archive",170,960,660,44); draw_node(out,archive,COL["green"],"Archive checkpoint as .jsonl.completed",("final scenario JSON is confirmed before rename",),11.5,8.2); out.append(direct(static,"bottom",archive,"top"))

    # Right-side explanatory panels.
    px=1100; pw=780
    p1=R(px,24,pw,288); p2=R(px,334,pw,314); p3=R(px,670,pw,358)
    for p in (p1,p2,p3): out.append(rect_svg(p,COL["white"],16,COL["detail_border"],1.2))
    out.append(label(px+22,56,"Two independent JSONL streams",COL["text"],17,700,"start"))
    lines=[
        "live/<sid>.jsonl: dashboard-only compact frames; always truncated at on_scenario_start().",
        "checkpoint/<sid>.jsonl: full TimestepRecord serialization; opened in append mode and never truncated by PublishHandle.",
        "update_every_k affects only live dashboard frames. Checkpoint writes occur on every on_timestep() callback.",
        ".elapsed stores cumulative wall-clock time across crash/restart attempts.",
        "cumulative_elapsed_s() = prior persisted elapsed + current attempt runtime.",
    ]
    for i,t in enumerate(lines): out.append(label(px+22,94+39*i,t,COL["text"],12.0,600,"start"))

    out.append(label(px+22,368,"Resume and completion semantics",COL["text"],17,700,"start"))
    lines=[
        "get_resume_records() first checks checkpoint/<sid>.jsonl, then .jsonl.completed if the live path is absent.",
        "Valid JSONL rows are reconstructed with TimestepRecord.from_checkpoint_dict().",
        "Corrupt or partial lines are skipped, which protects against a crash during the final append.",
        "After a successful runner return, benchmark_runner writes the final scenario JSON and then renames the checkpoint to .completed.",
        "Layer 1 resume in benchmark_runner can skip the runner entirely when scenarios/<sid>.json already exists and loads correctly.",
    ]
    for i,t in enumerate(lines): out.append(label(px+22,406+44*i,t,COL["text"],11.8,600,"start"))

    out.append(label(px+22,704,"Runner-integration caveats",COL["text"],17,700,"start"))
    lines=[
        "PublishHandle can only checkpoint records that a scenario runner actually passes to on_timestep().",
        "S2/S3 pre-PF failure branches append failed records directly and therefore skip publisher checkpoint/live callbacks for those steps.",
        "S1 accepts the publisher handle but does not call get_resume_records(), so crash-resume is not implemented for its run_timeseries sweep.",
        "S2-S5 call on_scenario_start() before get_resume_records(); partial resumes continue from the last recovered record.",
        "on_scenario_end() closes the open checkpoint handle before appending the terminal live event.",
    ]
    for i,t in enumerate(lines): out.append(label(px+22,742+49*i,t,COL["text"],11.8,600,"start"))

    for n,g in rects.items(): audit_rect_bounds(n,g,W,H,4)
    for n,g in diamonds.items(): audit_diamond_bounds(n,g,W,H,4)
    assert p1.left>1060
    assert archive.bottom<H-30
    assert loop_rail < bypass_rail < 60

    write("flow_publisher_live_presentation_final",W,H,"Publisher live/checkpoint lifecycle - presentation","\n".join(out))
    return W,H


if __name__=="__main__":
    iw,ih=build_ieee(); pw,ph=build_presentation()
    print(f"Publisher live IEEE audited: {iw} x {ih}")
    print(f"Publisher live presentation audited: {pw} x {ph}")
    print(f"Outputs: {OUT}")
