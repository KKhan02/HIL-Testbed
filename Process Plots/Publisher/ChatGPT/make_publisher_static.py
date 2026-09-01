from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import html
import cairosvg

OUT = Path(__file__).resolve().parent / "publisher_flowcharts_static"
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


def direct(src,ss,dst,ds,cls="edge-dark",gap=TARGET_GAP):
    return path([src.a(ss),dst.a(ds,gap)],cls)


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
    out.append(rect_svg(g,fill,rx))
    n = 1 + len(subs)
    line_gap = 19
    total = 0 if n == 1 else line_gap + (n - 2) * (line_gap - 2)
    base = g.cy - total / 2 + 6
    out.append(text_lines(g.cx,base,title,subs,ts,ss,line_gap))


def draw_decision(out,g,title,subs=(),ts=15,ss=11):
    out.append(diamond_svg(g,COL["decision"]))
    n = 1 + len(subs)
    line_gap = 15
    total = 0 if n == 1 else line_gap + (n - 2) * (line_gap - 2)
    base = g.cy - total / 2 + 5
    out.append(text_lines(g.cx,base,title,subs,ts,ss,line_gap))


def audit_rect_bounds(name,g,W,H,m=0):
    assert g.left>=m and g.top>=m and g.right<=W-m and g.bottom<=H-m, f"{name}: bounds"


def audit_diamond_bounds(name,g,W,H,m=0):
    assert g.left>=m and g.right<=W-m and g.top>=m and g.bottom<=H-m, f"{name}: bounds"


def build_ieee():
    W,H=900,1340
    out=[rect_svg(R(22,22,W-44,H-44),COL["panel"],18,COL["panel_border"],1.8)]
    out.append(label(W/2,54,"Publisher - Static JSON Export Lifecycle",COL["text"],21,700))
    cx=380; bw=500; x=cx-bw/2

    ready=R(x,82,bw,62)
    syswrite=R(x,180,bw,82)
    sysout=R(665,178,190,86)
    scen=R(x,310,bw,66)
    scenwrite=R(x,412,bw,78)
    dmore=D(cx,550,230,72)
    scenout=R(665,415,190,72)
    finish=R(x,624,bw,66)
    finalwrite=R(x,730,bw,80)
    dhc=D(cx,874,210,66)
    hcout=R(30,842,190,64)
    dcmp=D(cx,970,230,66)
    cmpout=R(665,938,190,64)
    ret=R(x,1058,bw,82)
    files=R(x,1180,bw,100)

    draw_node(out,ready,COL["blue"],"Network + profiles ready",("run_benchmark_script before scenario execution",),16,11)
    draw_node(out,syswrite,COL["purple"],"publish_topology_and_profiles()",("build_topology(net) -> topology.json", "build_profiles_payload() -> profiles.json"),15,10.5)
    draw_node(out,sysout,COL["green"],"System files",("topology.json", "profiles.json"),13,10,10)
    draw_node(out,scen,COL["blue"],"Each scenario finishes",("benchmark_runner receives ScenarioResult",),15,10.5)
    draw_node(out,scenwrite,COL["purple"],"publish_scenario_result()",("one scenario at a time; compact JSON",),15,10.5)
    draw_decision(out,dmore,"more scenarios?",ts=13)
    draw_node(out,scenout,COL["green"],"Scenario file",("scenarios/<sid>.json",),12.5,9.5,10)
    draw_node(out,finish,COL["blue"],"Scenario loop + HC complete",("post-run summary stage",),15,10.5)
    draw_node(out,finalwrite,COL["purple"],"publish_hc_and_comparison()",("writes only payloads that are available",),15,10.5)
    draw_decision(out,dhc,"hc_results present?",ts=12.5)
    draw_node(out,hcout,COL["green"],"HC file",("hc.json",),12.5,9.5,10)
    draw_decision(out,dcmp,"comparison_df nonempty?",ts=12.5)
    draw_node(out,cmpout,COL["green"],"Comparison file",("comparison.json",),12.5,9.5,10)
    draw_node(out,ret,COL["purple"],"Return written paths",("system/final functions return {logical name: Path}", "scenario function returns Path or None"),14,10)
    draw_node(out,files,COL["neutral"],"Static output set",("topology.json | profiles.json | scenarios/<sid>.json", "hc.json if available | comparison.json if available"),14,10.5)

    out.append(direct(ready,"bottom",syswrite,"top"))
    out.append(direct(syswrite,"right",sysout,"left","edge-assoc"))
    out.append(direct(syswrite,"bottom",scen,"top"))
    out.append(direct(scen,"bottom",scenwrite,"top"))
    out.append(direct(scenwrite,"bottom",dmore,"top"))
    out.append(direct(scenwrite,"right",scenout,"left","edge-assoc"))
    scenario_rail = 84
    out.append(path([dmore.a("left"),(scenario_rail,dmore.cy),(scenario_rail,scen.cy),scen.a("left",TARGET_GAP)],"edge-loop"))
    out.append(label(dmore.left-8,dmore.cy-8,"yes",COL["loop"],10,700,"end"))
    out.append(direct(dmore,"bottom",finish,"top"))
    out.append(label(dmore.cx+10,dmore.bottom+16,"no",COL["text"],10,700,"start"))
    out.append(direct(finish,"bottom",finalwrite,"top"))
    out.append(direct(finalwrite,"bottom",dhc,"top"))
    out.append(direct(
        dhc,
        "left",
        hcout,
        "right",
        "edge-gate",
    ))
    out.append(label(
        dhc.left-8,
        dhc.cy-8,
        "yes",
        COL["gate"],
        10,
        700,
        "end",
    ))
    out.append(direct(dhc,"bottom",dcmp,"top"))
    out.append(label(dhc.cx+10,dhc.bottom+16,"no",COL["text"],10,700,"start"))
    out.append(direct(
        dcmp,
        "right",
        cmpout,
        "left",
        "edge-gate",
    ))
    out.append(label(
        dcmp.right+8,
        dcmp.cy-8,
        "yes",
        COL["gate"],
        10,
        700,
        "start",
    ))
    out.append(path([dcmp.a("bottom"),(dcmp.cx,ret.top-22),ret.a("top",TARGET_GAP)],"edge-dark"))
    out.append(label(dcmp.cx+10,dcmp.bottom+16,"no",COL["text"],10,700,"start"))

    hc_exit_x = hcout.cx - 20

    out.append(path([
        (hc_exit_x, hcout.bottom),
        (hc_exit_x, ret.cy),
        ret.a("left", TARGET_GAP),
    ], "edge-assoc"))

    out.append(path([
        cmpout.a("bottom"),
        (cmpout.cx, ret.cy),
        ret.a("right", TARGET_GAP),
    ], "edge-assoc"))
    out.append(direct(ret,"bottom",files,"top"))

    rects={"ready":ready,"syswrite":syswrite,"sysout":sysout,"scen":scen,"scenwrite":scenwrite,"scenout":scenout,"finish":finish,"finalwrite":finalwrite,"hcout":hcout,"cmpout":cmpout,"ret":ret,"files":files}
    diamonds={"dmore":dmore,"dhc":dhc,"dcmp":dcmp}
    for n,g in rects.items(): audit_rect_bounds(n,g,W,H,16)
    for n,g in diamonds.items(): audit_diamond_bounds(n,g,W,H,16)
    assert syswrite.top-ready.bottom>=30
    assert scen.top-syswrite.bottom>=40
    assert scenwrite.top-scen.bottom>=30
    assert dmore.top-scenwrite.bottom>=24
    assert finish.top-dmore.bottom>=38
    assert finalwrite.top-finish.bottom>=30
    assert dhc.top-finalwrite.bottom>=30
    assert dcmp.top-dhc.bottom>=30
    assert ret.top-dcmp.bottom>=55
    assert files.top-ret.bottom>=30
    assert hcout.right < dhc.left - 20
    assert cmpout.left > dcmp.right + 20
    assert abs(hcout.cy - dhc.cy) < 1e-9
    assert abs(cmpout.cy - dcmp.cy) < 1e-9
    assert hc_exit_x < ret.left - 2
    assert hcout.left < hc_exit_x < hcout.right

    write("flow_publisher_static_ieee_final",W,H,"Publisher static export lifecycle - IEEE","\n".join(out))
    return W,H


def build_presentation():
    W,H=1920,1080
    out=[rect_svg(R(0,0,W,H),COL["panel"],0)]
    main=R(24,22,1080,1036); out.append(rect_svg(main,"#F4F8FB",22,COL["panel_border"],1.4))
    cx=465; bw=700; x=cx-bw/2

    ready=R(x,44,bw,54)
    syswrite=R(x,128,bw,70)
    sysout=R(875,124,190,78)
    scen=R(x,238,bw,54)
    scenwrite=R(x,322,bw,64)
    dmore=D(cx,438,250,58)
    scenout=R(850,323,215,62)
    finish=R(x,508,bw,54)
    finalwrite=R(x,594,bw,64)
    dhc=D(cx,716,220,54)
    hcout=R(40,687,215,58)
    dcmp=D(cx,798,245,54)
    cmpout=R(850,769,215,58)
    ret=R(x,866,bw,60)
    files=R(x,958,bw,72)

    draw_node(out,ready,COL["blue"],"Network + profiles ready",("network-load time",),21,14)
    draw_node(out,syswrite,COL["purple"],"publish_topology_and_profiles()",("topology.json + profiles.json",),20,14)
    draw_node(out,sysout,COL["green"],"System payloads",("build_topology", "build_profiles_payload"),15,10,10)
    draw_node(out,scen,COL["blue"],"Each scenario completes",("inside benchmark_runner scenario loop",),20,13.5)
    draw_node(out,scenwrite,COL["purple"],"publish_scenario_result()",("build_scenario_payload -> scenarios/<sid>.json",),19,13)
    draw_decision(out,dmore,"more scenarios?",ts=17)
    draw_node(out,scenout,COL["green"],"Per-scenario JSON",("compact full timeseries",),14.5,10,10)
    draw_node(out,finish,COL["blue"],"All scenarios + HC complete",("final summary write point",),20,13.5)
    draw_node(out,finalwrite,COL["purple"],"publish_hc_and_comparison()",("conditional final-summary files",),19,13)
    draw_decision(out,dhc,"HC available?",ts=16)
    draw_node(out,hcout,COL["green"],"hc.json",("build_hc_payload",),14.5,10,10)
    draw_decision(out,dcmp,"comparison available?",ts=16)
    draw_node(out,cmpout,COL["green"],"comparison.json",("comparison_df -> records",),14.5,10,10)
    draw_node(out,ret,COL["purple"],"Return written paths",("independent writes; no monolithic publish_result()",),18,12.5)
    draw_node(out,files,COL["neutral"],"Static publisher output",("system files + one JSON per completed scenario + final summaries",),17,12)

    out.append(direct(ready,"bottom",syswrite,"top")); out.append(direct(syswrite,"right",sysout,"left","edge-assoc")); out.append(direct(syswrite,"bottom",scen,"top"))
    out.append(direct(scen,"bottom",scenwrite,"top")); out.append(direct(scenwrite,"bottom",dmore,"top"))
    out.append(direct(scenwrite,"right",scenout,"left","edge-assoc"))
    scenario_rail = 82
    out.append(path([dmore.a("left"),(scenario_rail,dmore.cy),(scenario_rail,scen.cy),scen.a("left",TARGET_GAP)],"edge-loop")); out.append(label(dmore.left-12,dmore.cy-8,"yes",COL["loop"],12,700,"end"))
    out.append(direct(dmore,"bottom",finish,"top")); out.append(label(dmore.cx+12,dmore.bottom+16,"no",COL["text"],12,700,"start"))
    out.append(direct(finish,"bottom",finalwrite,"top")); out.append(direct(finalwrite,"bottom",dhc,"top"))
    out.append(direct(
        dhc,
        "left",
        hcout,
        "right",
        "edge-gate",
    ))
    out.append(label(
        dhc.left-10,
        dhc.cy-7,
        "yes",
        COL["gate"],
        12,
        700,
        "end",
    ))
    out.append(direct(dhc,"bottom",dcmp,"top")); out.append(label(dhc.cx+12,dhc.bottom+16,"no",COL["text"],12,700,"start"))
    out.append(direct(
        dcmp,
        "right",
        cmpout,
        "left",
        "edge-gate",
    ))
    out.append(label(
        dcmp.right+10,
        dcmp.cy-7,
        "yes",
        COL["gate"],
        12,
        700,
        "start",
    ))
    out.append(direct(dcmp,"bottom",ret,"top")); out.append(label(dcmp.cx+12,dcmp.bottom+16,"no",COL["text"],12,700,"start"))

    hc_exit_x = hcout.cx - 50

    out.append(path([
        (hc_exit_x, hcout.bottom),
        (hc_exit_x, ret.cy),
        ret.a("left", TARGET_GAP),
    ], "edge-assoc"))

    out.append(path([
        cmpout.a("bottom"),
        (cmpout.cx, ret.cy),
        ret.a("right", TARGET_GAP),
    ], "edge-assoc"))
    out.append(direct(ret,"bottom",files,"top"))

    # Right-side explanatory panels.
    px=1140; pw=740
    p1=R(px,26,pw,300); p2=R(px,346,pw,310); p3=R(px,676,pw,360)
    for p in (p1,p2,p3): out.append(rect_svg(p,COL["white"],16,COL["detail_border"],1.2))
    out.append(label(px+22,58,"Three independent static write points",COL["text"],17,700,"start"))
    lines=[
        "1. Network-load time: topology.json and profiles.json are written before scenario execution.",
        "2. Scenario-loop time: each successful ScenarioResult is written immediately to scenarios/<sid>.json.",
        "3. Post-run time: HC and comparison payloads are written after the scenario loop and HC analysis.",
        "The old monolithic publish_result() flow is no longer present in the current publisher.py.",
        "Splitting the writes improves crash resilience and avoids one large end-of-run write burst.",
    ]
    for i,t in enumerate(lines): out.append(label(px+22,96+39*i,t,COL["text"],12.2,600,"start"))

    out.append(label(px+22,380,"Payload-builder responsibilities",COL["text"],17,700,"start"))
    lines=[
        "build_topology(): buses, lines, trafos, sgens, loads, feeder distance and voltage limits.",
        "build_profiles_payload(): full-resolution and hourly load/PV/wind totals plus per-element series.",
        "build_scenario_payload(): summary plus per-timestep network, violation, control and energy fields.",
        "build_hc_payload(): baseline/Volt-Var HC summaries, gain and stored sweep-curve data.",
        "_dump() performs JSON-safe serialization and creates parent directories as needed.",
    ]
    for i,t in enumerate(lines): out.append(label(px+22,418+42*i,t,COL["text"],11.9,600,"start"))

    out.append(label(px+22,710,"Output/file semantics",COL["text"],17,700,"start"))
    lines=[
        "topology.json and profiles.json are system-level files.",
        "scenarios/<sid>.json is written compactly (indent=None) because annual payloads can be very large.",
        "publish_scenario_result(None) returns None and writes nothing.",
        "hc.json is omitted when hc_results is empty; comparison.json is omitted when the DataFrame is empty.",
        "The HC-stressed re-benchmark uses a separate output directory and repeats the same static write lifecycle.",
    ]
    for i,t in enumerate(lines): out.append(label(px+22,748+45*i,t,COL["text"],12.0,600,"start"))

    rects={"ready":ready,"syswrite":syswrite,"sysout":sysout,"scen":scen,"scenwrite":scenwrite,"scenout":scenout,"finish":finish,"finalwrite":finalwrite,"hcout":hcout,"cmpout":cmpout,"ret":ret,"files":files}
    diamonds={"dmore":dmore,"dhc":dhc,"dcmp":dcmp}
    for n,g in rects.items(): audit_rect_bounds(n,g,W,H,8)
    for n,g in diamonds.items(): audit_diamond_bounds(n,g,W,H,8)
    assert main.right < p1.left
    assert files.bottom < H-30
    assert hcout.right < dhc.left - 20
    assert cmpout.left > dcmp.right + 20
    assert abs(hcout.cy - dhc.cy) < 1e-9
    assert abs(cmpout.cy - dcmp.cy) < 1e-9
    assert hc_exit_x < ret.left - 2
    assert hcout.left < hc_exit_x < hcout.right

    write("flow_publisher_static_presentation_final",W,H,"Publisher static export lifecycle - presentation","\n".join(out))
    return W,H


if __name__ == "__main__":
    iw,ih=build_ieee(); pw,ph=build_presentation()
    print(f"Publisher static IEEE audited: {iw} x {ih}")
    print(f"Publisher static presentation audited: {pw} x {ph}")
    print(f"Outputs: {OUT}")
