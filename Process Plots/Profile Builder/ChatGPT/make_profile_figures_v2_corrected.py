from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import html
import json
import re
import xml.etree.ElementTree as ET

import cairosvg

OUT = Path(__file__).resolve().parent / "profile_flowcharts_v2_corrected"
OUT.mkdir(parents=True, exist_ok=True)

COL = {
    "blue": "#0C447C",
    "green": "#0F6E56",
    "purple": "#3C3489",
    "red": "#993C1D",
    "decision": "#854F0B",
    "neutral": "#5F5E5A",
    "white": "#FFFFFF",
    "text": "#111111",
    "gate": "#7A5C12",
    "loop": "#3D8FD9",
    "panel": "#F8FAFC",
    "panel_border": "#D9E1E8",
    "detail_border": "#8A887F",
}
TARGET_GAP = 2.0


def esc(v):
    return html.escape(str(v))


@dataclass(frozen=True)
class R:
    x: float
    y: float
    w: float
    h: float
    @property
    def left(self): return self.x
    @property
    def right(self): return self.x + self.w
    @property
    def top(self): return self.y
    @property
    def bottom(self): return self.y + self.h
    @property
    def cx(self): return self.x + self.w / 2
    @property
    def cy(self): return self.y + self.h / 2
    def a(self, side, gap=0.0):
        if side == "top": return (self.cx, self.top - gap)
        if side == "bottom": return (self.cx, self.bottom + gap)
        if side == "left": return (self.left - gap, self.cy)
        if side == "right": return (self.right + gap, self.cy)
        raise ValueError(side)


@dataclass(frozen=True)
class D:
    cx: float
    cy: float
    w: float
    h: float
    @property
    def left(self): return self.cx - self.w / 2
    @property
    def right(self): return self.cx + self.w / 2
    @property
    def top(self): return self.cy - self.h / 2
    @property
    def bottom(self): return self.cy + self.h / 2
    def a(self, side, gap=0.0):
        if side == "top": return (self.cx, self.top - gap)
        if side == "bottom": return (self.cx, self.bottom + gap)
        if side == "left": return (self.left - gap, self.cy)
        if side == "right": return (self.right + gap, self.cy)
        raise ValueError(side)


@dataclass(frozen=True)
class C:
    cx: float
    cy: float
    r: float
    @property
    def left(self): return self.cx - self.r
    @property
    def right(self): return self.cx + self.r
    @property
    def top(self): return self.cy - self.r
    @property
    def bottom(self): return self.cy + self.r
    def a(self, side, gap=0.0):
        if side == "top": return (self.cx, self.top - gap)
        if side == "bottom": return (self.cx, self.bottom + gap)
        if side == "left": return (self.left - gap, self.cy)
        if side == "right": return (self.right + gap, self.cy)
        raise ValueError(side)


def rect_svg(g, fill, rx=12, stroke="none", sw=0):
    return f'<rect x="{g.x}" y="{g.y}" width="{g.w}" height="{g.h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>'


def diamond_svg(g, fill):
    return f'<polygon points="{g.cx},{g.top} {g.right},{g.cy} {g.cx},{g.bottom} {g.left},{g.cy}" fill="{fill}"/>'


def circle_svg(g, fill):
    return f'<circle cx="{g.cx}" cy="{g.cy}" r="{g.r}" fill="{fill}"/>'


def label(x, y, text, fill, size=13, weight=700, anchor="middle"):
    return f'<text x="{x}" y="{y}" text-anchor="{anchor}" fill="{fill}" font-size="{size}" font-weight="{weight}">{esc(text)}</text>'


def text_lines(x, y, title, subs=(), title_size=15, sub_size=11, line_gap=16, fill="#FFFFFF", anchor="middle"):
    out = [f'<text x="{x}" y="{y}" text-anchor="{anchor}" class="node-title" font-size="{title_size}" fill="{fill}">{esc(title)}</text>']
    yy = y + line_gap
    for s in subs:
        out.append(f'<text x="{x}" y="{yy}" text-anchor="{anchor}" class="node-sub" font-size="{sub_size}" fill="{fill}" opacity="0.90">{esc(s)}</text>')
        yy += line_gap - 2
    return "\n".join(out)


def _clean_points(points):
    cleaned = []
    for p in points:
        p = (float(p[0]), float(p[1]))
        if not cleaned or p != cleaned[-1]:
            cleaned.append(p)
    assert len(cleaned) >= 2
    for a, b in zip(cleaned, cleaned[1:]):
        dx = abs(a[0] - b[0]); dy = abs(a[1] - b[1])
        assert dx < 1e-9 or dy < 1e-9, f"non-orthogonal segment {a}->{b}"
        assert dx + dy > 0.5, f"zero/near-zero segment {a}->{b}"
    return cleaned


def path(points, cls="edge-dark", marker=True):
    points = _clean_points(points)
    markers = {
        "edge-dark": "arrowDark",
        "edge-gate": "arrowGate",
        "edge-loop": "arrowLoop",
        "edge-assoc": "arrowAssoc",
        "edge-red": "arrowRed",
    }
    d = "M" + " L".join(f"{x:g} {y:g}" for x, y in points)
    mark = f' marker-end="url(#{markers[cls]})"' if marker else ""
    return f'<path d="{d}" class="{cls}"{mark}/>'


def direct(src, ss, dst, ds, cls="edge-dark", gap=TARGET_GAP):
    return path([src.a(ss), dst.a(ds, gap)], cls)


def ortho_vh(src, ss, dst, ds, cls="edge-dark", gap=TARGET_GAP, bend_y=None):
    s = src.a(ss); t = dst.a(ds, gap)
    by = t[1] if bend_y is None else bend_y
    return path([s, (s[0], by), (t[0], by), t], cls)


def ortho_hv(src, ss, dst, ds, cls="edge-dark", gap=TARGET_GAP, bend_x=None):
    s = src.a(ss); t = dst.a(ds, gap)
    bx = t[0] if bend_x is None else bend_x
    return path([s, (bx, s[1]), (bx, t[1]), t], cls)


def header(w, h, title):
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}"><title>{esc(title)}</title>
<defs>
<marker id="arrowDark" markerWidth="6" markerHeight="6" refX="8" refY="4.5" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L9,4.5 L0,9 z" fill="#263238"/></marker>
<marker id="arrowGate" markerWidth="6" markerHeight="6" refX="8" refY="4.5" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L9,4.5 L0,9 z" fill="#7A5C12"/></marker>
<marker id="arrowLoop" markerWidth="6" markerHeight="6" refX="8" refY="4.5" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L9,4.5 L0,9 z" fill="#3D8FD9"/></marker>
<marker id="arrowAssoc" markerWidth="6" markerHeight="6" refX="8" refY="4.5" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L9,4.5 L0,9 z" fill="#8A887F"/></marker>
<marker id="arrowRed" markerWidth="6" markerHeight="6" refX="8" refY="4.5" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L9,4.5 L0,9 z" fill="#993C1D"/></marker>
</defs>
<style>
text {{font-family:Helvetica,Arial,sans-serif}} .node-title{{font-weight:700}} .node-sub{{font-weight:400}}
.edge-dark{{fill:none;stroke:#263238;stroke-width:2.7;stroke-linejoin:round;stroke-linecap:round}}
.edge-gate{{fill:none;stroke:#7A5C12;stroke-width:2.7;stroke-linejoin:round;stroke-linecap:round}}
.edge-loop{{fill:none;stroke:#3D8FD9;stroke-width:2.6;stroke-dasharray:7 5;stroke-linejoin:round;stroke-linecap:round}}
.edge-assoc{{fill:none;stroke:#8A887F;stroke-width:2;stroke-dasharray:5 4;stroke-linejoin:round;stroke-linecap:round}}
.edge-red{{fill:none;stroke:#993C1D;stroke-width:2.8;stroke-linejoin:round;stroke-linecap:round}}
</style>'''


def write(name, w, h, title, body):
    svg = header(w, h, title) + "\n" + body + "\n</svg>"
    sp = OUT / f"{name}.svg"; pp = OUT / f"{name}.pdf"; pn = OUT / f"{name}.png"
    sp.write_text(svg, encoding="utf-8")
    cairosvg.svg2pdf(bytestring=svg.encode(), write_to=str(pp))
    cairosvg.svg2png(bytestring=svg.encode(), write_to=str(pn), output_width=w*2, output_height=h*2)
    return sp, pp, pn


def draw_node(out, g, fill, title, subs=(), ts=14, ss=10, rx=12):
    out.append(rect_svg(g, fill, rx))
    line_gap = 15
    n = 1 + len(subs)
    if n == 1:
        base = g.cy + 5
    else:
        total = line_gap + (n - 2) * (line_gap - 2)
        base = g.cy - total / 2 + 5
    out.append(text_lines(g.cx, base, title, subs, ts, ss, line_gap))


def draw_decision(out, g, title, subs=(), ts=13, ss=10):
    out.append(diamond_svg(g, COL["decision"]))
    line_gap = 14
    n = 1 + len(subs)
    if n == 1:
        base = g.cy + 4
    else:
        total = line_gap + (n - 2) * (line_gap - 2)
        base = g.cy - total / 2 + 4
    out.append(text_lines(g.cx, base, title, subs, ts, ss, line_gap))


def draw_panel(out, g, title, lines, title_size=16, line_size=11.5, line_step=31):
    out.append(rect_svg(g, COL["white"], 16, COL["detail_border"], 1.2))
    out.append(label(g.x + 22, g.y + 30, title, COL["text"], title_size, 700, "start"))
    yy = g.y + 66
    for line in lines:
        out.append(label(g.x + 22, yy, line, COL["text"], line_size, 600, "start"))
        yy += line_step


def audit_rect_bounds(name, g, W, H, m=0):
    assert g.left >= m and g.top >= m and g.right <= W-m and g.bottom <= H-m, f"{name}: bounds"


def audit_diamond_bounds(name, g, W, H, m=0):
    assert g.left >= m and g.right <= W-m and g.top >= m and g.bottom <= H-m, f"{name}: bounds"


# ---------------------------------------------------------------------------
# SimBench native path - IEEE
# ---------------------------------------------------------------------------
def build_simbench_ieee():
    W, H = 880, 1330
    out = [rect_svg(R(18,18,W-36,H-36), COL["panel"], 18, COL["panel_border"], 1.6)]
    rects = {}; diamonds = {}
    def Rg(n,x,y,w,h): g=R(x,y,w,h); rects[n]=g; return g
    def Dg(n,x,y,w,h): g=D(x,y,w,h); diamonds[n]=g; return g
    cx=440; bw=450; x=cx-bw/2; mr=12
    out.append(label(W/2,48,"profile_builder - SimBench Native Profile Path",COL["text"],20,700))

    start=Rg("start",x,78,bw,58); draw_node(out,start,COL["blue"],"build_annual_profiles()",("net, net_name, simbench_code",),15,10.5)
    dtype=Dg("dtype",cx,190,250,62); draw_decision(out,dtype,"net_type == 'simbench'?",ts=12.5)
    fallback=Rg("fallback",42,159,150,62); draw_node(out,fallback,COL["neutral"],"No",("use fallback path",),11,8.7,10)
    code=Dg("code",cx,286,250,62); draw_decision(out,code,"simbench_code provided?",ts=12.5)
    err=Rg("err",688,255,150,62); draw_node(out,err,COL["red"],"Raise ValueError",("code is required",),10.5,8.3,10)
    sbcall=Rg("sbcall",x,354,bw,88); draw_node(out,sbcall,COL["purple"],"Load SimBench profile source",("raw_net = get_simbench_net(code)", "profiles = get_absolute_values(...profiles=True)", "times = profiles[(load,p_mw)].index"),13,9.2)
    dtime=Dg("dtime",cx,510,280,64); draw_decision(out,dtime,"profile index is DatetimeIndex?",ts=11.8)
    keep=Rg("keep",78,480,200,60); draw_node(out,keep,COL["neutral"],"Use native timestamps",("times = profile index",),10.8,8.5,10)
    rebuild=Rg("rebuild",588,470,240,80); draw_node(out,rebuild,COL["green"],"Reconstruct timestamps",("start = 2016-01-01", "periods = len(times), freq = 15 min", "tz = Europe/Berlin"),11,8.4,10)
    mt=C(cx,590,mr); out.append(circle_svg(mt,COL["neutral"]))
    frames=Rg("frames",x,628,bw,92); draw_node(out,frames,COL["blue"],"Build load + sgen frames",("load_df = profiles[(load,p_mw)] -> index=times -> clip >=0", "sgen_prof = profiles[(sgen,p_mw)] -> shared source frame", "select only columns that exist in returned profile tables"),12.2,8.6)
    masks=Rg("masks",x,754,bw,92); draw_node(out,masks,COL["green"],"Classify DER columns",("PV mask: type contains pv | solar | lv_res", "wind mask: type contains wind | wp", "pv_idx/wind_idx from net.sgen then intersect with sgen_prof"),12.2,8.6)
    clean=Rg("clean",x,880,bw,92); draw_node(out,clean,COL["green"],"Physical cleanup",("PV night-zero for hour >=22 or <=4", "PV and wind clip >=0 to remove scaling artefacts", "empty selections remain empty DataFrames"),12.2,8.6)
    result=Rg("result",x,1008,bw,78); draw_node(out,result,COL["purple"],"Assemble profile result",("result = {load, pv, wind, times, net_type}", "PV/wind contain only classified SimBench sgen columns"),12.8,8.8)
    extreme=Rg("extreme",x,1120,bw,74); draw_node(out,extreme,COL["green"],"find_extreme_days(result)",("safe_sum handles empty frames", "daily means -> max/min DER day and max/min load day"),12.3,8.8)
    ret=Rg("ret",x,1230,bw,52); draw_node(out,ret,COL["blue"],"return profiles dict",(),13)

    out.append(direct(start,"bottom",dtype,"top"))
    out.append(direct(dtype,"left",fallback,"right","edge-gate")); out.append(label(dtype.left-8,dtype.cy-8,"no",COL["gate"],10,700,"end"))
    out.append(direct(dtype,"bottom",code,"top")); out.append(label(dtype.cx+10,dtype.bottom+16,"yes",COL["text"],10,700,"start"))
    out.append(direct(code,"right",err,"left","edge-red")); out.append(label(code.right+8,code.cy-8,"no",COL["red"],10,700,"start"))
    out.append(direct(code,"bottom",sbcall,"top")); out.append(label(code.cx+10,code.bottom+16,"yes",COL["text"],10,700,"start"))
    out.append(direct(sbcall,"bottom",dtime,"top"))
    out.append(direct(dtime,"left",keep,"right","edge-gate")); out.append(label(dtime.left-8,dtime.cy-8,"yes",COL["gate"],10,700,"end"))
    out.append(direct(dtime,"right",rebuild,"left","edge-gate")); out.append(label(dtime.right+8,dtime.cy-8,"no",COL["gate"],10,700,"start"))
    out.append(ortho_vh(keep,"bottom",mt,"left","edge-gate",bend_y=mt.cy)); out.append(ortho_vh(rebuild,"bottom",mt,"right","edge-gate",bend_y=mt.cy))
    out.append(direct(mt,"bottom",frames,"top")); out.append(direct(frames,"bottom",masks,"top")); out.append(direct(masks,"bottom",clean,"top")); out.append(direct(clean,"bottom",result,"top")); out.append(direct(result,"bottom",extreme,"top")); out.append(direct(extreme,"bottom",ret,"top"))

    for n,g in rects.items(): audit_rect_bounds(n,g,W,H,14)
    for n,g in diamonds.items(): audit_diamond_bounds(n,g,W,H,14)
    assert dtime.top-sbcall.bottom>=28
    assert frames.top-mt.bottom>=20
    assert ret.bottom < H-30

    write("flow_profile_simbench_ieee_v2",W,H,"profile_builder SimBench native profile path - IEEE","\n".join(out))
    return W,H


# ---------------------------------------------------------------------------
# SimBench native path - presentation
# ---------------------------------------------------------------------------
def build_simbench_presentation():
    W,H=1920,1080
    out=[rect_svg(R(0,0,W,H),COL["panel"],0)]
    rects={}; diamonds={}
    def Rg(n,x,y,w,h): g=R(x,y,w,h); rects[n]=g; return g
    def Dg(n,x,y,w,h): g=D(x,y,w,h); diamonds[n]=g; return g
    cx=470; bw=650; x=cx-bw/2; mr=12

    start=Rg("start",x,26,bw,48); draw_node(out,start,COL["blue"],"build_annual_profiles()",("net, net_name, simbench_code",),17,11)
    dtype=Dg("dtype",cx,126,280,46); draw_decision(out,dtype,"net_type == 'simbench'?",ts=13.5)
    fallback=Rg("fallback",55,103,190,46); draw_node(out,fallback,COL["neutral"],"No: fallback path",(),11.5)
    code=Dg("code",cx,204,270,46); draw_decision(out,code,"simbench_code provided?",ts=13)
    err=Rg("err",760,181,190,46); draw_node(out,err,COL["red"],"ValueError",("missing code",),11,8.5,10)
    sbcall=Rg("sbcall",x,244,bw,66); draw_node(out,sbcall,COL["purple"],"Load SimBench source network + absolute profiles",("get_simbench_net(code) -> get_absolute_values(...profiles=True)", "times = load-profile index"),13.5,9)
    dtime=Dg("dtime",cx,358,300,48); draw_decision(out,dtime,"profile index already DatetimeIndex?",ts=12.3)
    keep=Rg("keep",65,334,220,48); draw_node(out,keep,COL["neutral"],"Use native timestamps",("times = profile index",),11.2,8.2)
    rebuild=Rg("rebuild",705,326,250,64); draw_node(out,rebuild,COL["green"],"Reconstruct time axis",("2016-01-01; 15-min cadence", "Europe/Berlin; periods=len(times)"),11.1,8.2)
    mt=C(cx,420,mr); out.append(circle_svg(mt,COL["neutral"]))
    frames=Rg("frames",x,446,bw,76); draw_node(out,frames,COL["blue"],"Extract load and sgen active-power frames",("load_df -> index=times -> clip >=0", "sgen_prof -> index=times -> shared source for PV/wind"),12.8,8.7)
    masks=Rg("masks",x,548,bw,76); draw_node(out,masks,COL["green"],"Split sgen columns into PV and wind",("PV mask: pv | solar | lv_res", "Wind mask: wind | wp; intersect with available profile columns"),12.8,8.7)
    clean=Rg("clean",x,650,bw,76); draw_node(out,clean,COL["green"],"Apply physical cleanup",("PV night-zero for local hour >=22 or <=4", "clip PV/wind >=0; empty selections stay empty"),12.8,8.7)
    result=Rg("result",x,752,bw,66); draw_node(out,result,COL["purple"],"Assemble common output schema",("result = {load, pv, wind, times, net_type}", "add extreme_days after result assembly"),12.8,8.7)
    extreme=Rg("extreme",x,844,bw,66); draw_node(out,extreme,COL["green"],"find_extreme_days(result)",("safe_sum -> total PV + wind = DER; total load", "filter daily means to days with >=80% max daily sample count"),12.5,8.5)
    ret=Rg("ret",x,936,bw,44); draw_node(out,ret,COL["blue"],"return profiles dict",(),13.5)

    out.append(direct(start,"bottom",dtype,"top")); out.append(direct(dtype,"left",fallback,"right","edge-gate")); out.append(label(dtype.left-10,dtype.cy-7,"no",COL["gate"],11,700,"end")); out.append(direct(dtype,"bottom",code,"top")); out.append(label(dtype.cx+12,dtype.bottom+15,"yes",COL["text"],11,700,"start"))
    out.append(direct(code,"right",err,"left","edge-red")); out.append(label(code.right+10,code.cy-7,"no",COL["red"],11,700,"start")); out.append(direct(code,"bottom",sbcall,"top")); out.append(label(code.cx+12,code.bottom+15,"yes",COL["text"],11,700,"start"))
    out.append(direct(sbcall,"bottom",dtime,"top")); out.append(direct(dtime,"left",keep,"right","edge-gate")); out.append(label(dtime.left-10,dtime.cy-7,"yes",COL["gate"],11,700,"end")); out.append(direct(dtime,"right",rebuild,"left","edge-gate")); out.append(label(dtime.right+10,dtime.cy-7,"no",COL["gate"],11,700,"start")); out.append(ortho_vh(keep,"bottom",mt,"left","edge-gate",bend_y=mt.cy)); out.append(ortho_vh(rebuild,"bottom",mt,"right","edge-gate",bend_y=mt.cy)); out.append(direct(mt,"bottom",frames,"top"))
    out.append(direct(frames,"bottom",masks,"top")); out.append(direct(masks,"bottom",clean,"top")); out.append(direct(clean,"bottom",result,"top")); out.append(direct(result,"bottom",extreme,"top")); out.append(direct(extreme,"bottom",ret,"top"))

    px=1080; pw=800
    p1=R(px,28,pw,260); p2=R(px,310,pw,302); p3=R(px,634,pw,356)
    draw_panel(out,p1,"SimBench source + cadence",[
        "get_simbench_net(simbench_code) supplies the source network used for profile lookup.",
        "get_absolute_values(...profiles_instead_of_study_cases=True) supplies load/sgen P profiles.",
        "If the returned load-profile index is not a DatetimeIndex, timestamps are reconstructed.",
        "The reconstruction uses 2016-01-01, 15-minute cadence, and Europe/Berlin timezone.",
        "The branch stays at native SimBench resolution; there is no 10-minute resampling here.",
    ],16,11.2,37)
    draw_panel(out,p2,"DER classification + physical cleanup",[
        "PV classification uses net.sgen.type containing pv, solar, or lv_res.",
        "Wind classification uses net.sgen.type containing wind or wp.",
        "Only element indices also present in sgen_prof are retained as profile columns.",
        "PV is forced to zero at local hours >=22 or <=4.",
        "PV and wind are clipped at zero to remove small negative profile-scaling artefacts.",
        "If no matching columns exist, the resulting PV/wind DataFrame is simply empty.",
    ],16,11.0,39)
    draw_panel(out,p3,"Return schema + extreme-day calculation",[
        "Returned keys are load, pv, wind, times, net_type, and extreme_days.",
        "safe_sum() converts a missing or empty profile frame into a zero series on times.",
        "total_der = sum(PV columns) + sum(wind columns); total_load = sum(load columns).",
        "Each series is resampled to daily mean before selecting max/min days.",
        "Days with fewer than 80% of the maximum daily sample count are excluded.",
        "A series with no values or zero maximum yields None for that extreme-day key.",
        "Otherwise max_der, min_der, max_load, and min_load are YYYY-MM-DD strings.",
    ],16,10.8,39)

    for n,g in rects.items(): audit_rect_bounds(n,g,W,H,6)
    for n,g in diamonds.items(): audit_diamond_bounds(n,g,W,H,6)
    assert p1.left > 1000
    assert ret.bottom < H-80

    write("flow_profile_simbench_presentation_v2",W,H,"profile_builder SimBench native profile path - presentation","\n".join(out))
    return W,H


# ---------------------------------------------------------------------------
# Fallback / CIGRE path - IEEE
# ---------------------------------------------------------------------------
def build_fallback_ieee():
    W,H=920,1520
    out=[rect_svg(R(18,18,W-36,H-36),COL["panel"],18,COL["panel_border"],1.6)]
    rects={}; diamonds={}
    def Rg(n,x,y,w,h): g=R(x,y,w,h); rects[n]=g; return g
    def Dg(n,x,y,w,h): g=D(x,y,w,h); diamonds[n]=g; return g
    cx=460; bw=430; x=cx-bw/2; mr=12
    out.append(label(W/2,48,"profile_builder - Fallback Weather-to-Power Path",COL["text"],20,700))

    start=Rg("start",x,78,bw,58); draw_node(out,start,COL["blue"],"build_annual_profiles()",("net, net_name, data_dir, file_map, col_map",),15,10.2)
    load=Rg("load",x,170,bw,84); draw_node(out,load,COL["purple"],"Load weather source tables",("solar_df = load_dwd_solar(...)", "wind_raw = load_dwd_wind(...)", "temp_df = load_dwd_temperature(...)"),13,9.0)
    dsolar=Dg("dsolar",cx,320,280,64); draw_decision(out,dsolar,"custom RAD-G file mapped?",ts=12.3)
    custom=Rg("custom",55,289,225,62); draw_node(out,custom,COL["neutral"],"Solar unit semantics",("explicit RAD-G in file_map", "treat source as W/m2 directly"),10.8,8.2,10)
    dwd=Rg("dwd",640,281,225,78); draw_node(out,dwd,COL["green"],"DWD solar conversion",("if RAD-G not explicitly mapped:", "infer median sample interval", "J/cm2 -> W/m2 then clip GHI >=0"),10.6,8.0,10)
    ms=C(cx,404,mr); out.append(circle_svg(ms,COL["neutral"]))
    align=Rg("align",x,442,bw,92); draw_node(out,align,COL["blue"],"Weather cleanup + alignment",("loaders numeric-coerce; solar/wind drop NaN after cleanup", "drop duplicate timestamps in each weather table", "intersect solar, wind, and temperature indices"),12.0,8.4)
    dyear=Dg("dyear",cx,600,280,64); draw_decision(out,dyear,"complete calendar year exists?",ts=11.8)
    complete=Rg("complete",50,561,255,78); draw_node(out,complete,COL["green"],"Select complete year",("scan sorted years ascending", "pick first year covering months 1..12", "implementation therefore selects earliest complete year"),10.2,7.8,10)
    most=Rg("most",620,566,270,68); draw_node(out,most,COL["neutral"],"No complete year",("choose year with most aligned timesteps", "emit warning in console"),10.5,8.0,10)
    my=C(cx,694,mr); out.append(circle_svg(my,COL["neutral"]))
    trim=Rg("trim",x,732,bw,74); draw_node(out,trim,COL["blue"],"Trim weather tables to target_year",("times = times[times.year == target_year]", "solar_df / wind_raw / temp_df = .loc[times]"),12.3,8.8)
    build=Rg("build",x,840,bw,72); draw_node(out,build,COL["blue"],"Build per-element profiles",("load_df = compute_load_profiles_bdew(net, times)", "loop PV sgens -> compute_pv_profile(...)", "loop wind sgens -> compute_wind_profile(...)"),12.2,8.5)

    loadp=Rg("loadp",40,950,255,138); draw_node(out,loadp,COL["green"],"BDEW 2025 load profiles",("metadata -> H25/G25/L25", "missing metadata -> seeded 82/15/3 mix", "one raw shape per unique BDEW type", "normalize each shape by its own peak", "scale each load so profile max = net.load.p_mw", "clip >=0; fill NaN = 0"),9.7,7.2,10)
    pvp=Rg("pvp",332.5,942,255,154); draw_node(out,pvp,COL["green"],"PV per sgen",("Erbs: GHI -> DNI + DHI -> POA", "POA fill NaN=0; clip >=0; POA <10 W/m2 ->0", "NOCT air temp or 15 C fallback", "temperature-corrected DC -> inverter AC", "reindex fill NaN=0; clip 0..rated", "night-zero for hour >=22 or <=4", "empty DataFrame if no PV units found"),9.4,7.0,10)
    windp=Rg("windp",625,950,255,138); draw_node(out,windp,COL["green"],"Wind per sgen",("loader clips wind speed >=0 and drops NaN", "piecewise cubic power curve", "0 below 3 m/s; cubic to 12 m/s", "rated to 25 m/s; 0 at/above cut-out", "clip output 0..rated", "empty DataFrame if no wind units found"),9.7,7.2,10)
    mp=C(cx,1132,mr); out.append(circle_svg(mp,COL["neutral"]))
    result=Rg("result",x,1168,bw,82); draw_node(out,result,COL["purple"],"Assemble common profile result",("result = {load, pv, wind, times, net_type}", "fallback schema matches the SimBench branch"),12.8,8.9)
    extreme=Rg("extreme",x,1286,bw,78); draw_node(out,extreme,COL["green"],"find_extreme_days(result)",("safe_sum handles empty PV/wind frames", "daily means -> max/min DER and load day"),12.3,8.8)
    ret=Rg("ret",x,1400,bw,52); draw_node(out,ret,COL["blue"],"return profiles dict",(),13)

    out.append(direct(start,"bottom",load,"top")); out.append(direct(load,"bottom",dsolar,"top"))
    out.append(direct(dsolar,"left",custom,"right","edge-gate")); out.append(label(dsolar.left-8,dsolar.cy-8,"yes",COL["gate"],10,700,"end"))
    out.append(direct(dsolar,"right",dwd,"left","edge-gate")); out.append(label(dsolar.right+8,dsolar.cy-8,"no",COL["gate"],10,700,"start"))
    out.append(ortho_vh(custom,"bottom",ms,"left","edge-gate",bend_y=ms.cy)); out.append(ortho_vh(dwd,"bottom",ms,"right","edge-gate",bend_y=ms.cy)); out.append(direct(ms,"bottom",align,"top")); out.append(direct(align,"bottom",dyear,"top"))
    out.append(direct(dyear,"left",complete,"right","edge-gate")); out.append(label(dyear.left-8,dyear.cy-8,"yes",COL["gate"],10,700,"end"))
    out.append(direct(dyear,"right",most,"left","edge-gate")); out.append(label(dyear.right+8,dyear.cy-8,"no",COL["gate"],10,700,"start"))
    out.append(ortho_vh(complete,"bottom",my,"left","edge-gate",bend_y=my.cy)); out.append(ortho_vh(most,"bottom",my,"right","edge-gate",bend_y=my.cy)); out.append(direct(my,"bottom",trim,"top")); out.append(direct(trim,"bottom",build,"top"))
    branch_y=930
    out.append(path([build.a("bottom"),(build.cx,branch_y),(loadp.cx,branch_y),loadp.a("top",TARGET_GAP)],"edge-dark"))
    out.append(direct(build,"bottom",pvp,"top"))
    out.append(path([build.a("bottom"),(build.cx,branch_y),(windp.cx,branch_y),windp.a("top",TARGET_GAP)],"edge-dark"))
    out.append(ortho_vh(loadp,"bottom",mp,"left","edge-dark",bend_y=mp.cy))
    out.append(direct(pvp,"bottom",mp,"top"))
    out.append(ortho_vh(windp,"bottom",mp,"right","edge-dark",bend_y=mp.cy))
    out.append(direct(mp,"bottom",result,"top")); out.append(direct(result,"bottom",extreme,"top")); out.append(direct(extreme,"bottom",ret,"top"))

    for n,g in rects.items(): audit_rect_bounds(n,g,W,H,14)
    for n,g in diamonds.items(): audit_diamond_bounds(n,g,W,H,14)
    assert abs(custom.cy-dsolar.cy) < 1e-9
    assert abs(dwd.cy-dsolar.cy) < 1e-9
    assert abs(complete.cy-dyear.cy) < 1e-9
    assert abs(most.cy-dyear.cy) < 1e-9
    assert trim.top-my.bottom>=20
    assert build.top-trim.bottom>=25
    assert mp.top-max(loadp.bottom,pvp.bottom,windp.bottom)>=24
    assert ret.bottom < H-25

    write("flow_profile_fallback_ieee_v3",W,H,"profile_builder fallback weather-to-power path - IEEE","\n".join(out))
    return W,H


# ---------------------------------------------------------------------------
# Fallback / CIGRE path - presentation
# ---------------------------------------------------------------------------
def build_fallback_presentation():
    W,H=1920,1080
    out=[rect_svg(R(0,0,W,H),COL["panel"],0)]
    rects={}; diamonds={}
    def Rg(n,x,y,w,h): g=R(x,y,w,h); rects[n]=g; return g
    def Dg(n,x,y,w,h): g=D(x,y,w,h); diamonds[n]=g; return g
    cx=465; bw=650; x=cx-bw/2; mr=11

    start=Rg("start",x,22,bw,42); draw_node(out,start,COL["blue"],"CIGRE / fallback branch",("build_annual_profiles() after non-SimBench routing",),14,9.2)
    load=Rg("load",x,88,bw,58); draw_node(out,load,COL["purple"],"Load solar + wind + temperature",("file_map/col_map optional; DWD defaults otherwise", "read_weather_csv localizes UTC -> Europe/Berlin and numeric-coerces values"),12.8,8.6)
    dsolar=Dg("dsolar",cx,198,270,42); draw_decision(out,dsolar,"custom RAD-G file mapped?",ts=12)
    custom=Rg("custom",60,177,225,42); draw_node(out,custom,COL["neutral"],"Use solar as W/m2",(),10.8)
    dwd=Rg("dwd",655,170,250,56); draw_node(out,dwd,COL["green"],"DWD J/cm2 -> W/m2",("infer median sample interval; clip GHI >=0",),10.8,8.1)
    ms=C(cx,252,mr); out.append(circle_svg(ms,COL["neutral"]))
    align=Rg("align",x,278,bw,58); draw_node(out,align,COL["blue"],"Weather cleanup + common time index",("deduplicate each table; intersect solar, wind and temperature timestamps", "solar/wind loaders drop NaN after physical clipping"),12.1,8.4)
    dyear=Dg("dyear",cx,388,290,42); draw_decision(out,dyear,"complete calendar year exists?",ts=11.8)
    complete=Rg("complete",45,365,255,46); draw_node(out,complete,COL["green"],"Use first complete year",("sorted ascending year order",),10.5,8)
    most=Rg("most",650,361,285,54); draw_node(out,most,COL["neutral"],"Use year with most samples",("warning when no complete year exists",),10.5,8)
    my=C(cx,442,mr); out.append(circle_svg(my,COL["neutral"]))
    trim=Rg("trim",x,468,bw,52); draw_node(out,trim,COL["blue"],"Trim solar / wind / temperature to target_year",("all three tables use identical times after .loc[times]",),12.2,8.4)
    build=Rg("build",x,544,bw,46); draw_node(out,build,COL["blue"],"Build per-element profiles",("BDEW loads | PV sgens | wind sgens",),12.5,8.2)
    loadp=Rg("loadp",40,618,270,112); draw_node(out,loadp,COL["green"],"Loads: BDEW 2025",("metadata -> H25/G25/L25", "missing metadata -> deterministic 82/15/3 mix", "normalize raw BDEW shape by peak", "scale maximum to each net.load.p_mw", "clip >=0; fill NaN=0"),9.8,7.4)
    pvp=Rg("pvp",330,618,270,112); draw_node(out,pvp,COL["green"],"PV per sgen",("Erbs GHI -> DNI/DHI -> POA", "POA NaN/negative cleanup + 10 W/m2 threshold", "NOCT cell temperature -> corrected DC -> inverter AC", "reindex/fill 0; clip 0..rated", "night-zero >=22 or <=4; empty if no PV"),9.4,7.1)
    windp=Rg("windp",620,618,270,112); draw_node(out,windp,COL["green"],"Wind per sgen",("wind speed clipped >=0 / NaN dropped by loader", "piecewise cubic: 3 m/s -> 12 m/s -> 25 m/s", "0 below cut-in and at/above cut-out", "clip output 0..rated; empty if no wind"),9.6,7.3)
    mp=C(cx,770,mr); out.append(circle_svg(mp,COL["neutral"]))
    result=Rg("result",x,798,bw,52); draw_node(out,result,COL["purple"],"Assemble common output schema",("load | pv | wind | times | net_type",),12.2,8.3)
    extreme=Rg("extreme",x,876,bw,56); draw_node(out,extreme,COL["green"],"find_extreme_days()",("safe_sum handles empty PV/wind; daily means -> max/min DER and load",),12.0,8.2)
    ret=Rg("ret",x,958,bw,40); draw_node(out,ret,COL["blue"],"return profiles dict",(),12.5)

    out.append(direct(start,"bottom",load,"top")); out.append(direct(load,"bottom",dsolar,"top"))
    out.append(direct(dsolar,"left",custom,"right","edge-gate")); out.append(label(dsolar.left-9,dsolar.cy-7,"yes",COL["gate"],10,700,"end"))
    out.append(direct(dsolar,"right",dwd,"left","edge-gate")); out.append(label(dsolar.right+9,dsolar.cy-7,"no",COL["gate"],10,700,"start"))
    out.append(ortho_vh(custom,"bottom",ms,"left","edge-gate",bend_y=ms.cy)); out.append(ortho_vh(dwd,"bottom",ms,"right","edge-gate",bend_y=ms.cy)); out.append(direct(ms,"bottom",align,"top")); out.append(direct(align,"bottom",dyear,"top"))
    out.append(direct(dyear,"left",complete,"right","edge-gate")); out.append(label(dyear.left-9,dyear.cy-7,"yes",COL["gate"],10,700,"end"))
    out.append(direct(dyear,"right",most,"left","edge-gate")); out.append(label(dyear.right+9,dyear.cy-7,"no",COL["gate"],10,700,"start"))
    out.append(ortho_vh(complete,"bottom",my,"left","edge-gate",bend_y=my.cy)); out.append(ortho_vh(most,"bottom",my,"right","edge-gate",bend_y=my.cy)); out.append(direct(my,"bottom",trim,"top")); out.append(direct(trim,"bottom",build,"top"))
    branch_y=604
    out.append(path([build.a("bottom"),(build.cx,branch_y),(loadp.cx,branch_y),loadp.a("top",TARGET_GAP)],"edge-dark"))
    out.append(direct(build,"bottom",pvp,"top"))
    out.append(path([build.a("bottom"),(build.cx,branch_y),(windp.cx,branch_y),windp.a("top",TARGET_GAP)],"edge-dark"))
    out.append(ortho_vh(loadp,"bottom",mp,"left","edge-dark",bend_y=mp.cy))
    out.append(direct(pvp,"bottom",mp,"top"))
    out.append(ortho_vh(windp,"bottom",mp,"right","edge-dark",bend_y=mp.cy))
    out.append(direct(mp,"bottom",result,"top")); out.append(direct(result,"bottom",extreme,"top")); out.append(direct(extreme,"bottom",ret,"top"))

    px=1040; pw=840
    p1=R(px,26,pw,292); p2=R(px,338,pw,330); p3=R(px,688,pw,336)
    draw_panel(out,p1,"Weather ingestion + cleanup",[
        "find_dwd_file(): explicit file_map first; otherwise DWD CDC glob naming.",
        "read_weather_csv(): parse timestamps, sort, UTC-localize, convert to Europe/Berlin.",
        "Values are numeric-coerced; non-numeric source entries become NaN.",
        "Solar: explicit RAD-G mapping is assumed W/m2; otherwise DWD J/cm2 is converted.",
        "DWD solar interval is median timestamp spacing, with 600 s only as short-file fallback.",
        "Solar GHI and wind speed are clipped >=0 and NaNs are dropped by their loaders.",
        "Temperature is assumed degC and NaNs are dropped; then all timestamps are intersected.",
    ],16,10.6,33)
    draw_panel(out,p2,"Per-element conversion models",[
        "Loads: H25/G25/L25 from oemof.demand; semantic metadata is used first.",
        "Unclassified loads receive seeded 82% H25 / 15% G25 / 3% L25 assignment.",
        "BDEW time variation is preserved: raw shape / raw peak, then multiply by load p_mw.",
        "Thus net.load.p_mw anchors the profile maximum, not a constant demand value.",
        "PV: Erbs -> POA -> NOCT temperature -> efficiency correction -> DC -> inverter AC.",
        "PV cleanup: POA NaN/negative cleanup, 10 W/m2 cutoff, rated clipping, night-zero.",
        "Wind: 0 below 3 m/s, cubic to 12 m/s, rated to 25 m/s, then 0 at cut-out.",
        "Wind output is clipped 0..rated; absent PV/wind types leave an empty DataFrame.",
    ],16,10.6,33)
    draw_panel(out,p3,"Year selection, result schema, and edge cases",[
        "Implementation note: the code scans candidate years in ascending order and breaks on the first complete year.",
        "Therefore the current implementation selects the earliest complete year, despite nearby wording that suggests 'most recent'.",
        "If no complete year exists, the code chooses the year with the most aligned timesteps and emits a warning.",
        "All three profile families are returned with the common schema load, pv, wind, times, and net_type.",
        "If no matching PV or wind sgen types are found, the corresponding DataFrame remains empty.",
        "safe_sum in find_extreme_days keeps daily aggregation valid even when PV or wind frames are empty.",
        "The fallback path computes extreme days after assembling the common schema.",
    ],16,10.6,33)

    for n,g in rects.items(): audit_rect_bounds(n,g,W,H,12)
    for n,g in diamonds.items(): audit_diamond_bounds(n,g,W,H,12)
    assert abs(custom.cy-dsolar.cy) < 1e-9
    assert abs(dwd.cy-dsolar.cy) < 1e-9
    assert abs(complete.cy-dyear.cy) < 1e-9
    assert abs(most.cy-dyear.cy) < 1e-9
    assert ret.bottom < H-60

    write("flow_profile_fallback_presentation_v3",W,H,"profile_builder fallback weather-to-power path - presentation","\n".join(out))
    return W,H


def run_all():
    return {
        "simbench_ieee": build_simbench_ieee(),
        "simbench_presentation": build_simbench_presentation(),
        "fallback_ieee": build_fallback_ieee(),
        "fallback_presentation": build_fallback_presentation(),
    }


if __name__ == "__main__":
    dims = run_all()
    for k,v in dims.items(): print(k, v)
    print("Outputs:", OUT)