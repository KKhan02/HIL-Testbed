from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import html
import cairosvg

OUT = Path(__file__).resolve().parent / "s3_flowcharts_top_level"
OUT.mkdir(parents=True, exist_ok=True)

COL = {
    "blue": "#0C447C",
    "green": "#0F6E56",
    "purple": "#3C3489",
    "red": "#993C1D",
    "dry": "#3B6D11",
    "decision": "#854F0B",
    "neutral": "#5F5E5A",
    "white": "#FFFFFF",
    "text": "#111111",
    "edge": "#263238",
    "hil": "#D85A30",
    "gate": "#7A5C12",
    "loop": "#3D8FD9",
    "panel": "#F8FAFC",
    "panel_border": "#D9E1E8",
    "detail_border": "#8A887F",
}

TARGET_GAP = 2.0


def esc(value):
    return html.escape(str(value))


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
    def cx(self): return self.x + self.w / 2
    @property
    def cy(self): return self.y + self.h / 2
    def a(self, side: str, gap: float = 0.0):
        if side == "top": return (self.cx, self.top - gap)
        if side == "bottom": return (self.cx, self.bottom + gap)
        if side == "left": return (self.left - gap, self.cy)
        if side == "right": return (self.right + gap, self.cy)
        raise ValueError(side)


@dataclass(frozen=True)
class D:
    cx: float; cy: float; w: float; h: float
    @property
    def left(self): return self.cx - self.w / 2
    @property
    def right(self): return self.cx + self.w / 2
    @property
    def top(self): return self.cy - self.h / 2
    @property
    def bottom(self): return self.cy + self.h / 2
    def a(self, side: str, gap: float = 0.0):
        if side == "top": return (self.cx, self.top - gap)
        if side == "bottom": return (self.cx, self.bottom + gap)
        if side == "left": return (self.left - gap, self.cy)
        if side == "right": return (self.right + gap, self.cy)
        raise ValueError(side)


@dataclass(frozen=True)
class C:
    cx: float; cy: float; r: float
    @property
    def left(self): return self.cx - self.r
    @property
    def right(self): return self.cx + self.r
    @property
    def top(self): return self.cy - self.r
    @property
    def bottom(self): return self.cy + self.r
    def a(self, side: str, gap: float = 0.0):
        if side == "top": return (self.cx, self.top - gap)
        if side == "bottom": return (self.cx, self.bottom + gap)
        if side == "left": return (self.left - gap, self.cy)
        if side == "right": return (self.right + gap, self.cy)
        raise ValueError(side)


def rect_svg(g: R, fill, rx=14, stroke="none", sw=0):
    return f'<rect x="{g.x}" y="{g.y}" width="{g.w}" height="{g.h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>'


def diamond_svg(g: D, fill):
    pts = f"{g.cx},{g.top} {g.right},{g.cy} {g.cx},{g.bottom} {g.left},{g.cy}"
    return f'<polygon points="{pts}" fill="{fill}"/>'


def circle_svg(g: C, fill):
    return f'<circle cx="{g.cx}" cy="{g.cy}" r="{g.r}" fill="{fill}"/>'


def label(x, y, text, fill, size=14, weight=700, anchor="middle"):
    return f'<text x="{x}" y="{y}" text-anchor="{anchor}" fill="{fill}" font-size="{size}" font-weight="{weight}">{esc(text)}</text>'


def text_lines(x, y, title, subs=(), title_size=17, sub_size=13, line_gap=20, fill="#FFFFFF", anchor="middle"):
    out = [f'<text x="{x}" y="{y}" text-anchor="{anchor}" class="node-title" font-size="{title_size}" fill="{fill}">{esc(title)}</text>']
    yy = y + line_gap
    for sub in subs:
        out.append(f'<text x="{x}" y="{yy}" text-anchor="{anchor}" class="node-sub" font-size="{sub_size}" fill="{fill}" opacity="0.90">{esc(sub)}</text>')
        yy += line_gap - 2
    return "\n".join(out)


def _clean_points(points):
    cleaned = []
    for pt in points:
        pt = (float(pt[0]), float(pt[1]))
        if not cleaned or pt != cleaned[-1]:
            cleaned.append(pt)
    assert len(cleaned) >= 2, "path requires at least two distinct points"
    for a, b in zip(cleaned, cleaned[1:]):
        dx = abs(a[0] - b[0]); dy = abs(a[1] - b[1])
        assert dx < 1e-9 or dy < 1e-9, f"non-orthogonal segment: {a} -> {b}"
        assert dx + dy > 0.5, f"zero/near-zero segment: {a} -> {b}"
    return cleaned


def path(points, cls="edge-dark", marker=True):
    points = _clean_points(points)
    markers = {"edge-dark":"arrowDark", "edge-white":"arrowWhite", "edge-hil":"arrowHil", "edge-gate":"arrowGate", "edge-loop":"arrowLoop", "edge-assoc":"arrowAssoc"}
    d = "M" + " L".join(f"{x:g} {y:g}" for x, y in points)
    mark = f' marker-end="url(#{markers[cls]})"' if marker else ""
    return f'<path d="{d}" class="{cls}"{mark}/>'


def direct(src, src_side, dst, dst_side, cls="edge-dark", gap=TARGET_GAP):
    return path([src.a(src_side), dst.a(dst_side, gap)], cls)


def ortho_vh(src, src_side, dst, dst_side, cls="edge-dark", gap=TARGET_GAP, bend_y=None):
    s = src.a(src_side); t = dst.a(dst_side, gap); by = t[1] if bend_y is None else bend_y
    return path([s, (s[0], by), (t[0], by), t], cls)


def ortho_hv(src, src_side, dst, dst_side, cls="edge-dark", gap=TARGET_GAP, bend_x=None):
    s = src.a(src_side); t = dst.a(dst_side, gap); bx = t[0] if bend_x is None else bend_x
    return path([s, (bx, s[1]), (bx, t[1]), t], cls)


def header(w, h, title):
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">
<title>{esc(title)}</title>
<defs>
  <marker id="arrowDark" markerWidth="6" markerHeight="6" refX="8" refY="4.5" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L9,4.5 L0,9 z" fill="#263238"/></marker>
  <marker id="arrowWhite" markerWidth="6" markerHeight="6" refX="8" refY="4.5" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L9,4.5 L0,9 z" fill="#FFFFFF"/></marker>
  <marker id="arrowHil" markerWidth="6" markerHeight="6" refX="8" refY="4.5" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L9,4.5 L0,9 z" fill="#D85A30"/></marker>
  <marker id="arrowGate" markerWidth="6" markerHeight="6" refX="8" refY="4.5" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L9,4.5 L0,9 z" fill="#7A5C12"/></marker>
  <marker id="arrowLoop" markerWidth="6" markerHeight="6" refX="8" refY="4.5" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L9,4.5 L0,9 z" fill="#3D8FD9"/></marker>
  <marker id="arrowAssoc" markerWidth="6" markerHeight="6" refX="8" refY="4.5" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L9,4.5 L0,9 z" fill="#8A887F"/></marker>
</defs>
<style>
text {{ font-family: Helvetica, Arial, sans-serif; }}
.node-title {{ font-weight:700; }} .node-sub {{ font-weight:400; }}
.edge-dark {{ fill:none; stroke:#263238; stroke-width:2.8; stroke-linejoin:round; stroke-linecap:round; }}
.edge-white {{ fill:none; stroke:#FFFFFF; stroke-width:3.0; stroke-linejoin:round; stroke-linecap:round; }}
.edge-hil {{ fill:none; stroke:#D85A30; stroke-width:3.0; stroke-linejoin:round; stroke-linecap:round; }}
.edge-gate {{ fill:none; stroke:#7A5C12; stroke-width:2.8; stroke-linejoin:round; stroke-linecap:round; }}
.edge-loop {{ fill:none; stroke:#3D8FD9; stroke-width:2.8; stroke-dasharray:7 5; stroke-linejoin:round; stroke-linecap:round; }}
.edge-assoc {{ fill:none; stroke:#8A887F; stroke-width:2.0; stroke-dasharray:6 4; stroke-linejoin:round; stroke-linecap:round; }}
</style>'''


def write(name, w, h, title, body):
    svg = header(w, h, title) + "\n" + body + "\n</svg>"
    svg_path = OUT / f"{name}.svg"; pdf_path = OUT / f"{name}.pdf"; png_path = OUT / f"{name}.png"
    svg_path.write_text(svg, encoding="utf-8")
    cairosvg.svg2pdf(bytestring=svg.encode(), write_to=str(pdf_path))
    cairosvg.svg2png(bytestring=svg.encode(), write_to=str(png_path), output_width=w*2, output_height=h*2)
    return svg_path, pdf_path, png_path


def draw_node(out, g: R, fill, title, subs=(), ts=16, ss=12, rx=14):
    out.append(rect_svg(g, fill, rx)); base = g.y + (28 if subs else g.h / 2 + 6)
    out.append(text_lines(g.cx, base, title, subs, ts, ss, 19))


def draw_decision(out, g: D, title, subs=(), ts=15, ss=11):
    out.append(diamond_svg(g, COL["decision"])); base = g.cy + (5 if not subs else -3)
    out.append(text_lines(g.cx, base, title, subs, ts, ss, 15))


def audit_rect_bounds(name, g: R, W, H, margin=0):
    assert g.left >= margin and g.top >= margin, f"{name}: top/left outside canvas"
    assert g.right <= W-margin and g.bottom <= H-margin, f"{name}: bottom/right outside canvas"


def audit_diamond_bounds(name, g: D, W, H, margin=0):
    assert g.left >= margin and g.right <= W-margin, f"{name}: x bounds invalid"
    assert g.top >= margin and g.bottom <= H-margin, f"{name}: y bounds invalid"


def build_ieee():
    W, H = 800, 1230
    out = [rect_svg(R(34, 26, 732, 1172), COL["panel"], 20, COL["panel_border"], 2)]
    out.append(label(W/2, 60, "Scenario 3 - Top-Level SVC Flow", COL["text"], 22, 700))

    cx = 400; main_w = 390; x = cx-main_w/2
    run = R(x, 88, main_w, 64)
    prep = R(x, 184, main_w, 82)
    ckpt = D(cx, 326, 220, 80)
    setup = R(x, 410, main_w, 102)
    loop = R(x, 550, main_w, 78)
    cleanup = R(x, 666, main_w, 72)
    skipmeta = R(35, 284, 155, 84)
    merge = C(cx, 800, 14)
    result = R(x, 846, main_w, 74)
    pubend = R(x, 956, main_w, 66)
    ret = R(x, 1058, main_w, 48)

    draw_node(out, run, COL["blue"], "run_scenario_3()", ("network, profiles, voltage limits",), 18, 13)
    draw_node(out, prep, COL["blue"], "Prepare simulation", ("adapt_profiles(); publisher start", "load checkpoint -> start_t"), 17, 12)
    draw_decision(out, ckpt, "checkpoint covers all T?", ts=15)
    draw_node(out, setup, COL["green"], "Prepare SVC", ("reset controllers/results; compute Q_MAX + k_q", "select fixed MV bus; create SVC sgen"), 16, 12)
    draw_node(out, loop, COL["neutral"], "Per-timestep SVC loop", ("pre-PF -> droop -> optional post-PF", "record / publish; see Diagram 2"), 16, 12)
    draw_node(out, cleanup, COL["red"], "finally: remove SVC", ("drop temporary sgen; reset_results()",), 16, 12)
    draw_node(out, skipmeta, COL["green"], "Recompute metadata", ("Q_MAX + k_q", "select SVC bus"), 13, 10)
    out.append(circle_svg(merge, COL["neutral"]))
    draw_node(out, result, COL["purple"], "ScenarioResult.from_records()", ("svc_bus + svc_q_max",), 16, 12)
    draw_node(out, pubend, COL["purple"], "on_scenario_end()", ("close checkpoint; final live event",), 16, 12)
    draw_node(out, ret, COL["blue"], "return ScenarioResult", (), 16)

    out.append(direct(run, "bottom", prep, "top"))
    out.append(direct(prep, "bottom", ckpt, "top"))
    out.append(direct(ckpt, "bottom", setup, "top"))
    out.append(label(cx+14, ckpt.bottom+19, "no / resume", COL["text"], 11, 700, "start"))
    out.append(direct(setup, "bottom", loop, "top"))
    out.append(direct(loop, "bottom", cleanup, "top"))
    out.append(direct(cleanup, "bottom", merge, "top"))

    # Completed checkpoint branch still recomputes SVC metadata and bus selection.
    out.append(direct(ckpt, "left", skipmeta, "right", "edge-gate"))
    out.append(label(ckpt.left-12, ckpt.cy-10, "yes - skip loop", COL["gate"], 11, 700, "end"))
    out.append(path([skipmeta.a("bottom"), (skipmeta.cx, merge.cy), merge.a("left", TARGET_GAP)], "edge-gate"))

    out.append(direct(merge, "bottom", result, "top"))
    out.append(direct(result, "bottom", pubend, "top"))
    out.append(direct(pubend, "bottom", ret, "top"))

    for name, g in {"run":run,"prep":prep,"setup":setup,"loop":loop,"cleanup":cleanup,"skipmeta":skipmeta,"result":result,"pubend":pubend,"ret":ret}.items():
        audit_rect_bounds(name,g,W,H,30)
    audit_diamond_bounds("ckpt",ckpt,W,H,30)
    assert prep.top-run.bottom >= 28
    assert ckpt.top-prep.bottom >= 20
    assert setup.top-ckpt.bottom >= 44
    assert loop.top-setup.bottom >= 38
    assert cleanup.top-loop.bottom >= 38
    assert merge.top-cleanup.bottom >= 48
    assert result.top-merge.bottom >= 28
    assert pubend.top-result.bottom >= 32
    assert ret.top-pubend.bottom >= 32
    assert skipmeta.right < setup.left-10
    assert skipmeta.cy == ckpt.cy

    write("flow_s3_top_ieee_final", W, H, "Scenario 3 top-level SVC flow - IEEE", "\n".join(out))
    return W,H


def build_presentation():
    """Presentation-specific S3 top-level view.

    Unlike S5, S3 benefits from extra placement detail because fixed-bus selection
    executes a stress solve with two fallbacks before the SVC is created. The main
    execution path stays on the left; right-side panels are explanatory only.
    """
    W,H = 1920,1080
    out = [rect_svg(R(0,0,W,H), COL["panel"], 0)]
    main_panel = R(28,24,1070,1032)
    out.append(rect_svg(main_panel, "#F4F8FB", 24, COL["panel_border"], 1.5))

    cx=500; w=720; x=cx-w/2
    run=R(x,42,w,62); prep=R(x,132,w,72); ckpt=D(cx,264,300,82)
    setup=R(x,340,w,92); loop=R(x,466,w,76); cleanup=R(x,576,w,68)
    skipmeta=R(875,225,190,78); merge=C(cx,700,15)
    result=R(x,740,w,70); pubend=R(x,844,w,62); ret=R(x,944,w,50)

    draw_node(out,run,COL["blue"],"run_scenario_3()",("network, profiles, voltage limits",),25,17)
    draw_node(out,prep,COL["blue"],"Prepare simulation",("adapt_profiles(); on_scenario_start()", "get_resume_records() -> start_t"),23,16)
    draw_decision(out,ckpt,"checkpoint covers all T?",ts=21)
    draw_node(out,setup,COL["green"],"Prepare SVC",("reset controllers/results; compute Q_MAX + k_q", "select fixed MV bus; create temporary SVC sgen"),22,15)
    draw_node(out,loop,COL["neutral"],"Per-timestep SVC loop",("pre-PF -> droop -> optional post-PF -> record",),22,15)
    draw_node(out,cleanup,COL["red"],"finally: remove temporary SVC",("drop svc_idx; pp.reset_results(net)",),21,15)
    draw_node(out,skipmeta,COL["green"],"Skip branch metadata",("compute Q_MAX + k_q", "select SVC bus again"),16,11)
    out.append(circle_svg(merge,COL["neutral"]))
    draw_node(out,result,COL["purple"],"ScenarioResult.from_records()",("resumed + newly simulated records; svc_bus / svc_q_max",),21,14)
    draw_node(out,pubend,COL["purple"],"on_scenario_end()",("close checkpoint; persist elapsed; final live event",),20,14)
    draw_node(out,ret,COL["blue"],"return ScenarioResult",(),20)

    out.append(direct(run,"bottom",prep,"top")); out.append(direct(prep,"bottom",ckpt,"top"))
    out.append(direct(ckpt,"bottom",setup,"top")); out.append(label(cx+18,ckpt.bottom+23,"no / resume",COL["text"],15,700,"start"))
    out.append(direct(setup,"bottom",loop,"top")); out.append(direct(loop,"bottom",cleanup,"top")); out.append(direct(cleanup,"bottom",merge,"top"))
    out.append(direct(ckpt,"right",skipmeta,"left","edge-gate"))
    out.append(label(ckpt.right+18,ckpt.cy-12,"yes - skip loop",COL["gate"],15,700,"start"))
    out.append(path([skipmeta.a("bottom"),(skipmeta.cx,merge.cy),merge.a("right",TARGET_GAP)],"edge-gate"))
    out.append(direct(merge,"bottom",result,"top")); out.append(direct(result,"bottom",pubend,"top")); out.append(direct(pubend,"bottom",ret,"top"))

    # Explanatory panels only, no execution transitions through them.
    px=1135; pw=750
    panels=[R(px,36,pw,300),R(px,358,pw,250),R(px,630,pw,390)]
    for p in panels: out.append(rect_svg(p,COL["white"],18,COL["detail_border"],1.2))
    out.append(label(px+24,66,"Fixed SVC bus selection",COL["text"],18,700,"start"))
    selection_lines=[
        "Eligible buses: in-service buses at the dominant MV level, excluding ext_grid slack.",
        "Primary: deepcopy(net) -> apply_overvoltage_stress() -> runpp().",
        "Stress state: sgen P = 0.90*sn_mva; load P and Q = 0.20*nominal; sgen Q=0.",
        "Primary result: choose the eligible MV bus with minimum vm_pu in that stress snapshot.",
        "Fallback 1: scan up to 7*24*6 profile steps; choose worst violated MV bus at first violation.",
        "Fallback 2: use mv_buses[-1].",
    ]
    for i,t in enumerate(selection_lines): out.append(label(px+24,100+34*i,t,COL["text"],12.5,600,"start"))
    out.append(label(px+24,390,"SVC parameter / lifecycle semantics",COL["text"],18,700,"start"))
    life_lines=[
        "Q_MAX = 0.20 * sum(net.trafo.sn_mva).",
        "k_q = Q_MAX / (0.04 - 0.01) = Q_MAX / 0.03.",
        "The SVC is created only on the simulation branch, after bus selection.",
        "Its sgen is removed in finally, so cleanup also runs if an uncaught loop exception escapes.",
    ]
    for i,t in enumerate(life_lines): out.append(label(px+24,426+36*i,t,COL["text"],12.5,600,"start"))
    out.append(label(px+24,662,"Checkpoint and failure semantics",COL["text"],18,700,"start"))
    sem_lines=[
        "on_scenario_start() runs before get_resume_records().",
        "A complete checkpoint skips SVC creation and the timestep loop, but still recomputes Q_MAX and bus selection.",
        "Partial resume creates a new SVC and continues at start_t using resumed_records as the prefix.",
        "Selection / setup exceptions occur before the loop try/finally and therefore propagate without SVC cleanup.",
        "Loop exceptions after SVC creation execute the finally cleanup before propagating.",
        "on_scenario_end() closes the checkpoint handle and appends the scenario_complete live event.",
    ]
    for i,t in enumerate(sem_lines): out.append(label(px+24,700+38*i,t,COL["text"],12.5,600,"start"))

    # Dashed explanatory associations.

    for name,g in {"run":run,"prep":prep,"setup":setup,"loop":loop,"cleanup":cleanup,"skipmeta":skipmeta,"result":result,"pubend":pubend,"ret":ret}.items(): audit_rect_bounds(name,g,W,H,18)
    audit_diamond_bounds("ckpt",ckpt,W,H,18)
    assert main_panel.right < panels[0].left
    assert setup.top-ckpt.bottom >= 35
    assert loop.top-setup.bottom >= 32
    assert cleanup.top-loop.bottom >= 34
    assert merge.top-cleanup.bottom >= 40
    assert result.top-merge.bottom >= 25
    assert pubend.top-result.bottom >= 30
    assert ret.top-pubend.bottom >= 32
    assert ret.bottom < H-30
    assert skipmeta.cy == ckpt.cy

    write("flow_s3_top_presentation_final",W,H,"Scenario 3 top-level SVC flow - presentation","\n".join(out))
    return W,H


if __name__ == "__main__":
    iw,ih=build_ieee(); pw,ph=build_presentation()
    print(f"IEEE top-level audited: {iw} x {ih}")
    print(f"Presentation top-level audited: {pw} x {ph}")
    print(f"Outputs: {OUT}")
