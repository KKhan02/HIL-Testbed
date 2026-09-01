from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import html
import cairosvg

OUT = Path(__file__).resolve().parent / "s5_flowcharts_per_timestep"
OUT.mkdir(parents=True, exist_ok=True)

COL = {
    "blue": "#0C447C",
    "blue_lane": "#3D8FD9",
    "green": "#0F6E56",
    "green_lane": "#1F9D78",
    "purple": "#3C3489",
    "purple_lane": "#6C63B8",
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

    def a(self, side: str, gap: float = 0.0):
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

    def a(self, side: str, gap: float = 0.0):
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

    def a(self, side: str, gap: float = 0.0):
        if side == "top": return (self.cx, self.top - gap)
        if side == "bottom": return (self.cx, self.bottom + gap)
        if side == "left": return (self.left - gap, self.cy)
        if side == "right": return (self.right + gap, self.cy)
        raise ValueError(side)


def rect_svg(g: R, fill, rx=14, stroke="none", sw=0):
    return (
        f'<rect x="{g.x}" y="{g.y}" width="{g.w}" height="{g.h}" rx="{rx}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>'
    )


def diamond_svg(g: D, fill):
    pts = f"{g.cx},{g.top} {g.right},{g.cy} {g.cx},{g.bottom} {g.left},{g.cy}"
    return f'<polygon points="{pts}" fill="{fill}"/>'


def circle_svg(g: C, fill):
    return f'<circle cx="{g.cx}" cy="{g.cy}" r="{g.r}" fill="{fill}"/>'


def label(x, y, text, fill, size=14, weight=700, anchor="middle"):
    return (
        f'<text x="{x}" y="{y}" text-anchor="{anchor}" fill="{fill}" '
        f'font-size="{size}" font-weight="{weight}">{esc(text)}</text>'
    )


def text_lines(x, y, title, subs=(), title_size=17, sub_size=13,
               line_gap=20, fill="#FFFFFF", anchor="middle"):
    out = [
        f'<text x="{x}" y="{y}" text-anchor="{anchor}" class="node-title" '
        f'font-size="{title_size}" fill="{fill}">{esc(title)}</text>'
    ]
    yy = y + line_gap
    for sub in subs:
        out.append(
            f'<text x="{x}" y="{yy}" text-anchor="{anchor}" class="node-sub" '
            f'font-size="{sub_size}" fill="{fill}" opacity="0.90">{esc(sub)}</text>'
        )
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
        dx = abs(a[0] - b[0])
        dy = abs(a[1] - b[1])
        assert dx < 1e-9 or dy < 1e-9, f"non-orthogonal segment: {a} -> {b}"
        assert dx + dy > 0.5, f"zero/near-zero segment: {a} -> {b}"
    return cleaned


def path(points, cls="edge-dark", marker=True):
    points = _clean_points(points)
    markers = {
        "edge-dark": "arrowDark",
        "edge-white": "arrowWhite",
        "edge-red": "arrowRed",
        "edge-gate": "arrowGate",
        "edge-loop": "arrowLoop",
    }
    d = "M" + " L".join(f"{x:g} {y:g}" for x, y in points)
    mark = f' marker-end="url(#{markers[cls]})"' if marker else ""
    return f'<path d="{d}" class="{cls}"{mark}/>'


def direct(src, src_side, dst, dst_side, cls="edge-dark", gap=TARGET_GAP):
    return path([src.a(src_side), dst.a(dst_side, gap)], cls)


def ortho_vh(src, src_side, dst, dst_side, cls="edge-dark", gap=TARGET_GAP,
             bend_y=None):
    s = src.a(src_side)
    t = dst.a(dst_side, gap)
    by = t[1] if bend_y is None else bend_y
    return path([s, (s[0], by), (t[0], by), t], cls)


def ortho_hv(src, src_side, dst, dst_side, cls="edge-dark", gap=TARGET_GAP,
             bend_x=None):
    s = src.a(src_side)
    t = dst.a(dst_side, gap)
    bx = t[0] if bend_x is None else bend_x
    return path([s, (bx, s[1]), (bx, t[1]), t], cls)


def header(w, h, title):
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">
<title>{esc(title)}</title>
<defs>
  <marker id="arrowDark" markerWidth="6" markerHeight="6" refX="8" refY="4.5" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L9,4.5 L0,9 z" fill="#263238"/></marker>
  <marker id="arrowWhite" markerWidth="6" markerHeight="6" refX="8" refY="4.5" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L9,4.5 L0,9 z" fill="#FFFFFF"/></marker>
  <marker id="arrowRed" markerWidth="6" markerHeight="6" refX="8" refY="4.5" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L9,4.5 L0,9 z" fill="#D85A30"/></marker>
  <marker id="arrowGate" markerWidth="6" markerHeight="6" refX="8" refY="4.5" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L9,4.5 L0,9 z" fill="#7A5C12"/></marker>
  <marker id="arrowLoop" markerWidth="6" markerHeight="6" refX="8" refY="4.5" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L9,4.5 L0,9 z" fill="#3D8FD9"/></marker>
</defs>
<style>
text {{ font-family: Helvetica, Arial, sans-serif; }}
.node-title {{ font-weight:700; }} .node-sub {{ font-weight:400; }}
.edge-dark {{ fill:none; stroke:#263238; stroke-width:2.8; stroke-linejoin:round; stroke-linecap:round; }}
.edge-white {{ fill:none; stroke:#FFFFFF; stroke-width:3.0; stroke-linejoin:round; stroke-linecap:round; }}
.edge-red {{ fill:none; stroke:#D85A30; stroke-width:3.0; stroke-linejoin:round; stroke-linecap:round; }}
.edge-gate {{ fill:none; stroke:#7A5C12; stroke-width:2.8; stroke-linejoin:round; stroke-linecap:round; }}
.edge-loop {{ fill:none; stroke:#3D8FD9; stroke-width:2.8; stroke-dasharray:7 5; stroke-linejoin:round; stroke-linecap:round; }}
</style>'''


def write(name, w, h, title, body):
    svg = header(w, h, title) + "\n" + body + "\n</svg>"
    svg_path = OUT / f"{name}.svg"
    pdf_path = OUT / f"{name}.pdf"
    png_path = OUT / f"{name}.png"
    svg_path.write_text(svg, encoding="utf-8")
    cairosvg.svg2pdf(bytestring=svg.encode(), write_to=str(pdf_path))
    cairosvg.svg2png(
        bytestring=svg.encode(),
        write_to=str(png_path),
        output_width=w * 2,
        output_height=h * 2,
    )
    return svg_path, pdf_path, png_path


def draw_node(out, g: R, fill, title, subs=(), ts=16, ss=12, rx=14):
    out.append(rect_svg(g, fill, rx))
    base = g.y + (28 if subs else g.h / 2 + 6)
    out.append(text_lines(g.cx, base, title, subs, ts, ss, 19))


def draw_decision(out, g: D, title, subs=(), ts=15, ss=11):
    out.append(diamond_svg(g, COL["decision"]))
    base = g.cy + (5 if not subs else -3)
    out.append(text_lines(g.cx, base, title, subs, ts, ss, 15))


def overlaps(a: R, b: R, pad=0.0):
    return not (
        a.right + pad <= b.left or b.right + pad <= a.left or
        a.bottom + pad <= b.top or b.bottom + pad <= a.top
    )


def audit_rect_bounds(name, g: R, W, H, margin=0):
    assert g.left >= margin, f"{name}: left outside canvas"
    assert g.top >= margin, f"{name}: top outside canvas"
    assert g.right <= W - margin, f"{name}: right outside canvas"
    assert g.bottom <= H - margin, f"{name}: bottom outside canvas"


def audit_diamond_bounds(name, g: D, W, H, margin=0):
    assert g.left >= margin and g.right <= W - margin, f"{name}: x bounds invalid"
    assert g.top >= margin and g.bottom <= H - margin, f"{name}: y bounds invalid"


def build_ieee():
    W, H = 780, 1060
    out = [rect_svg(R(42, 32, 696, 996), COL["panel"], 20, COL["panel_border"], 2)]
    out.append(label(W / 2, 68, "Scenario 5 - Top-Level AC OPF Flow", COL["text"], 22, 700))

    cx = 390
    main_w = 420
    x = cx - main_w / 2
    loop_rail = 104

    run = R(x, 98, main_w, 70)
    prep = R(x, 198, main_w, 90)
    setup = R(x, 318, main_w, 100)
    ckpt = D(cx, 486, 190, 78)
    loop = R(x, 566, main_w, 82)
    merge = C(cx, 714, 14)
    result = R(x, 758, main_w, 78)
    pubend = R(x, 868, main_w, 66)
    ret = R(x, 966, main_w, 48)

    draw_node(out, run, COL["blue"], "run_scenario_5()",
              ("network, profiles, OPF limits / options",), 18, 13)
    draw_node(out, prep, COL["blue"], "Prepare simulation",
              ("continuous bus index + adapt profiles", "publisher start + checkpoint resume"), 17, 13)
    draw_node(out, setup, COL["green"], "Prepare OPF model",
              ("clear controllers / results; static OPF bounds", "PF options; diagnostic; derive sn_rated"), 17, 13)
    draw_decision(out, ckpt, "checkpoint complete?", ts=15)
    draw_node(out, loop, COL["neutral"], "Per-timestep OPF loop",
              ("state build + runopp + record / publish", "see Diagram 2"), 17, 13)
    out.append(circle_svg(merge, COL["neutral"]))
    draw_node(out, result, COL["purple"], "ScenarioResult.from_records()",
              ("aggregate resumed + newly simulated records",), 16, 12)
    draw_node(out, pubend, COL["purple"], "on_scenario_end()",
              ("final live event / elapsed persistence",), 16, 12)
    draw_node(out, ret, COL["blue"], "return ScenarioResult", (), 16)

    out.append(direct(run, "bottom", prep, "top"))
    out.append(direct(prep, "bottom", setup, "top"))
    out.append(direct(setup, "bottom", ckpt, "top"))
    out.append(direct(ckpt, "bottom", loop, "top", "edge-dark"))
    out.append(label(cx + 12, ckpt.bottom + 20, "no", COL["text"], 11, 700, "start"))
    out.append(direct(loop, "bottom", merge, "top"))

    # Completed checkpoint bypass. It uses its own rail and joins only at the merge circle.
    out.append(path([ckpt.a("left"), (loop_rail, ckpt.cy), (loop_rail, merge.cy),
                     merge.a("left", TARGET_GAP)], "edge-gate"))
    out.append(label(ckpt.left - 10, ckpt.cy - 10, "yes - skip loop", COL["gate"], 11, 700, "end"))

    out.append(direct(merge, "bottom", result, "top"))
    out.append(direct(result, "bottom", pubend, "top"))
    out.append(direct(pubend, "bottom", ret, "top"))

    # Coordinate audit.
    for name, g in {
        "run": run, "prep": prep, "setup": setup, "loop": loop,
        "result": result, "pubend": pubend, "return": ret,
    }.items():
        audit_rect_bounds(name, g, W, H, 36)
    audit_diamond_bounds("checkpoint", ckpt, W, H, 36)
    assert loop_rail > 42 and loop_rail < ckpt.left - 20
    assert prep.top - run.bottom >= 28
    assert setup.top - prep.bottom >= 28
    assert ckpt.top - setup.bottom >= 28
    assert loop.top - ckpt.bottom >= 38
    assert merge.top - loop.bottom >= 48
    assert result.top - merge.bottom >= 28
    assert pubend.top - result.bottom >= 30
    assert ret.top - pubend.bottom >= 30

    write("flow_s5_top_ieee_final", W, H, "Scenario 5 top-level AC OPF flow - IEEE", "\n".join(out))
    return W, H


def build_presentation():
    W, H = 1280, 1160
    out = [rect_svg(R(0, 0, W, H), COL["panel"], 0)]

    # Conceptual lanes. No overall figure title; the slide supplies it.
    lane_master = R(28, 62, 400, 1070)
    lane_solver = R(444, 62, 400, 1070)
    lane_pub = R(860, 62, 392, 1070)
    out += [
        rect_svg(lane_master, COL["blue_lane"], 24),
        rect_svg(lane_solver, COL["green_lane"], 24),
        rect_svg(lane_pub, COL["purple_lane"], 24),
        label(lane_master.cx, 40, "Master - scenario_5_opf.py", COL["text"], 19, 700),
        label(lane_solver.cx, 40, "OPF setup / solver utilities", COL["text"], 19, 700),
        label(lane_pub.cx, 40, "Publisher / results / status", COL["text"], 19, 700),
    ]

    master_x = 78
    master_w = 300
    solver_x = 494
    solver_w = 300
    pub_x = 906
    pub_w = 300

    run = R(master_x, 94, master_w, 66)
    normalize = R(master_x, 188, master_w, 72)
    adapt = R(solver_x, 188, solver_w, 72)
    pubstart = R(pub_x, 286, pub_w, 84)
    reset = R(master_x, 396, master_w, 66)
    setup = R(solver_x, 492, solver_w, 96)
    rated = R(solver_x, 620, solver_w, 82)
    ckpt = D(lane_master.cx, 766, 176, 82)
    timesteps = R(master_x, 850, master_w, 78)
    mfinal = C(lane_pub.cx, 946, 15)
    result = R(pub_x, 978, pub_w, 68)
    pubend = R(pub_x, 1070, pub_w, 52)
    ret = R(master_x, 1070, master_w, 52)
    fail = R(pub_x, 500, pub_w, 84)

    draw_node(out, run, COL["blue"], "run_scenario_5()",
              ("network, profiles, OPF options",), 17, 12)
    draw_node(out, normalize, COL["blue"], "Validate + renumber buses",
              ("opf_init in {flat,pf}", "create_continuous_bus_index(start=0)"), 16, 11)
    draw_node(out, adapt, COL["green"], "adapt_profiles()",
              ("align load / DER profiles", "require at least one profiled DER"), 16, 11)
    draw_node(out, pubstart, COL["purple"], "Publisher start + resume",
              ("on_scenario_start()", "get_resume_records() -> start_t"), 16, 11)
    draw_node(out, reset, COL["blue"], "Reset network state",
              ("drop stale controllers; reset results",), 16, 11)
    draw_node(out, setup, COL["green"], "_setup_opf()",
              ("DER bound columns; ext_grid limits", "V / thermal constraints; pp.diagnostic"), 16, 11)
    draw_node(out, rated, COL["green"], "PF options + sn_rated",
              ("voltage_depend_loads=False", "sn_mva -> |p_mw| -> profile peak"), 16, 11)
    draw_decision(out, ckpt, "checkpoint covers all T?", ts=14)
    draw_node(out, timesteps, COL["neutral"], "Per-timestep OPF execution",
              ("range(start_t, T)", "see detailed Diagram 2"), 16, 11)
    out.append(circle_svg(mfinal, COL["neutral"]))
    draw_node(out, result, COL["purple"], "ScenarioResult.from_records()",
              ("records include resumed prefix",), 15, 11)
    draw_node(out, pubend, COL["purple"], "on_scenario_end()", (), 15)
    draw_node(out, ret, COL["blue"], "return ScenarioResult", (), 15)
    draw_node(out, fail, COL["red"], "Initialization failure semantics",
              ("invalid opf_init / no DER / no ext_grid / invalid sn_rated",
               "exception propagates; benchmark marks scenario failed"), 13, 9.5)

    # Main execution path.
    out.append(direct(run, "bottom", normalize, "top", "edge-white"))
    out.append(direct(normalize, "right", adapt, "left", "edge-white"))
    out.append(ortho_hv(adapt, "bottom", pubstart, "left", "edge-white", bend_x=lane_pub.left + 18))
    out.append(ortho_hv(pubstart, "left", reset, "right", "edge-white", bend_x=lane_master.right - 18))
    out.append(ortho_hv(reset, "right", setup, "left", "edge-white", bend_x=(lane_master.right + lane_solver.left) / 2))
    out.append(direct(setup, "bottom", rated, "top", "edge-white"))
    out.append(ortho_hv(rated, "left", ckpt, "top", "edge-white", bend_x=lane_master.cx))

    # Failure box is explanatory. It summarizes pre-loop exceptions without
    # creating a false normal execution branch from one particular setup call.

    out.append(direct(ckpt, "bottom", timesteps, "top", "edge-white"))
    out.append(label(ckpt.cx + 12, ckpt.bottom + 20, "no / resume", "#FFFFFF", 11, 700, "start"))
    out.append(ortho_hv(timesteps, "right", mfinal, "top", "edge-white", bend_x=mfinal.cx))

    # Completed checkpoint bypasses the timestep loop and joins at the final merge.
    bypass_x = lane_pub.left + 24
    out.append(path([ckpt.a("right"), (bypass_x, ckpt.cy), (bypass_x, mfinal.cy),
                     mfinal.a("left", TARGET_GAP)], "edge-gate"))
    out.append(label(ckpt.right + 12, ckpt.cy - 10, "yes - skip loop", "#FFFFFF", 11, 700, "start"))

    out.append(direct(mfinal, "bottom", result, "top", "edge-white"))
    out.append(direct(result, "bottom", pubend, "top", "edge-white"))
    out.append(direct(pubend, "left", ret, "right", "edge-white"))

    # Audit.
    nodes = {
        "run": run, "normalize": normalize, "adapt": adapt, "pubstart": pubstart,
        "reset": reset, "setup": setup, "rated": rated, "timesteps": timesteps,
        "result": result, "pubend": pubend, "return": ret, "failure": fail,
    }
    for name, g in nodes.items():
        audit_rect_bounds(name, g, W, H, 18)
    audit_diamond_bounds("checkpoint", ckpt, W, H, 18)
    for lane_name, lane in {"master_lane": lane_master, "solver_lane": lane_solver, "publisher_lane": lane_pub}.items():
        audit_rect_bounds(lane_name, lane, W, H, 18)
    assert normalize.bottom < reset.top
    assert setup.bottom < rated.top
    assert ckpt.bottom < timesteps.top
    assert mfinal.top - timesteps.bottom >= 3
    assert result.top - mfinal.bottom >= 15
    assert pubend.top - result.bottom >= 20
    assert fail.left > lane_pub.left and fail.right < lane_pub.right
    assert bypass_x > ckpt.right + 20 and bypass_x < mfinal.left - 10

    write("flow_s5_top_presentation_final", W, H,
          "Scenario 5 top-level AC OPF flow - presentation", "\n".join(out))
    return W, H


if __name__ == "__main__":
    iw, ih = build_ieee()
    pw, ph = build_presentation()
    print(f"IEEE top-level audited: {iw} x {ih}")
    print(f"Presentation top-level audited: {pw} x {ph}")
    print(f"Outputs: {OUT}")