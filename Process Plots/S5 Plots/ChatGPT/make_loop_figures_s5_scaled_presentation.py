from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import html
import math
import cairosvg

OUT = Path(__file__).resolve().parent / "s5_flowcharts_per_timestep"
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
    "dry_edge": "#6B9E26",
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


def rect_svg(g: R, fill, rx=12, stroke="none", sw=0):
    return (
        f'<rect x="{g.x}" y="{g.y}" width="{g.w}" height="{g.h}" rx="{rx}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>'
    )


def diamond_svg(g: D, fill):
    pts = f"{g.cx},{g.top} {g.right},{g.cy} {g.cx},{g.bottom} {g.left},{g.cy}"
    return f'<polygon points="{pts}" fill="{fill}"/>'


def circle_svg(g: C, fill, stroke="none", sw=0):
    return (
        f'<circle cx="{g.cx}" cy="{g.cy}" r="{g.r}" fill="{fill}" '
        f'stroke="{stroke}" stroke-width="{sw}"/>'
    )


def label(x, y, text, fill, size=13, weight=700, anchor="middle"):
    return (
        f'<text x="{x}" y="{y}" text-anchor="{anchor}" fill="{fill}" '
        f'font-size="{size}" font-weight="{weight}">{esc(text)}</text>'
    )


def text_lines(x, y, title, subs=(), title_size=15, sub_size=11,
               line_gap=16, fill="#FFFFFF", anchor="middle"):
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
        "edge-hil": "arrowHil",
        "edge-dry": "arrowDry",
        "edge-gate": "arrowGate",
        "edge-loop": "arrowLoop",
        "edge-assoc": "arrowAssoc",
    }
    d = "M" + " L".join(f"{x:g} {y:g}" for x, y in points)
    mark = f' marker-end="url(#{markers[cls]})"' if marker else ""
    return f'<path d="{d}" class="{cls}"{mark}/>'


def direct(src, src_side, dst, dst_side, cls="edge-dark", gap=TARGET_GAP):
    s = src.a(src_side)
    t = dst.a(dst_side, gap)
    return path([s, t], cls)


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
  <marker id="arrowHil" markerWidth="6" markerHeight="6" refX="8" refY="4.5" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L9,4.5 L0,9 z" fill="#D85A30"/></marker>
  <marker id="arrowDry" markerWidth="6" markerHeight="6" refX="8" refY="4.5" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L9,4.5 L0,9 z" fill="#6B9E26"/></marker>
  <marker id="arrowGate" markerWidth="6" markerHeight="6" refX="8" refY="4.5" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L9,4.5 L0,9 z" fill="#7A5C12"/></marker>
  <marker id="arrowLoop" markerWidth="6" markerHeight="6" refX="8" refY="4.5" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L9,4.5 L0,9 z" fill="#3D8FD9"/></marker>
  <marker id="arrowAssoc" markerWidth="6" markerHeight="6" refX="8" refY="4.5" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L9,4.5 L0,9 z" fill="#8A887F"/></marker>
</defs>
<style>
text {{ font-family: Helvetica, Arial, sans-serif; }}
.node-title {{ font-weight:700; }} .node-sub {{ font-weight:400; }}
.edge-dark {{ fill:none; stroke:#263238; stroke-width:2.6; stroke-linejoin:round; stroke-linecap:round; }}
.edge-white {{ fill:none; stroke:#FFFFFF; stroke-width:2.8; stroke-linejoin:round; stroke-linecap:round; }}
.edge-hil {{ fill:none; stroke:#D85A30; stroke-width:2.8; stroke-linejoin:round; stroke-linecap:round; }}
.edge-dry {{ fill:none; stroke:#6B9E26; stroke-width:2.8; stroke-linejoin:round; stroke-linecap:round; }}
.edge-gate {{ fill:none; stroke:#7A5C12; stroke-width:2.6; stroke-linejoin:round; stroke-linecap:round; }}
.edge-loop {{ fill:none; stroke:#3D8FD9; stroke-width:2.6; stroke-dasharray:7 5; stroke-linejoin:round; stroke-linecap:round; }}
.edge-assoc {{ fill:none; stroke:#8A887F; stroke-width:2.0; stroke-dasharray:5 4; stroke-linejoin:round; stroke-linecap:round; }}
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


def draw_node(out, g: R, fill, title, subs=(), ts=14, ss=10, rx=12):
    out.append(rect_svg(g, fill, rx))
    base = g.y + (20 if subs else g.h / 2 + 5)
    out.append(text_lines(g.cx, base, title, subs, ts, ss, 15))


def draw_detail(out, g: R, fill, title, subs=(), ts=13, ss=10):
    out.append(rect_svg(g, fill, 10, COL["detail_border"], 1.2))
    base = g.y + (20 if subs else g.h / 2 + 5)
    out.append(text_lines(g.cx, base, title, subs, ts, ss, 15))


def draw_decision(out, g: D, title, subs=(), ts=13, ss=10):
    out.append(diamond_svg(g, COL["decision"]))
    base = g.cy + (4 if not subs else -4)
    out.append(text_lines(g.cx, base, title, subs, ts, ss, 14))


# ----------------------------------------------------------------------
# Geometry audit utilities
# ----------------------------------------------------------------------

def overlaps(a: R, b: R, pad=0.0):
    return not (
        a.right + pad <= b.left or b.right + pad <= a.left or
        a.bottom + pad <= b.top or b.bottom + pad <= a.top
    )


def audit_rect_bounds(name, g: R, W, H, margin=0):
    assert g.left >= margin, f"{name}: left={g.left} outside canvas"
    assert g.top >= margin, f"{name}: top={g.top} outside canvas"
    assert g.right <= W - margin, f"{name}: right={g.right} outside canvas"
    assert g.bottom <= H - margin, f"{name}: bottom={g.bottom} outside canvas"


def audit_diamond_bounds(name, g: D, W, H, margin=0):
    assert g.left >= margin and g.right <= W - margin, f"{name}: x bounds invalid"
    assert g.top >= margin and g.bottom <= H - margin, f"{name}: y bounds invalid"


def assert_separated(name_a, a: R, name_b, b: R, pad=2):
    assert not overlaps(a, b, pad), f"Unexpected overlap: {name_a} vs {name_b}"


def run_audit(W, H, rects, diamonds, separation_pairs):
    for name, g in rects.items():
        audit_rect_bounds(name, g, W, H)
    for name, g in diamonds.items():
        audit_diamond_bounds(name, g, W, H)
    for a, b, pad in separation_pairs:
        assert_separated(a, rects[a], b, rects[b], pad)


# ======================================================================
# IEEE COMPACT FIGURE
# ======================================================================

def build_ieee():
    W, H = 840, 1320
    out = [rect_svg(R(14, 14, W - 28, H - 28), COL["panel"], 16, COL["panel_border"], 1.5)]
    rects = {}
    diamonds = {}
    sep = []

    def Rg(name, x, y, w, h):
        g = R(x, y, w, h); rects[name] = g; return g

    def Dg(name, x, y, w, h):
        g = D(x, y, w, h); diamonds[name] = g; return g

    cx = 350
    loop_rail = 28
    merge_r = 12

    out.append(label(W / 2, 34, "Scenario 5 - Per-Timestep AC OPF Execution", COL["text"], 17, 700))

    loop = Dg("loop", cx, 82, 150, 50)
    out.append(diamond_svg(loop, COL["neutral"]))
    out.append(text_lines(loop.cx, loop.cy - 4, "for each t", ("in time_steps",), 14, 10, 14))

    partition = Rg("partition", 58, 120, 584, 360)
    out.append(rect_svg(partition, "none", 16, COL["detail_border"], 1.0))
    out.append(label(partition.right - 250, partition.y - 6,
                     "_write_timestep_opf_state() - scenario_5_opf.py",
                     COL["text"], 10, 700, anchor="start"))

    pbound = Rg("pbound", 190, 142, 320, 58)
    draw_node(out, pbound, COL["green"], "[1] available P + active DERs",
              ("p_bound=max(profile,0); active if > eps",), 13, 9)
    out.append(direct(loop, "bottom", pbound, "top"))

    qlim = Rg("qlim", 190, 224, 320, 64)
    draw_node(out, qlim, COL["green"], "[2] compute Q capability",
              ("q_lim=min(Q_RATIO*sn, sqrt(sn^2-p^2))",), 13, 9)
    out.append(direct(pbound, "bottom", qlim, "top"))

    ders = Rg("ders", 190, 312, 320, 68)
    draw_node(out, ders, COL["green"], "[3] rebuild DER OPF state",
              ("reset all DERs to 0 / uncontrollable", "activate only p_bound>eps with P/Q bounds"), 12, 9)
    out.append(direct(qlim, "bottom", ders, "top"))

    costs = Rg("costs", 190, 404, 320, 58)
    draw_node(out, costs, COL["green"], "[4] write loads + rebuild costs",
              ("ext_grid +0.001; active DER -1.0",), 13, 9)
    out.append(direct(ders, "bottom", costs, "top"))

    dbg = Dg("debug", cx, 524, 212, 58)
    draw_decision(out, dbg, "debug requested for t?", ("debug_opf_task + first-only gate",), 11, 8.5)
    out.append(direct(costs, "bottom", dbg, "top"))

    dbgbox = Rg("dbgbox", 570, 496, 220, 56)
    draw_node(out, dbgbox, COL["neutral"], "_print_opf_debug()",
              ("opf_task + bounds checks",), 11, 9, 10)
    out.append(direct(dbg, "right", dbgbox, "left", "edge-gate"))
    out.append(label(dbg.right + 8, dbg.cy - 8, "yes", COL["gate"], 10, 700, anchor="start"))

    mdbg = C(cx, 596, merge_r)
    out.append(direct(dbg, "bottom", mdbg, "top"))
    out.append(label(dbg.cx + 10, dbg.bottom + 16, "no", COL["text"], 10, 700, anchor="start"))
    out.append(ortho_vh(dbgbox, "bottom", mdbg, "right", "edge-gate"))
    out.append(circle_svg(mdbg, COL["neutral"]))

    runopp = Rg("runopp", 190, 628, 320, 54)
    draw_node(out, runopp, COL["green"], "[5] pp.runopp()",
              ("init=opf_init; OPF_SOLVER=cyipopt",), 13, 9)
    out.append(direct(mdbg, "bottom", runopp, "top"))

    outcome = Dg("outcome", cx, 742, 224, 70)
    draw_decision(out, outcome, "runopp outcome?", ts=13)
    out.append(direct(runopp, "bottom", outcome, "top"))

    failed = Rg("failed", 42, 709, 160, 66)
    draw_node(out, failed, COL["red"], "OPF not converged",
              ("empty result series", "P/Q/target=None"), 11, 9, 10)
    out.append(direct(outcome, "left", failed, "right", "edge-hil"))
    out.append(label(outcome.left - 8, outcome.cy - 14, "caught", COL["hil"], 9.5, 700, anchor="end"))

    abort = Rg("abort", 604, 709, 180, 66)
    draw_node(out, abort, COL["red"], "Other exception",
              ("propagates; scenario aborts", "benchmark marks failed"), 11, 9, 10)
    out.append(direct(outcome, "right", abort, "left", "edge-hil"))
    out.append(label(outcome.right + 8, outcome.cy - 9, "uncaught", COL["hil"], 9.5, 700, anchor="start"))

    success = Rg("success", 190, 820, 320, 84)
    draw_node(out, success, COL["blue"], "Converged optimum",
              ("read V / line / trafo / DER P,Q", "threshold violations; losses + grid import", "p_target=p_bound"), 12, 9)
    out.append(direct(outcome, "bottom", success, "top"))
    out.append(label(outcome.cx + 10, outcome.bottom + 16, "converged", COL["text"], 10, 700, anchor="start"))

    mrec = C(cx, 940, merge_r)
    out.append(direct(success, "bottom", mrec, "top"))
    out.append(ortho_vh(failed, "bottom", mrec, "left", "edge-hil", bend_y=mrec.cy))
    out.append(circle_svg(mrec, COL["neutral"]))

    record = Rg("record", 150, 974, 400, 70)
    draw_node(out, record, COL["purple"], "[6] build TimestepRecord",
              ("append -> publish_fn.on_timestep()", "curtailment_needed = bool(recorded violations)"), 12, 9)
    out.append(direct(mrec, "bottom", record, "top"))

    periodic = Dg("periodic", cx, 1090, 196, 54)
    draw_decision(out, periodic, "t % 96 == 0?", ts=12)
    out.append(direct(record, "bottom", periodic, "top"))

    progress = Rg("progress", 570, 1060, 220, 60)
    draw_node(out, progress, COL["purple"], "Periodic progress",
              ("optional live CSV partial result", "log active DER / violations / failures"), 10.5, 8.5, 10)
    out.append(direct(periodic, "right", progress, "left", "edge-gate"))
    out.append(label(periodic.right + 8, periodic.cy - 8, "yes", COL["gate"], 10, 700, anchor="start"))

    mnext = C(cx, 1160, merge_r)

    # no -> merge from above
    out.append(direct(periodic, "bottom", mnext, "top"))
    out.append(label(
        periodic.cx + 10,
        periodic.bottom + 16,
        "no",
        COL["text"],
        10,
        700,
        anchor="start",
    ))

    # yes -> Periodic progress -> same merge from the right
    # This return path stays entirely above the "more timesteps?" decision.
    out.append(ortho_vh(
        progress,
        "bottom",
        mnext,
        "right",
        "edge-gate",
        bend_y=mnext.cy,
    ))

    out.append(circle_svg(mnext, COL["neutral"]))

    # After both periodic branches merge, continue vertically downward.
    more = Dg("more", cx, 1220, 170, 46)
    draw_decision(out, more, "more timesteps?", ts=12)

    out.append(direct(
        mnext,
        "bottom",
        more,
        "top",
        "edge-dark",
    ))

    # no -> leave timestep loop and aggregate records
    result = Rg("result", 500, 1198, 146, 44)
    draw_node(
        out,
        result,
        COL["purple"],
        "from_records()",
        (),
        11,
        8,
        9,
    )

    out.append(direct(
        more,
        "right",
        result,
        "left",
    ))

    out.append(label(
        more.right + 8,
        more.cy - 7,
        "no",
        COL["text"],
        10,
        700,
        anchor="start",
    ))

    # yes -> dedicated next-timestep loopback rail
    loopback_y = 1275

    out.append(path([
        more.a("bottom"),
        (more.cx, loopback_y),
        (loop_rail, loopback_y),
        (loop_rail, loop.cy),
        loop.a("left", TARGET_GAP),
    ], "edge-loop"))

    out.append(label(
        more.cx + 12,
        more.bottom + 16,
        "yes",
        COL["loop"],
        10,
        700,
        anchor="start",
    ))

    sep += [
        ("failed", "success", 8), ("success", "abort", 8),
        ("record", "progress", 8), ("progress", "result", 8),
    ]
    run_audit(W, H, rects, diamonds, sep)

    # Partition and vertical shaft checks.
    assert partition.top < pbound.top and partition.bottom > costs.bottom
    assert partition.bottom < dbg.top
    assert pbound.top - loop.bottom >= 33
    assert qlim.top - pbound.bottom >= 20
    assert ders.top - qlim.bottom >= 20
    assert costs.top - ders.bottom >= 20
    assert dbg.top - costs.bottom >= 30
    assert mdbg.top - dbg.bottom >= 31
    assert runopp.top - mdbg.bottom >= 20
    assert outcome.top - runopp.bottom >= 25
    assert success.top - outcome.bottom >= 43
    assert mrec.top - success.bottom >= 24
    assert record.top - mrec.bottom >= 22
    assert periodic.top - record.bottom >= 19
    assert mnext.top - periodic.bottom >= 31
    assert more.top - mnext.bottom >= 20
    assert loop_rail < failed.left - 10
    assert result.bottom < loopback_y - 16
    assert loopback_y < H - 18

    write("flow_s5_loop_ieee_final", W, H,
          "Scenario 5 per-timestep AC OPF flow - IEEE", "\n".join(out))
    return W, H


# ======================================================================
# PRESENTATION FIGURE - 1920 x 1080
# ======================================================================


def build_presentation():
    """Presentation-scaled export of the IEEE per-timestep graph.

    The graph, branch semantics, merge points, and loopback structure are the
    same as build_ieee().  Only the 16:9 geometry and typography are enlarged
    for slide projection.  There are no presentation-only detail panels.
    """
    W, H = 1920, 1080
    out = [rect_svg(R(22, 16, W - 44, H - 32), COL["panel"], 24,
                    COL["panel_border"], 1.5)]
    rects = {}
    diamonds = {}
    sep = []

    def Rg(name, x, y, w, h):
        g = R(x, y, w, h); rects[name] = g; return g

    def Dg(name, x, y, w, h):
        g = D(x, y, w, h); diamonds[name] = g; return g

    # Same graph as IEEE, expanded horizontally and compressed only enough
    # vertically to fit a standard 16:9 slide.
    cx = 820
    loop_rail = 42
    merge_r = 14

    loop = Dg("loop", cx, 42, 250, 52)
    out.append(diamond_svg(loop, COL["neutral"]))
    out.append(text_lines(loop.cx, loop.cy - 4, "for each t",
                          ("in time_steps",), 21, 14, 17))

    partition = Rg("partition", 120, 76, 1400, 290)
    out.append(rect_svg(partition, "none", 18, COL["detail_border"], 1.2))
    out.append(label(partition.right - 520, partition.y - 8,
                     "_write_timestep_opf_state() - scenario_5_opf.py",
                     COL["text"], 15, 700, anchor="start"))

    pbound = Rg("pbound", 410, 94, 820, 50)
    draw_node(out, pbound, COL["green"], "[1] available P + active DERs",
              ("p_bound=max(profile,0); active if > eps",), 20, 14)
    out.append(direct(loop, "bottom", pbound, "top"))

    qlim = Rg("qlim", 410, 160, 820, 52)
    draw_node(out, qlim, COL["green"], "[2] compute Q capability",
              ("q_lim=min(Q_RATIO*sn, sqrt(sn^2-p^2))",), 20, 14)
    out.append(direct(pbound, "bottom", qlim, "top"))

    ders = Rg("ders", 410, 228, 820, 56)
    draw_node(out, ders, COL["green"], "[3] rebuild DER OPF state",
              ("reset all DERs to 0 / uncontrollable",
               "activate only p_bound>eps with P/Q bounds"), 19, 13)
    out.append(direct(qlim, "bottom", ders, "top"))

    costs = Rg("costs", 410, 300, 820, 50)
    draw_node(out, costs, COL["green"], "[4] write loads + rebuild costs",
              ("ext_grid +0.001; active DER -1.0",), 20, 14)
    out.append(direct(ders, "bottom", costs, "top"))

    dbg = Dg("debug", cx, 402, 330, 62)
    draw_decision(out, dbg, "debug requested for t?",
                  ("debug_opf_task + first-only gate",), 17, 12)
    out.append(direct(costs, "bottom", dbg, "top"))

    dbgbox = Rg("dbgbox", 1270, 373, 430, 58)
    draw_node(out, dbgbox, COL["neutral"], "_print_opf_debug()",
              ("opf_task + bounds checks",), 17, 12, 12)
    out.append(direct(dbg, "right", dbgbox, "left", "edge-gate"))
    out.append(label(dbg.right + 14, dbg.cy - 10, "yes",
                     COL["gate"], 15, 700, anchor="start"))

    mdbg = C(cx, 462, merge_r)
    out.append(direct(dbg, "bottom", mdbg, "top"))
    out.append(label(dbg.cx + 14, dbg.bottom + 20, "no",
                     COL["text"], 15, 700, anchor="start"))
    out.append(ortho_vh(dbgbox, "bottom", mdbg, "right", "edge-gate"))
    out.append(circle_svg(mdbg, COL["neutral"]))

    runopp = Rg("runopp", 410, 488, 820, 54)
    draw_node(out, runopp, COL["green"], "[5] pp.runopp()",
              ("init=opf_init; OPF_SOLVER=cyipopt",), 20, 14)
    out.append(direct(mdbg, "bottom", runopp, "top"))

    outcome = Dg("outcome", cx, 590, 350, 72)
    draw_decision(out, outcome, "runopp outcome?", ts=20)
    out.append(direct(runopp, "bottom", outcome, "top"))

    failed = Rg("failed", 72, 557, 280, 66)
    draw_node(out, failed, COL["red"], "OPF not converged",
              ("empty result series", "P/Q/target=None"), 17, 12, 12)
    out.append(direct(outcome, "left", failed, "right", "edge-hil"))
    out.append(label(outcome.left - 14, outcome.cy - 16, "caught",
                     COL["hil"], 14, 700, anchor="end"))

    abort = Rg("abort", 1320, 557, 330, 66)
    draw_node(out, abort, COL["red"], "Other exception",
              ("propagates; scenario aborts", "benchmark marks failed"), 17, 12, 12)
    out.append(direct(outcome, "right", abort, "left", "edge-hil"))
    out.append(label(outcome.right + 14, outcome.cy - 12, "uncaught",
                     COL["hil"], 14, 700, anchor="start"))

    success = Rg("success", 410, 652, 820, 80)
    draw_node(out, success, COL["blue"], "Converged optimum",
              ("read V / line / trafo / DER P,Q",
               "threshold violations; losses + grid import",
               "p_target=p_bound"), 18, 12)
    out.append(direct(outcome, "bottom", success, "top"))
    out.append(label(outcome.cx + 15, outcome.bottom + 22, "converged",
                     COL["text"], 15, 700, anchor="start"))

    mrec = C(cx, 762, merge_r)
    out.append(direct(success, "bottom", mrec, "top"))
    out.append(ortho_vh(failed, "bottom", mrec, "left", "edge-hil", bend_y=mrec.cy))
    out.append(circle_svg(mrec, COL["neutral"]))

    record = Rg("record", 330, 790, 980, 62)
    draw_node(out, record, COL["purple"], "[6] build TimestepRecord",
              ("append -> publish_fn.on_timestep()",
               "curtailment_needed = bool(recorded violations)"), 18, 12)
    out.append(direct(mrec, "bottom", record, "top"))

    periodic = Dg("periodic", cx, 894, 300, 54)
    draw_decision(out, periodic, "t % 96 == 0?", ts=18)
    out.append(direct(record, "bottom", periodic, "top"))

    progress = Rg("progress", 1340, 865, 420, 58)
    draw_node(out, progress, COL["purple"], "Periodic progress",
              ("optional live CSV partial result",
               "log active DER / violations / failures"), 16, 11, 12)
    out.append(direct(periodic, "right", progress, "left", "edge-gate"))
    out.append(label(periodic.right + 14, periodic.cy - 10, "yes",
                     COL["gate"], 15, 700, anchor="start"))

    mnext = C(cx, 954, merge_r)
    out.append(direct(periodic, "bottom", mnext, "top"))
    out.append(label(periodic.cx + 14, periodic.bottom + 20, "no",
                     COL["text"], 15, 700, anchor="start"))
    out.append(ortho_vh(progress, "bottom", mnext, "right", "edge-gate"))
    out.append(circle_svg(mnext, COL["neutral"]))

    more = Dg("more", 1230, 1004, 270, 50)
    draw_decision(out, more, "more timesteps?", ts=18)
    out.append(path([
        mnext.a("bottom"),
        (mnext.cx, more.cy),
        more.a("left", TARGET_GAP),
    ], "edge-dark"))

    result = Rg("result", 1545, 979, 320, 50)
    draw_node(out, result, COL["purple"], "from_records()", (), 18, 12, 10)
    out.append(direct(more, "right", result, "left"))
    out.append(label(more.right + 14, more.cy - 9, "no",
                     COL["text"], 15, 700, anchor="start"))

    # Same dedicated next-timestep loopback rail as IEEE.
    loopback_y = 1052
    out.append(path([
        more.a("bottom"),
        (more.cx, loopback_y),
        (loop_rail, loopback_y),
        (loop_rail, loop.cy),
        loop.a("left", TARGET_GAP),
    ], "edge-loop"))
    out.append(label(more.cx + 16, more.bottom + 20, "yes",
                     COL["loop"], 15, 700, anchor="start"))

    sep += [
        ("failed", "success", 20), ("success", "abort", 20),
        ("record", "progress", 10), ("progress", "result", 6),
    ]
    run_audit(W, H, rects, diamonds, sep)

    # Scaled-presentation coordinate audit.
    assert partition.top < pbound.top and partition.bottom > costs.bottom
    assert partition.bottom < dbg.top
    assert pbound.top - loop.bottom >= 26
    assert qlim.top - pbound.bottom >= 16
    assert ders.top - qlim.bottom >= 16
    assert costs.top - ders.bottom >= 16
    assert dbg.top - costs.bottom >= 20
    assert mdbg.top - dbg.bottom >= 14
    assert runopp.top - mdbg.bottom >= 11
    assert outcome.top - runopp.bottom >= 11
    assert success.top - outcome.bottom >= 25
    assert mrec.top - success.bottom >= 15
    assert record.top - mrec.bottom >= 13
    assert periodic.top - record.bottom >= 14
    assert mnext.top - periodic.bottom >= 18
    assert more.top - mnext.bottom >= 10
    assert loop_rail < failed.left - 20
    assert result.bottom < loopback_y - 12
    assert loopback_y < H - 12

    body = "\n".join(out)
    write("flow_s5_loop_presentation_final", W, H,
          "Scenario 5 per-timestep AC OPF flow - presentation scaled", body)
    write("flow_s5_loop_presentation_scaled", W, H,
          "Scenario 5 per-timestep AC OPF flow - presentation scaled", body)
    return W, H


if __name__ == "__main__":
    iw, ih = build_ieee()
    pw, ph = build_presentation()
    print(f"IEEE loop audited: {iw} x {ih}")
    print(f"Presentation loop audited: {pw} x {ph}")
    print(f"Outputs: {OUT}")
