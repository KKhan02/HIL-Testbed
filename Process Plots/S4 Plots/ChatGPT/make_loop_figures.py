"""
Scenario 4 - detailed per-timestep flow generator.

Outputs
-------
  flow_s4_loop_ieee_audited_final.{svg,pdf,png}
  flow_s4_loop_presentation_audited_final.{svg,pdf,png}

This version uses geometry objects and shape anchors for every execution
connection. Arrow endpoints are therefore derived from the actual rectangle,
diamond, and merge-circle coordinates instead of manually approximating them.

Logic follows the supplied PlantUML and the current project implementation:
  * scenario_4_volt_var.py
  * sensitivity_coordinator.py::run_coordinated_timestep
  * volt_var_controller.py::exchange_batched
  * der_dynamics.py::step

Important flow semantics
------------------------
  * diamonds = decisions only
  * circles = pure merge points
  * 4A and 4B are explicit branches after q_initial
  * early pre-PF failure and clean-timestep gate bypass control and converge
    only at the final record merge
  * curtailment is outside run_coordinated_timestep()
  * HIL serial failure falls back to local Q(V)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import html
import math
import cairosvg

OUT = Path(r"D:\My Files\Personal Projects\HIL-Testbed\Process Plots\S4 Plots\ChatGPT\s4_per_timestep")

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

# Marker tips land at the path endpoint. A small gap keeps the marker visible
# without burying the entire arrowhead underneath the target fill.
TARGET_GAP = 2.0


def esc(s):
    return html.escape(str(s))


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


def path(points, cls="edge-dark", marker=True):
    markers = {
        "edge-dark": "arrowDark",
        "edge-white": "arrowWhite",
        "edge-hil": "arrowHil",
        "edge-dry": "arrowDry",
        "edge-gate": "arrowGate",
        "edge-loop": "arrowLoop",
    }
    d = "M" + " L".join(f"{x:g} {y:g}" for x, y in points)
    mark = f' marker-end="url(#{markers[cls]})"' if marker else ""
    return f'<path d="{d}" class="{cls}"{mark}/>'

def path_bidir(points, cls="edge-dark"):
    """Path with arrowheads at both ends."""
    markers = {
        "edge-dark": "arrowDark",
        "edge-white": "arrowWhite",
        "edge-hil": "arrowHil",
        "edge-dry": "arrowDry",
        "edge-gate": "arrowGate",
        "edge-loop": "arrowLoop",
    }

    d = "M" + " L".join(f"{x:g} {y:g}" for x, y in points)
    marker = markers[cls]

    return (
        f'<path d="{d}" class="{cls}" '
        f'marker-start="url(#{marker})" '
        f'marker-end="url(#{marker})"/>'
    )

def direct(src, src_side, dst, dst_side, cls="edge-dark", gap=TARGET_GAP):
    """Straight connection when source and target anchors are collinear."""
    s = src.a(src_side)
    t = dst.a(dst_side, gap)
    return path([s, t], cls)

def direct_bidir(src, src_side, dst, dst_side,
                 cls="edge-dark", gap=TARGET_GAP):
    """Straight bidirectional connection between collinear anchors."""

    # Gap at BOTH ends because both ends now have arrowheads
    s = src.a(src_side, gap)
    t = dst.a(dst_side, gap)

    return path_bidir([s, t], cls)

def ortho_vh(src, src_side, dst, dst_side, cls="edge-dark", gap=TARGET_GAP,
             bend_y=None):
    """Leave source, move vertically, then horizontally into target."""
    s = src.a(src_side)
    t = dst.a(dst_side, gap)
    by = t[1] if bend_y is None else bend_y
    return path([s, (s[0], by), t], cls)


def ortho_hv(src, src_side, dst, dst_side, cls="edge-dark", gap=TARGET_GAP,
             bend_x=None):
    """Leave source, move horizontally, then vertically into target."""
    s = src.a(src_side)
    t = dst.a(dst_side, gap)
    bx = t[0] if bend_x is None else bend_x
    return path([s, (bx, s[1]), t], cls)


def header(w, h, title):
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">
<title>{esc(title)}</title>
<defs>
  <marker id="arrowDark" markerWidth="6" markerHeight="6" refX="8" refY="4.5" orient="auto-start-reverse" markerUnits="strokeWidth"><path d="M0,0 L9,4.5 L0,9 z" fill="#263238"/></marker>
  <marker id="arrowWhite" markerWidth="6" markerHeight="6" refX="8" refY="4.5" orient="auto-start-reverse" markerUnits="strokeWidth"><path d="M0,0 L9,4.5 L0,9 z" fill="#FFFFFF"/></marker>
  <marker id="arrowHil" markerWidth="6" markerHeight="6" refX="8" refY="4.5" orient="auto-start-reverse" markerUnits="strokeWidth"><path d="M0,0 L9,4.5 L0,9 z" fill="#D85A30"/></marker>
  <marker id="arrowDry" markerWidth="6" markerHeight="6" refX="8" refY="4.5" orient="auto-start-reverse" markerUnits="strokeWidth"><path d="M0,0 L9,4.5 L0,9 z" fill="#6B9E26"/></marker>
  <marker id="arrowGate" markerWidth="6" markerHeight="6" refX="8" refY="4.5" orient="auto-start-reverse" markerUnits="strokeWidth"><path d="M0,0 L9,4.5 L0,9 z" fill="#7A5C12"/></marker>
  <marker id="arrowLoop" markerWidth="6" markerHeight="6" refX="8" refY="4.5" orient="auto-start-reverse" markerUnits="strokeWidth"><path d="M0,0 L9,4.5 L0,9 z" fill="#3D8FD9"/></marker>
</defs>
<style>
text {{ font-family: Helvetica, Arial, sans-serif; }}
.node-title {{ font-weight:700; }}
.node-sub {{ font-weight:400; }}
.edge-dark {{ fill:none; stroke:#263238; stroke-width:2.6; stroke-linejoin:round; stroke-linecap:round; }}
.edge-white {{ fill:none; stroke:#FFFFFF; stroke-width:2.8; stroke-linejoin:round; stroke-linecap:round; }}
.edge-hil {{ fill:none; stroke:#D85A30; stroke-width:2.8; stroke-linejoin:round; stroke-linecap:round; }}
.edge-dry {{ fill:none; stroke:#6B9E26; stroke-width:2.8; stroke-linejoin:round; stroke-linecap:round; }}
.edge-gate {{ fill:none; stroke:#7A5C12; stroke-width:2.6; stroke-linejoin:round; stroke-linecap:round; }}
.edge-loop {{ fill:none; stroke:#3D8FD9; stroke-width:2.6; stroke-dasharray:7 5; stroke-linejoin:round; stroke-linecap:round; }}
</style>'''


def write(name, w, h, title, body):
    svg = header(w, h, title) + "\n" + body + "\n</svg>"
    (OUT / f"{name}.svg").write_text(svg, encoding="utf-8")
    cairosvg.svg2pdf(bytestring=svg.encode(), write_to=str(OUT / f"{name}.pdf"))
    cairosvg.svg2png(
        bytestring=svg.encode(),
        write_to=str(OUT / f"{name}.png"),
        output_width=w * 2,
        output_height=h * 2,
    )


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
    W = 820
    out = []
    cx = 330
    merge_r = 12
    bypass_x = 24

    rects = {}
    diamonds = {}
    sep = []

    def Rg(name, x, y, w, h):
        g = R(x, y, w, h); rects[name] = g; return g
    def Dg(name, x, y, w, h):
        g = D(x, y, w, h); diamonds[name] = g; return g

    out.append(label(W / 2, 34, "Scenario 4 - Per-Timestep Execution Loop", COL["text"], 17, 700))
    out.append('<g transform="translate(0,-10)">')

    loop = Dg("loop", cx, 88, 146, 50)
    out.append(diamond_svg(loop, COL["neutral"]))
    out.append(text_lines(loop.cx, loop.cy - 4, "for each t", ("in time_steps",), 14, 10, 14))

    ab = Rg("ab", 200, 128, 260, 54)
    draw_node(out, ab, COL["blue"], "[A][B] select p_target + write loads",
              ("reindex to sgen order ; load p_mw / q_mvar",), 13, 9)
    out.append(direct(loop, "bottom", ab, "top"))

    part = Rg("partition", 34, 198, 766, 904)
    out.append(rect_svg(part, "none", 16, COL["detail_border"], 1.0))
    out.append(label(part.x + 16, part.y - 5,
                     "run_coordinated_timestep() · sensitivity_coordinator.py",
                     COL["text"], 10, 700, anchor="start"))

    p0 = Rg("p0", 200, 220, 260, 48)
    draw_node(out, p0, COL["blue"], "[0][1] write P=p_target ; reset Q=0")
    out.append(direct(ab, "bottom", p0, "top"))

    pre = Rg("pre", 200, 286, 260, 50)
    draw_node(out, pre, COL["blue"], "[2] pre-PF -> report_pre",
              ("pp.runpp ; detect_violations",), 13, 9)
    out.append(direct(p0, "bottom", pre, "top"))

    dpre = Dg("dpre", cx, 376, 194, 60)
    draw_decision(out, dpre, "pre-PF converged?", ts=13)
    out.append(direct(pre, "bottom", dpre, "top"))

    early = Rg("early", 42, 347, 154, 58)
    draw_node(out, early, COL["red"], "early return",
              ("no dynamics.step", "q=0 ; p=p_target"), 11, 9, 10)
    out.append(direct(dpre, "left", early, "right", "edge-hil"))
    out.append(label(dpre.left - 8, dpre.cy - 9, "no", COL["hil"], 10, 700, anchor="end"))

    dviol = Dg("dviol", cx, 468, 208, 62)
    draw_decision(out, dviol, "report_pre has violations?", ts=12)
    out.append(direct(dpre, "bottom", dviol, "top", "edge-gate"))
    out.append(label(dpre.cx + 10, dpre.bottom + 18, "yes", COL["gate"], 10, 700, anchor="start"))

    clean = Rg("clean", 42, 437, 154, 62)
    draw_node(out, clean, COL["neutral"], "GATE: clean",
              ("dynamics.step(q=0)", "reuse report_pre"), 11, 9, 10)
    out.append(direct(dviol, "left", clean, "right", "edge-gate"))
    out.append(label(dviol.left - 8, dviol.cy - 9, "no", COL["gate"], 10, 700, anchor="end"))

    readvm = Rg("readvm", 200, 528, 260, 46)
    draw_node(out, readvm, COL["blue"], "[3] read vm_pu at DER buses", (), 13)
    out.append(direct(dviol, "bottom", readvm, "top"))
    out.append(label(dviol.cx + 10, dviol.bottom + 18, "yes", COL["text"], 10, 700, anchor="start"))

    dhil = Dg("dhil", cx, 616, 196, 58)
    draw_decision(out, dhil, "HIL interface active?", ts=12)
    out.append(direct(readvm, "bottom", dhil, "top"))

    hil = Rg("hil", 78, 662, 230, 58)
    dry = Rg("dry", 352, 662, 230, 58)
    draw_node(out, hil, COL["red"], "exchange_batched()",
              ("V:<vm> -> Arduino -> Q:<q>", "serial failure -> local Q(V)"), 12, 9, 10)
    draw_node(out, dry, COL["dry"], "QVCharacteristic",
              ("piecewise Q(V) -> q_initial",), 12, 9, 10)
    out.append(ortho_hv(dhil, "left", hil, "top", "edge-hil"))
    out.append(label(dhil.left - 8, dhil.cy - 8, "HIL", COL["hil"], 10, 700, anchor="end"))
    out.append(ortho_hv(dhil, "right", dry, "top", "edge-dry"))
    out.append(label(dhil.right + 8, dhil.cy - 8, "dry / fallback", COL["dry_edge"], 10, 700, anchor="start"))

    m1 = C(cx, 748, merge_r)
    out.append(ortho_vh(hil, "bottom", m1, "left", "edge-hil"))
    out.append(ortho_vh(dry, "bottom", m1, "right", "edge-dry"))
    out.append(circle_svg(m1, COL["neutral"]))
    out.append(label(m1.right + 18, m1.cy - 6, "q_initial", COL["text"], 10, 700, anchor="start"))

    dcoord = Dg("dcoord", cx, 804, 188, 58)
    draw_decision(out, dcoord, "coordination?", ts=12)
    out.append(direct(m1, "bottom", dcoord, "top"))

    a4 = Rg("a4", 78, 850, 230, 54)
    b4 = Rg("b4", 352, 850, 230, 54)
    draw_node(out, a4, COL["dry"], "4A: local Q(V)", ("q_adjusted = clip(q_initial)",), 12, 9, 10)
    draw_node(out, b4, COL["green"], "4B: coordinated", ("q_adjusted = coordinate(q_initial)",), 12, 9, 10)
    out.append(ortho_hv(dcoord, "left", a4, "top", "edge-dry"))
    out.append(label(dcoord.left - 8, dcoord.cy - 8, "no -> 4A", COL["dry_edge"], 10, 700, anchor="end"))
    out.append(ortho_hv(dcoord, "right", b4, "top", "edge-dark"))
    out.append(label(dcoord.right + 8, dcoord.cy - 8, "yes -> 4B", COL["text"], 10, 700, anchor="start"))

    coord_detail = Rg("coord_detail", 604, 838, 186, 78)
    draw_detail(out, coord_detail, COL["green"], "coordinate() - 4B",
                ("J blocks -> splu(J_PP)", "Schur J_red", "residual dV/dQ solve"), 10, 8)
    out.append(path([(b4.right, b4.cy - 6), (coord_detail.left - TARGET_GAP, coord_detail.cy - 6)], "edge-dark"))
    out.append(path([(coord_detail.left, coord_detail.cy + 6), (b4.right + TARGET_GAP, b4.cy + 6)], "edge-dark"))

    m2 = C(cx, 932, merge_r)
    out.append(ortho_vh(a4, "bottom", m2, "left", "edge-dry"))
    out.append(ortho_vh(b4, "bottom", m2, "right", "edge-dark"))
    out.append(circle_svg(m2, COL["neutral"]))
    out.append(label(m2.right + 18, m2.cy - 6, "q_adjusted", COL["text"], 10, 700, anchor="start"))

    dyn = Rg("dyn", 200, 964, 260, 50)
    draw_node(out, dyn, COL["blue"], "[5] DERDynamics.step",
              ("PT1 on Q ; symmetric P ramp",), 13, 9)
    out.append(direct(m2, "bottom", dyn, "top"))

    dyn_detail = Rg("dyn_detail", 604, 953, 186, 72)
    draw_detail(out, dyn_detail, COL["green"], "DER dynamics",
                ("alpha=1-exp(-dt/tau)", "dP_max=rate*p_rated*dt"), 10, 8)
    out.append(path([(dyn.right, dyn.cy - 6), (dyn_detail.left - TARGET_GAP, dyn_detail.cy - 6)], "edge-dark"))
    out.append(path([(dyn_detail.left, dyn_detail.cy + 6), (dyn.right + TARGET_GAP, dyn.cy + 6)], "edge-dark"))

    applypf = Rg("applypf", 200, 1032, 260, 58)
    draw_node(out, applypf, COL["blue"], "[6][7][8] apply + post-PF",
              ("write p_applied first ; clamp q_applied",
               "runpp -> report_post ; set curtailment_needed"), 12, 9)
    out.append(direct(dyn, "bottom", applypf, "top"))

    dcurt = Dg("dcurt", cx, 1142, 278, 70)
    draw_decision(out, dcurt, "curtailment_needed", ("and converged?",), 12, 10)
    out.append(direct(applypf, "bottom", dcurt, "top"))

    curt = Rg("curt", 500, 1096, 290, 92)
    draw_detail(out, curt, COL["red"], "[E] curtailment sub-loop",
                ("p_curtailed=max(p_target-k*step,0)",
                 "runpp + detect_violations, max 10 iterations",
                 "success: dynamics.p_prev=p_curtailed"), 11, 9)
    out.append(direct(dcurt, "right", curt, "left", "edge-hil"))
    out.append(label(dcurt.right + 8, dcurt.cy - 8, "yes", COL["hil"], 10, 700, anchor="start"))

    mrec = C(cx, 1220, merge_r)
    out.append(direct(dcurt, "bottom", mrec, "top"))
    out.append(label(dcurt.cx + 10, dcurt.bottom + 17, "no", COL["text"], 10, 700, anchor="start"))
    out.append(ortho_vh(curt, "bottom", mrec, "right", "edge-hil"))

    # Dedicated bypass rail. Branches meet the rail without arrowheads;
    # a single arrow enters the record merge from the left.
    out.append(path([early.a("left"), (bypass_x, early.cy)], "edge-gate", marker=False))
    out.append(path([clean.a("left"), (bypass_x, clean.cy)], "edge-gate", marker=False))
    out.append(path([(bypass_x, early.cy), (bypass_x, mrec.cy)], "edge-gate", marker=False))
    out.append(path([(bypass_x, mrec.cy), mrec.a("left", TARGET_GAP)], "edge-gate"))
    out.append(circle_svg(mrec, COL["neutral"]))

    record = Rg("record", 200, 1252, 260, 58)
    draw_node(out, record, COL["purple"], "[D][F] build TimestepRecord",
              ("publish_fn.on_timestep ; records.append", "checkpoint if t % 96 == 0"), 12, 9)
    out.append(direct(mrec, "bottom", record, "top"))

    dmore = Dg("dmore", cx, 1350, 176, 54)
    draw_decision(out, dmore, "more timesteps?", ts=12)
    out.append(direct(record, "bottom", dmore, "top"))

    result = Rg("result", 200, 1392, 260, 42)
    draw_node(out, result, COL["green"], "ScenarioResult.from_records()", (), 13)
    out.append(direct(dmore, "bottom", result, "top"))

    # loopback from left point of more-timesteps decision to a far-left rail,
    # then up to the left point of the initial loop diamond.
    loop_rail = 44
    out.append(path([dmore.a("left"), (loop_rail, dmore.cy),
                     (loop_rail, loop.cy), loop.a("left", TARGET_GAP)], "edge-loop"))
    out.append(label(dmore.left - 8, dmore.cy - 8, "yes", COL["loop"], 10, 700, anchor="end"))

    H = 1464

    sep += [
        ("early", "dpre", 8), ("clean", "dviol", 8),
        ("hil", "dry", 8), ("a4", "b4", 8),
        ("curt", "record", 8),
    ]
    
    # Only rectangular geometry is audited for separation. Decision diamonds
    # are already checked by anchor construction and visual row spacing.
    rect_sep = [(a, b, p) for a, b, p in sep if a in rects and b in rects]
    run_audit(W, H, rects, diamonds, rect_sep)
    out.append('</g>')
    body = [rect_svg(R(14, 14, W - 28, H - 28), COL["panel"], 16, COL["panel_border"], 1.5)] + out
    write("flow_s4_loop_ieee_audited_final", W, H,
          "Scenario 4 per-timestep loop - IEEE audited", "\n".join(body))
    return W, H

# ======================================================================
# PRESENTATION FIGURE - 1920 x 1080
# ======================================================================

def build_presentation():
    W, H = 1920, 1080
    out = [rect_svg(R(0, 0, W, H), COL["panel"], 0)]
    rects = {}
    diamonds = {}
    sep = []

    def Rg(name, x, y, w, h):
        g = R(x, y, w, h)
        rects[name] = g
        return g

    def Dg(name, x, y, w, h):
        g = D(x, y, w, h)
        diamonds[name] = g
        return g

    # ------------------------------------------------------------------
    # Main execution flow.
    # No title/subtitle here because the PowerPoint slide supplies the title.
    # Vertical gaps on the central spine are deliberately kept large enough
    # that the path shaft remains visible in front of the SVG marker head.
    # ------------------------------------------------------------------
    cx = 430
    merge_r = 14

    # Separate rails: gate/early bypass is NOT the next-timestep loopback.
    bypass_x = 44
    loop_rail = 18

    loop = Dg("loop", cx, 18, 150, 32)
    out.append(diamond_svg(loop, COL["neutral"]))
    out.append(text_lines(loop.cx, loop.cy - 1, "for each t", ("in time_steps",), 14, 9, 13))

    ab = Rg("ab", 180, 50, 500, 36)
    draw_node(out, ab, COL["blue"], "[A][B] p_target + load profiles",
              ("reindex DER P ; write load p_mw / q_mvar",), 13, 9)
    out.append(direct(loop, "bottom", ab, "top"))

    partition = Rg("partition", 62, 96, 1024, 780)
    out.append(rect_svg(partition, "#FFFFFF", 18, COL["detail_border"], 1.2))
    out.append(label(partition.right - 395, partition.y - 5,
                     "[C] run_coordinated_timestep() · sensitivity_coordinator.py",
                     COL["text"], 13, 700, anchor="start"))

    p0 = Rg("p0", 180, 100, 500, 36)
    draw_node(out, p0, COL["blue"], "[0][1] write P=p_target ; reset Q=0", (), 14)
    out.append(direct(ab, "bottom", p0, "top"))

    pre = Rg("pre", 180, 158, 500, 38)
    draw_node(out, pre, COL["blue"], "[2] pre-PF -> report_pre",
              ("runpp ; detect_violations",), 14, 9)
    out.append(direct(p0, "bottom", pre, "top"))

    dpre = Dg("dpre", cx, 242, 230, 48)
    draw_decision(out, dpre, "pre-PF converged?", ts=13)
    out.append(direct(pre, "bottom", dpre, "top"))

    early = Rg("early", 64, 214, 210, 56)
    draw_node(out, early, COL["red"], "early return",
              ("no dynamics.step ; q=0 ; p=p_target",), 12, 9, 10)
    out.append(direct(dpre, "left", early, "right", "edge-hil"))
    out.append(label(dpre.left - 10, dpre.cy - 8, "no", COL["hil"], 11, 700, anchor="end"))

    dviol = Dg("dviol", cx, 312, 250, 48)
    draw_decision(out, dviol, "report_pre has violations?", ts=13)
    out.append(direct(dpre, "bottom", dviol, "top", "edge-gate"))
    out.append(label(dpre.cx + 12, dpre.bottom + 16, "yes", COL["gate"], 11, 700, anchor="start"))

    clean = Rg("clean", 64, 284, 210, 56)
    draw_node(out, clean, COL["neutral"], "GATE: clean timestep",
              ("dynamics.step(q=0) ; reuse report_pre",), 12, 9, 10)
    out.append(direct(dviol, "left", clean, "right", "edge-gate"))
    out.append(label(dviol.left - 10, dviol.cy - 8, "no", COL["gate"], 11, 700, anchor="end"))

    readvm = Rg("readvm", 180, 358, 500, 36)
    draw_node(out, readvm, COL["blue"], "[3] read vm_pu at DER buses", (), 14)
    out.append(direct(dviol, "bottom", readvm, "top"))
    out.append(label(dviol.cx + 12, dviol.bottom + 16, "yes", COL["text"], 11, 700, anchor="start"))

    dhil = Dg("dhil", cx, 435, 240, 46)
    draw_decision(out, dhil, "HIL interface active?", ts=13)
    out.append(direct(readvm, "bottom", dhil, "top"))

    hil = Rg("hil", 100, 474, 300, 48)
    dry = Rg("dry", 570, 474, 300, 48)
    draw_node(out, hil, COL["red"], "exchange_batched() -> q_initial",
              ("serial failure -> local Q(V)",), 13, 9, 10)
    draw_node(out, dry, COL["dry"], "QVCharacteristic -> q_initial",
              ("piecewise Q(V)",), 13, 9, 10)
    out.append(ortho_hv(dhil, "left", hil, "top", "edge-hil"))
    out.append(label(dhil.left - 10, dhil.cy - 8, "HIL", COL["hil"], 11, 700, anchor="end"))
    out.append(ortho_hv(dhil, "right", dry, "top", "edge-dry"))
    out.append(label(dhil.right + 10, dhil.cy - 8, "dry / fallback", COL["dry_edge"], 11, 700, anchor="start"))

    m1 = C(cx, 550, merge_r)
    out.append(ortho_vh(hil, "bottom", m1, "left", "edge-hil"))
    out.append(ortho_vh(dry, "bottom", m1, "right", "edge-dry"))
    out.append(circle_svg(m1, COL["neutral"]))
    out.append(label(m1.right + 18, m1.cy - 7, "q_initial", COL["text"], 11, 700, anchor="start"))

    dcoord = Dg("dcoord", cx, 609, 206, 46)
    draw_decision(out, dcoord, "coordination?", ts=13)
    out.append(direct(m1, "bottom", dcoord, "top"))

    a4 = Rg("a4", 100, 646, 300, 48)
    b4 = Rg("b4", 570, 646, 300, 48)
    draw_node(out, a4, COL["dry"], "4A - local Q(V)",
              ("q_adjusted = clip(q_initial, +/-q_max)",), 13, 9, 10)
    draw_node(out, b4, COL["green"], "4B - coordinated Q(V)",
              ("q_adjusted = coordinate(q_initial)",), 13, 9, 10)
    out.append(ortho_hv(dcoord, "left", a4, "top", "edge-dry"))
    out.append(label(dcoord.left - 10, dcoord.cy - 8, "no -> 4A", COL["dry_edge"], 11, 700, anchor="end"))
    out.append(ortho_hv(dcoord, "right", b4, "top", "edge-dark"))
    out.append(label(dcoord.right + 10, dcoord.cy - 8, "yes -> 4B", COL["text"], 11, 700, anchor="start"))

    m2 = C(cx, 724, merge_r)
    out.append(ortho_vh(a4, "bottom", m2, "left", "edge-dry"))
    out.append(ortho_vh(b4, "bottom", m2, "right", "edge-dark"))
    out.append(circle_svg(m2, COL["neutral"]))
    out.append(label(m2.right + 18, m2.cy - 7, "q_adjusted", COL["text"], 11, 700, anchor="start"))

    dyn = Rg("dyn", 180, 760, 500, 38)
    draw_node(out, dyn, COL["blue"], "[5] DERDynamics.step",
              ("PT1 on Q ; symmetric P ramp",), 14, 9)
    out.append(direct(m2, "bottom", dyn, "top"))

    applypf = Rg("applypf", 180, 820, 500, 44)
    draw_node(out, applypf, COL["blue"], "[6][7][8] apply setpoints + post-PF",
              ("write p_applied ; clamp q ; report_post ; curtailment flag",), 13, 9)
    out.append(direct(dyn, "bottom", applypf, "top"))

    # ------------------------------------------------------------------
    # Outer runner. It begins after the [C] partition.
    # ------------------------------------------------------------------
    dcurt = Dg("dcurt", cx, 912, 300, 52)
    draw_decision(out, dcurt, "curtailment_needed", ("and converged?",), 13, 10)
    out.append(direct(applypf, "bottom", dcurt, "top"))

    curt = Rg("curt", 650, 882, 360, 60)
    draw_node(out, curt, COL["red"], "[E] curtailment sub-loop",
              ("-10% P / iter ; runpp ; detect ; max 10", "success: update dynamics.p_prev once"), 12, 9, 10)
    out.append(direct(dcurt, "right", curt, "left", "edge-hil"))
    out.append(label(dcurt.right + 10, dcurt.cy - 8, "yes", COL["hil"], 11, 700, anchor="start"))

    mrec = C(cx, 974, merge_r)
    out.append(direct(dcurt, "bottom", mrec, "top"))
    out.append(label(dcurt.cx + 12, dcurt.bottom + 17, "no", COL["text"], 11, 700, anchor="start"))
    out.append(ortho_vh(curt, "bottom", mrec, "right", "edge-hil"))

    # Early-return and clean-timestep paths share a dedicated gate/bypass rail.
    # The blue next-timestep loopback uses a different x-coordinate below.
    out.append(path([early.a("left"), (bypass_x, early.cy)], "edge-gate", marker=False))
    out.append(path([clean.a("left"), (bypass_x, clean.cy)], "edge-gate", marker=False))
    out.append(path([(bypass_x, early.cy), (bypass_x, mrec.cy)], "edge-gate", marker=False))
    out.append(path([(bypass_x, mrec.cy), mrec.a("left", TARGET_GAP)], "edge-gate"))
    out.append(circle_svg(mrec, COL["neutral"]))

    record = Rg("record", 200, 1010, 460, 40)
    draw_node(out, record, COL["purple"], "[D][F] build TimestepRecord",
              ("publish ; append ; checkpoint if t % 96 == 0",), 13, 9, 10)
    out.append(direct(mrec, "bottom", record, "top"))

    dmore = Dg("dmore", 780, 1030, 180, 40)
    draw_decision(out, dmore, "more timesteps?", ts=13)
    out.append(direct(record, "right", dmore, "left"))

    result = Rg("result", 890, 1010, 220, 40)
    draw_node(out, result, COL["green"], "ScenarioResult.from_records()", (), 12, 9, 10)
    out.append(direct(dmore, "right", result, "left"))

    # Separate next-timestep loopback rail. It never shares the gate/bypass rail.
    loopback_y = 1070
    out.append(path([dmore.a("bottom"), (dmore.cx, loopback_y),
                     (loop_rail, loopback_y), (loop_rail, loop.cy),
                     loop.a("left", TARGET_GAP)], "edge-loop"))
    out.append(label(dmore.cx + 12, dmore.bottom + 16, "yes", COL["loop"], 11, 700, anchor="start"))

    # ------------------------------------------------------------------
    # Right-side explanatory panels.
    # Space is used for implementation detail rather than a large empty block.
    # ------------------------------------------------------------------
    dx = 1160
    dw = 720
    gap = 20
    cw = (dw - gap) / 2
    c1 = dx
    c2 = dx + cw + gap

    p1 = Rg("panel_hil", c1, 20, cw, 300)
    p2 = Rg("panel_mode", c2, 20, cw, 300)
    p3 = Rg("panel_dyn", c1, 340, cw, 300)
    p4 = Rg("panel_outer", c2, 340, cw, 300)
    p5 = Rg("panel_inv", dx, 660, dw, 360)

    for p in (p1, p2, p3, p4, p5):
        out.append(rect_svg(p, "#FFFFFF", 16, COL["detail_border"], 1.2))

    # Item 2
    out.append(label(p1.cx, p1.y + 26, "Item 2 - HIL / dry execution", COL["text"], 15, 700))
    d11 = R(p1.x + 16, p1.y + 46, p1.w - 32, 62)
    d12 = R(p1.x + 16, p1.y + 118, p1.w - 32, 62)
    d13 = R(p1.x + 16, p1.y + 190, p1.w - 32, 62)
    draw_detail(out, d11, COL["red"], "exchange_batched()",
                ("DER buses batched to ARDUINO_MAX_DERS", "V:<vm> sent at fixed serial precision"), 12, 9)
    draw_detail(out, d12, COL["red"], "Arduino firmware",
                ("parse V -> compute_q() in float", "emit Q:<q> for the same DER order"), 12, 9)
    draw_detail(out, d13, COL["dry"], "dry / serial fallback",
                ("QVCharacteristic.compute_setpoints()", "used for dry-run or serial/protocol failure"), 12, 9)
    out.append(direct(d11, "bottom", d12, "top", "edge-hil"))
    out.append(label(p1.cx, p1.bottom - 18,
                     "Both routes return the same q_initial vector interface",
                     COL["text"], 9, 600))

    # Item 3
    out.append(label(p2.cx, p2.y + 26, "Item 3 - 4A versus 4B", COL["text"], 15, 700))
    d21 = R(p2.x + 16, p2.y + 46, p2.w - 32, 62)
    d22 = R(p2.x + 16, p2.y + 118, p2.w - 32, 62)
    d23 = R(p2.x + 16, p2.y + 190, p2.w - 32, 72)
    draw_detail(out, d21, COL["dry"], "4A - local Q(V)",
                ("coordination=False", "clip q_initial directly to +/-q_max"), 12, 9)
    draw_detail(out, d22, COL["green"], "4B - coordinated Q(V)",
                ("coordination=True", "coordinate(q_initial) before dynamics"), 12, 9)
    draw_detail(out, d23, COL["green"], "coordinate() internals",
                ("J blocks -> splu(J_PP)", "Schur J_red -> residual dV/dQ solve", "final q_adjusted clipped to +/-q_max"), 11, 8)
    out.append(label(p2.cx, p2.bottom - 16,
                     "4A bypasses coordinate(); both modes rejoin at q_adjusted",
                     COL["text"], 9, 600))

    # Items 4-8
    out.append(label(p3.cx, p3.y + 26, "Items 4-8 - physical response", COL["text"], 15, 700))
    d31 = R(p3.x + 16, p3.y + 46, p3.w - 32, 68)
    d32 = R(p3.x + 16, p3.y + 124, p3.w - 32, 68)
    d33 = R(p3.x + 16, p3.y + 202, p3.w - 32, 68)
    draw_detail(out, d31, COL["green"], "[5] DERDynamics.step",
                ("Q: exact discrete PT1, tau=t95/3", "P: symmetric ramp limited by rated power"), 12, 9)
    draw_detail(out, d32, COL["blue"], "[6] apply setpoints",
                ("write p_applied first", "clamp q_applied to network limits"), 12, 9)
    draw_detail(out, d33, COL["blue"], "[7][8] post-PF + trigger",
                ("runpp -> report_post", "residual violations set curtailment_needed"), 12, 9)

    # Outer runner
    out.append(label(p4.cx, p4.y + 26, "Outer runner - curtailment + record", COL["text"], 15, 700))
    d41 = R(p4.x + 16, p4.y + 46, p4.w - 32, 68)
    d42 = R(p4.x + 16, p4.y + 124, p4.w - 32, 68)
    d43 = R(p4.x + 16, p4.y + 202, p4.w - 32, 68)
    draw_detail(out, d41, COL["red"], "Curtailment trigger",
                ("only after converged post-PF", "and residual voltage violations remain"), 12, 9)
    draw_detail(out, d42, COL["red"], "[E] active-P sub-loop",
                ("-10% of original P target per iteration", "Q is held; no dynamics.step in the sub-loop"), 12, 9)
    draw_detail(out, d43, COL["purple"], "[D][F] record + publish",
                ("record final post-curtailment/post-Q state", "publish -> append -> periodic checkpoint"), 12, 9)

    # Invariants + state/legend. Two information-dense columns avoid dead space.
    out.append(label(p5.cx, p5.y + 28, "Execution invariants, state semantics and legend", COL["text"], 15, 700))

    left_x = p5.x + 24
    right_x = p5.x + 382

    out.append(label(left_x, p5.y + 58, "Execution invariants", COL["text"], 11, 700, anchor="start"))
    left_lines = [
        "1. pre-PF always evaluates p_target with q=0.",
        "2. clean pre-PF skips Q(V), coordinate() and post-PF.",
        "3. failed pre-PF returns before DERDynamics state advances.",
        "4. 4A and 4B share q_initial acquisition and DERDynamics.",
        "5. q is clamped only after p_applied is written to the net.",
        "6. post-PF is the authoritative curtailment trigger.",
        "7. curtailment changes P only; Q remains at the applied value.",
        "8. p_prev is updated once after successful curtailment.",
    ]
    for i, txt in enumerate(left_lines):
        out.append(label(left_x, p5.y + 84 + 25 * i, txt,
                         COL["text"], 9.5, 600, anchor="start"))

    out.append(label(right_x, p5.y + 58, "State / branch semantics", COL["text"], 11, 700, anchor="start"))
    state_lines = [
        "q_initial: raw local/HIL Q(V) setpoint",
        "q_adjusted: 4A clip or 4B coordinated setpoint",
        "q_applied / p_applied: dynamic actuator outputs",
        "report_pre: gate decision before Q control",
        "report_post: final Q-control voltage assessment",
        "record: final state after optional active-P curtailment",
    ]
    for i, txt in enumerate(state_lines):
        out.append(label(right_x, p5.y + 84 + 25 * i, txt,
                         COL["text"], 9.5, 600, anchor="start"))

    out.append(label(right_x, p5.y + 244, "Legend", COL["text"], 11, 700, anchor="start"))
    legend_rows = [
        ("edge-dark", "execution", COL["text"]),
        ("edge-hil", "HIL / curtailment branch", COL["hil"]),
        ("edge-dry", "dry / 4A branch", COL["dry_edge"]),
        ("edge-gate", "gate / record bypass", COL["gate"]),
        ("edge-loop", "next-timestep return", COL["loop"]),
    ]
    for i, (cls, txt, color) in enumerate(legend_rows):
        yy = p5.y + 266 + 19 * i
        out.append(path([(right_x, yy), (right_x + 52, yy)], cls))
        out.append(label(right_x + 64, yy + 4, txt, color, 9.5, 600, anchor="start"))

    # ---------------- coordinate / placement audit ----------------
    sep += [
        ("early", "dpre", 8), ("clean", "dviol", 8),
        ("hil", "dry", 20), ("a4", "b4", 20),
        ("record", "result", 20),
        ("panel_hil", "panel_mode", 16),
        ("panel_dyn", "panel_outer", 16),
        ("panel_hil", "panel_dyn", 16),
        ("panel_mode", "panel_outer", 16),
        ("panel_dyn", "panel_inv", 16),
        ("panel_outer", "panel_inv", 16),
    ]
    rect_sep = [(a, b, p) for a, b, p in sep if a in rects and b in rects]
    run_audit(W, H, rects, diamonds, rect_sep)

    # Explicit shaft-clearance checks for the vertical links that previously
    # collapsed visually into marker heads.
    assert pre.top - p0.bottom >= 22
    assert dpre.top - pre.bottom >= 22
    assert readvm.top - dviol.bottom >= 22
    assert dhil.top - readvm.bottom >= 18
    assert dcoord.top - m1.bottom >= 22
    assert dyn.top - m2.bottom >= 22
    assert applypf.top - dyn.bottom >= 22
    assert dcurt.top - applypf.bottom >= 22
    assert mrec.top - dcurt.bottom >= 22
    assert record.top - mrec.bottom >= 22

    # Structural checks.
    assert partition.bottom > applypf.bottom
    assert partition.bottom < dcurt.top
    assert early.right < dpre.left
    assert clean.right < dviol.left
    assert loop_rail < bypass_x < early.left
    assert record.bottom < loopback_y - 12
    assert result.bottom < loopback_y - 12
    assert loopback_y < H - 6

    write("flow_s4_loop_presentation_rebalanced", W, H,
          "Scenario 4 per-timestep loop - presentation rebalanced", "\n".join(out))
    return W, H



if __name__ == "__main__":
    iw, ih = build_ieee()
    pw, ph = build_presentation()
    print(f"IEEE audited: {iw} x {ih}  ratio={iw/ih:.3f}")
    print(f"Presentation audited: {pw} x {ph}  ratio={pw/ph:.3f}")
    print("Coordinate audit: PASS")