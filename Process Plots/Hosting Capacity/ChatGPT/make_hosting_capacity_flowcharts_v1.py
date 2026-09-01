from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import html
import xml.etree.ElementTree as ET

import cairosvg

OUT = Path(__file__).resolve().parent / "hosting_capacity_flowcharts_v1"
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
    "hil": "#D85A30",
    "gate": "#7A5C12",
    "loop": "#3D8FD9",
    "assoc": "#8A887F",
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


def rect_svg(g: R, fill, rx=12, stroke="none", sw=0):
    return (
        f'<rect x="{g.x}" y="{g.y}" width="{g.w}" height="{g.h}" rx="{rx}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>'
    )


def diamond_svg(g: D, fill):
    pts = f"{g.cx},{g.top} {g.right},{g.cy} {g.cx},{g.bottom} {g.left},{g.cy}"
    return f'<polygon points="{pts}" fill="{fill}"/>'


def circle_svg(g: C, fill):
    return f'<circle cx="{g.cx}" cy="{g.cy}" r="{g.r}" fill="{fill}"/>'


def label(x, y, text, fill, size=13, weight=700, anchor="middle"):
    return (
        f'<text x="{x}" y="{y}" text-anchor="{anchor}" fill="{fill}" '
        f'font-size="{size}" font-weight="{weight}">{esc(text)}</text>'
    )


def text_lines_centered(x, cy, title, subs=(), title_size=14, sub_size=10,
                        line_gap=16, fill="#FFFFFF", anchor="middle"):
    lines = [(title, title_size, 700)] + [(s, sub_size, 400) for s in subs]
    total = (len(lines) - 1) * line_gap
    first = cy - total / 2 + title_size * 0.34
    out = []
    for i, (txt, size, weight) in enumerate(lines):
        cls = "node-title" if weight >= 700 else "node-sub"
        opacity = "" if weight >= 700 else ' opacity="0.92"'
        out.append(
            f'<text x="{x}" y="{first + i * line_gap}" text-anchor="{anchor}" '
            f'class="{cls}" font-size="{size}" fill="{fill}"{opacity}>{esc(txt)}</text>'
        )
    return "\n".join(out)


def _clean_points(points):
    cleaned = []
    for p in points:
        p = (float(p[0]), float(p[1]))
        if not cleaned or p != cleaned[-1]:
            cleaned.append(p)
    assert len(cleaned) >= 2
    for a, b in zip(cleaned, cleaned[1:]):
        dx = abs(a[0] - b[0])
        dy = abs(a[1] - b[1])
        assert dx < 1e-9 or dy < 1e-9, f"non-orthogonal segment: {a}->{b}"
        assert dx + dy > 0.5, f"zero/near-zero segment: {a}->{b}"
    return cleaned


def path(points, cls="edge-dark", marker=True):
    pts = _clean_points(points)
    markers = {
        "edge-dark": "arrowDark",
        "edge-hil": "arrowHil",
        "edge-gate": "arrowGate",
        "edge-loop": "arrowLoop",
        "edge-assoc": "arrowAssoc",
        "edge-dry": "arrowDry",
    }
    d = "M" + " L".join(f"{x:g} {y:g}" for x, y in pts)
    m = f' marker-end="url(#{markers[cls]})"' if marker else ""
    return f'<path d="{d}" class="{cls}"{m}/>'


def direct(src, ss, dst, ds, cls="edge-dark", gap=TARGET_GAP):
    return path([src.a(ss), dst.a(ds, gap)], cls)


def header(w, h, title):
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">
<title>{esc(title)}</title>
<defs>
<marker id="arrowDark" markerWidth="6" markerHeight="6" refX="8" refY="4.5" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L9,4.5 L0,9 z" fill="#263238"/></marker>
<marker id="arrowHil" markerWidth="6" markerHeight="6" refX="8" refY="4.5" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L9,4.5 L0,9 z" fill="#D85A30"/></marker>
<marker id="arrowGate" markerWidth="6" markerHeight="6" refX="8" refY="4.5" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L9,4.5 L0,9 z" fill="#7A5C12"/></marker>
<marker id="arrowLoop" markerWidth="6" markerHeight="6" refX="8" refY="4.5" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L9,4.5 L0,9 z" fill="#3D8FD9"/></marker>
<marker id="arrowAssoc" markerWidth="6" markerHeight="6" refX="8" refY="4.5" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L9,4.5 L0,9 z" fill="#8A887F"/></marker>
<marker id="arrowDry" markerWidth="6" markerHeight="6" refX="8" refY="4.5" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L9,4.5 L0,9 z" fill="#6B9E26"/></marker>
</defs>
<style>
text {{font-family:Helvetica,Arial,sans-serif}}
.node-title {{font-weight:700}}
.node-sub {{font-weight:400}}
.edge-dark {{fill:none;stroke:#263238;stroke-width:2.6;stroke-linejoin:round;stroke-linecap:round}}
.edge-hil {{fill:none;stroke:#D85A30;stroke-width:2.8;stroke-linejoin:round;stroke-linecap:round}}
.edge-gate {{fill:none;stroke:#7A5C12;stroke-width:2.6;stroke-linejoin:round;stroke-linecap:round}}
.edge-loop {{fill:none;stroke:#3D8FD9;stroke-width:2.6;stroke-dasharray:7 5;stroke-linejoin:round;stroke-linecap:round}}
.edge-assoc {{fill:none;stroke:#8A887F;stroke-width:2;stroke-dasharray:5 4;stroke-linejoin:round;stroke-linecap:round}}
.edge-dry {{fill:none;stroke:#6B9E26;stroke-width:2.6;stroke-linejoin:round;stroke-linecap:round}}
</style>'''


def write(name, w, h, title, body):
    svg = header(w, h, title) + "\n" + body + "\n</svg>"
    sp = OUT / f"{name}.svg"
    pp = OUT / f"{name}.pdf"
    pn = OUT / f"{name}.png"
    sp.write_text(svg, encoding="utf-8")
    cairosvg.svg2pdf(bytestring=svg.encode(), write_to=str(pp))
    cairosvg.svg2png(bytestring=svg.encode(), write_to=str(pn), output_width=w*2, output_height=h*2)
    ET.parse(sp)
    assert min(sp.stat().st_size, pp.stat().st_size, pn.stat().st_size) > 1000


def draw_node(out, g, fill, title, subs=(), ts=14, ss=10, rx=12):
    out += [rect_svg(g, fill, rx), text_lines_centered(g.cx, g.cy, title, subs, ts, ss, 16)]


def draw_decision(out, g, title, subs=(), ts=13, ss=9.5, fill=None):
    out += [diamond_svg(g, fill or COL["decision"]), text_lines_centered(g.cx, g.cy, title, subs, ts, ss, 14)]


def branch_label(out, x, y, text, fill=COL["text"], anchor="middle", size=9.5):
    out.append(label(x, y, text, fill, size, 700, anchor))


def _bbox(g):
    return (g.left, g.top, g.right, g.bottom)


def _interiors_overlap(a, b, margin=0.5):
    al, at, ar, ab = _bbox(a)
    bl, bt, br, bb = _bbox(b)
    return (min(ar, br) - max(al, bl) > margin and min(ab, bb) - max(at, bt) > margin)


def audit_node_overlaps(nodes, allowed=()):
    allowed = {frozenset(p) for p in allowed}
    items = list(nodes.items())
    for i, (na, ga) in enumerate(items):
        for nb, gb in items[i+1:]:
            if frozenset((na, nb)) in allowed:
                continue
            assert not _interiors_overlap(ga, gb), f"node overlap: {na} vs {nb}"


def audit_bounds(nodes, W, H, margin=0):
    for n, g in nodes.items():
        assert g.left >= margin and g.top >= margin and g.right <= W-margin and g.bottom <= H-margin, f"{n}: bounds"


def seg_hits_bbox(a, b, g, margin=1.5):
    l, t, r, bot = _bbox(g)
    l += margin; t += margin; r -= margin; bot -= margin
    if l >= r or t >= bot:
        return False
    if abs(a[0]-b[0]) < 1e-9:
        x = a[0]
        y0, y1 = sorted((a[1], b[1]))
        return l < x < r and max(y0, t) < min(y1, bot)
    y = a[1]
    x0, x1 = sorted((a[0], b[0]))
    return t < y < bot and max(x0, l) < min(x1, r)


def audit_routes(routes, nodes):
    for name, points, ignore in routes:
        pts = _clean_points(points)
        for a, b in zip(pts, pts[1:]):
            for nn, g in nodes.items():
                if nn in ignore:
                    continue
                assert not seg_hits_bbox(a, b, g), f"route {name} intersects node {nn}: {a}->{b}"


def add_route(out, routes, name, points, cls="edge-dark", marker=True, ignore=()):
    pts = [(float(p[0]), float(p[1])) for p in points]
    if len(pts) == 2 and pts[0][0] != pts[1][0] and pts[0][1] != pts[1][1]:
        pts = [pts[0], (pts[1][0], pts[0][1]), pts[1]]
    pts = _clean_points(pts)
    routes.append((name, pts, set(ignore)))
    out.append(path(pts, cls, marker))



# =============================================================================
# Hosting capacity flowcharts
# Source of truth: hosting_capacity.py
# =============================================================================


def build_baseline_ieee():
    W, H = 1120, 1210

    out = [
        rect_svg(
            R(14, 14, W - 28, H - 28),
            COL["panel"],
            16,
            COL["panel_border"],
            1.5,
        )
    ]

    nodes, routes = {}, []

    def Rg(n, x, y, w, h):
        g = R(x, y, w, h)
        nodes[n] = g
        return g

    def Dg(n, x, y, w, h):
        g = D(x, y, w, h)
        nodes[n] = g
        return g

    def Cg(n, x, y, r):
        g = C(x, y, r)
        nodes[n] = g
        return g

    cx = 590

    # Dedicated terminal / loop rails.
    pf_fail_rail_x = 30
    outer_loop_x = 1060

    out.append(
        label(
            W / 2,
            38,
            "Hosting Capacity - Baseline Sweep (Case A, no voltage control)",
            COL["text"],
            17,
            700,
        )
    )

    # ==================================================================
    # Baseline-HC setup
    # ==================================================================

    start = Rg(
        "start",
        260, 60,
        660, 58,
    )

    draw_node(
        out,
        start,
        COL["blue"],
        "run_baseline_hc(net, network_id, runpp_kwargs)",
        (
            "caller network is not modified",
        ),
        12.0,
        8.4,
    )

    setup = Rg(
        "setup",
        240, 145,
        700, 76,
    )

    draw_node(
        out,
        setup,
        COL["green"],
        "Deep-copy network + build runpp kwargs",
        (
            "algorithm='nr'; voltage_depend_loads=False is always enforced",
            "infer modal distribution voltage; select MV/LV HC_PARAMS",
        ),
        10.8,
        7.7,
    )

    add_route(
        out, routes,
        "start-setup",
        [
            start.a("bottom"),
            setup.a("top", TARGET_GAP),
        ],
        ignore=("start", "setup"),
    )

    prep = Rg(
        "prep",
        220, 248,
        740, 90,
    )

    draw_node(
        out,
        prep,
        COL["green"],
        "Prepare deterministic worst-case HC snapshot",
        (
            "end-of-feeder = farthest reachable distribution-level bus from slack",
            "load P/Q *= 0.1; existing in-service sgens set to inferred rated P; all existing Q=0",
            "initialize HC/result metrics, step_mw and total_mw",
        ),
        10.6,
        7.4,
    )

    add_route(
        out, routes,
        "setup-prep",
        [
            setup.a("bottom"),
            prep.a("top", TARGET_GAP),
        ],
        ignore=("setup", "prep"),
    )

    # ==================================================================
    # Outer HC sweep entry
    #
    # Initial entry and accepted clean increments meet here.
    # ==================================================================

    sweepm = Cg(
        "sweepm",
        cx, 365,
        10,
    )

    out.append(
        circle_svg(
            sweepm,
            COL["neutral"],
        )
    )

    add_route(
        out, routes,
        "prep-sweepm",
        [
            prep.a("bottom"),
            sweepm.a("top", TARGET_GAP),
        ],
        ignore=("prep", "sweepm"),
    )

    dloop = Dg(
        "dloop",
        cx, 420,
        300, 62,
    )

    draw_decision(
        out,
        dloop,
        "total_mw <= params['max']?",
        ts=11.0,
    )

    add_route(
        out, routes,
        "sweepm-loop",
        [
            sweepm.a("bottom"),
            dloop.a("top", TARGET_GAP),
        ],
        ignore=("sweepm", "dloop"),
    )

    # Centre-aligned with the loop decision.
    # Shifted left slightly so the outer-loop rail has its own space.
    limit = Rg(
        "limit",
        780, 387,
        260, 66,
    )

    draw_node(
        out,
        limit,
        COL["purple"],
        "Sweep completed without break",
        (
            "hc_limit_reached=True; retain last accepted hc_mw",
        ),
        9.4,
        6.9,
        9,
    )

    add_route(
        out, routes,
        "loop-limit",
        [
            dloop.a("right"),
            limit.a("left", TARGET_GAP),
        ],
        "edge-gate",
        ignore=("dloop", "limit"),
    )

    branch_label(
        out,
        (dloop.right + limit.left) / 2,
        dloop.cy - 8,
        "no - loop exhausted",
        COL["gate"],
        size=7.8,
    )

    add = Rg(
        "add",
        390, 480,
        400, 68,
    )

    draw_node(
        out,
        add,
        COL["green"],
        "Add one PV sgen at end-of-feeder",
        (
            "p_mw=step_mw, q_mvar=0; increment total_mw and round to 6 decimals",
        ),
        10.1,
        7.3,
    )

    add_route(
        out, routes,
        "loop-add",
        [
            dloop.a("bottom"),
            add.a("top", TARGET_GAP),
        ],
        "edge-gate",
        ignore=("dloop", "add"),
    )

    branch_label(
        out,
        dloop.cx + 16,
        dloop.bottom + 15,
        "yes - next step",
        COL["gate"],
        "start",
        8.3,
    )

    # ==================================================================
    # Snapshot PF
    # ==================================================================

    pf = Rg(
        "pf",
        390, 575,
        400, 58,
    )

    draw_node(
        out,
        pf,
        COL["blue"],
        "try pp.runpp(net, **kwargs)",
        (
            "evaluate the incremented snapshot",
        ),
        10.6,
        7.7,
    )

    add_route(
        out, routes,
        "add-pf",
        [
            add.a("bottom"),
            pf.a("top", TARGET_GAP),
        ],
        ignore=("add", "pf"),
    )

    dpf = Dg(
        "dpf",
        cx, 690,
        270, 58,
    )

    draw_decision(
        out,
        dpf,
        "runpp succeeded?",
        ts=10.7,
    )

    add_route(
        out, routes,
        "pf-dpf",
        [
            pf.a("bottom"),
            dpf.a("top", TARGET_GAP),
        ],
        ignore=("pf", "dpf"),
    )

    # Exact vertical centre match with dpf.
    pffail = Rg(
        "pffail",
        60, 657,
        280, 66,
    )

    draw_node(
        out,
        pffail,
        COL["red"],
        "Treat PF failure as violating step",
        (
            "violated_at_mw=total_mw; binding=-1/NaN; drop last-added sgen; break",
        ),
        9.2,
        6.7,
        9,
    )

    add_route(
        out, routes,
        "dpf-fail",
        [
            dpf.a("left"),
            pffail.a("right", TARGET_GAP),
        ],
        "edge-hil",
        ignore=("dpf", "pffail"),
    )

    branch_label(
        out,
        (dpf.left + pffail.right) / 2,
        dpf.cy - 8,
        "no / exception",
        COL["hil"],
        size=8.0,
    )

    detect = Rg(
        "detect",
        390, 750,
        400, 58,
    )

    draw_node(
        out,
        detect,
        COL["green"],
        "report = detect_violations(net)",
        (
            "voltage violation detector on solved snapshot",
        ),
        10.3,
        7.5,
    )

    add_route(
        out, routes,
        "dpf-detect",
        [
            dpf.a("bottom"),
            detect.a("top", TARGET_GAP),
        ],
        ignore=("dpf", "detect"),
    )

    branch_label(
        out,
        dpf.cx + 14,
        dpf.bottom + 14,
        "yes",
        anchor="start",
        size=8.2,
    )

    # ==================================================================
    # Solved-snapshot violation gate
    # ==================================================================

    dv = Dg(
        "dv",
        cx, 865,
        280, 60,
    )

    draw_decision(
        out,
        dv,
        "report.any_violations?",
        ts=10.7,
    )

    add_route(
        out, routes,
        "detect-dv",
        [
            detect.a("bottom"),
            dv.a("top", TARGET_GAP),
        ],
        ignore=("detect", "dv"),
    )

    # Exact centre match with dv.
    violate = Rg(
        "violate",
        55, 829,
        305, 72,
    )

    draw_node(
        out,
        violate,
        COL["red"],
        "Record first violating step",
        (
            "extract binding bus/vm; append sweep point; drop last-added sgen; break",
        ),
        9.4,
        6.8,
        9,
    )

    add_route(
        out, routes,
        "dv-viol",
        [
            dv.a("left"),
            violate.a("right", TARGET_GAP),
        ],
        "edge-hil",
        ignore=("dv", "violate"),
    )

    branch_label(
        out,
        (dv.left + violate.right) / 2,
        dv.cy - 8,
        "yes",
        COL["hil"],
        size=8.2,
    )

    clean = Rg(
        "clean",
        390, 925,
        400, 68,
    )

    draw_node(
        out,
        clean,
        COL["purple"],
        "Accept violation-free increment",
        (
            "hc_mw=total_mw; append {mw, max_vm_pu} to sweep_curve",
        ),
        10.0,
        7.3,
    )

    add_route(
        out, routes,
        "dv-clean",
        [
            dv.a("bottom"),
            clean.a("top", TARGET_GAP),
        ],
        ignore=("dv", "clean"),
    )

    branch_label(
        out,
        dv.cx + 14,
        dv.bottom + 14,
        "no",
        anchor="start",
        size=8.2,
    )

    # ==================================================================
    # Failure merge
    #
    # Actual violation drops vertically.
    # Earlier PF failure enters from the left.
    # ==================================================================

    failm = Cg(
        "failm",
        violate.cx,
        1040,
        10,
    )

    out.append(
        circle_svg(
            failm,
            COL["neutral"],
        )
    )

    add_route(
        out, routes,
        "viol-failm",
        [
            violate.a("bottom"),
            failm.a("top", TARGET_GAP),
        ],
        "edge-hil",
        ignore=("violate", "failm"),
    )

    add_route(
        out, routes,
        "pffail-failm",
        [
            pffail.a("left"),
            (pf_fail_rail_x, pffail.cy),
            (pf_fail_rail_x, failm.cy),
            failm.a("left", TARGET_GAP),
        ],
        "edge-hil",
        ignore=("pffail", "failm"),
    )

    # ==================================================================
    # Terminal merge
    # ==================================================================

    endm = Cg(
        "endm",
        cx, 1040,
        10,
    )

    out.append(
        circle_svg(
            endm,
            COL["neutral"],
        )
    )

    add_route(
        out, routes,
        "failm-endm",
        [
            failm.a("right"),
            endm.a("left", TARGET_GAP),
        ],
        "edge-hil",
        ignore=("failm", "endm"),
    )

    # Normal loop exhaustion remains on the right of the main execution
    # chain but inside the outer-loop perimeter.
    add_route(
        out, routes,
        "limit-endm",
        [
            limit.a("bottom"),
            (limit.cx, endm.cy),
            endm.a("right", TARGET_GAP),
        ],
        "edge-gate",
        ignore=("limit", "endm"),
    )

    # ==================================================================
    # Clean-step outer loopback
    #
    # Dedicated far-right rail. It rejoins at sweepm and no longer
    # overlaps the PF-failure terminal path.
    # ==================================================================

    add_route(
        out, routes,
        "clean-loop",
        [
            clean.a("right"),
            (outer_loop_x, clean.cy),
            (outer_loop_x, sweepm.cy),
            sweepm.a("right", TARGET_GAP),
        ],
        "edge-loop",
        ignore=("clean", "sweepm"),
    )

    branch_label(
        out,
        outer_loop_x - 8,
        clean.cy - 10,
        "next step",
        COL["loop"],
        "end",
        8.0,
    )

    # ==================================================================
    # Return
    # ==================================================================

    ret = Rg(
        "ret",
        270, 1080,
        640, 86,
    )

    draw_node(
        out,
        ret,
        COL["purple"],
        "Return (HCResult, net at last violation-free capacity)",
        (
            "HCResult: hc_mw, violated_at, binding bus/vm, n_steps, limit flag, sweep_curve",
            "on violation/PF failure, the last-added PV increment is removed",
        ),
        10.7,
        7.6,
    )

    add_route(
        out, routes,
        "end-ret",
        [
            endm.a("bottom"),
            ret.a("top", TARGET_GAP),
        ],
        ignore=("endm", "ret"),
    )

    # ==================================================================
    # Geometry checks
    # ==================================================================

    audit_bounds(nodes, W, H, 8)
    audit_node_overlaps(nodes)
    audit_routes(routes, nodes)

    # Side failure branches are true horizontal connections.
    assert abs(pffail.cy - dpf.cy) < 1e-9
    assert abs(violate.cy - dv.cy) < 1e-9

    # Actual violation enters its failure merge vertically.
    assert abs(failm.cx - violate.cx) < 1e-9

    # Outer HC iteration and PF-failure routing occupy opposite sides.
    assert pf_fail_rail_x < pffail.left
    assert outer_loop_x > limit.right + TARGET_GAP
    assert outer_loop_x < W - 14

    # Outer loop rejoins before the while-condition decision.
    assert sweepm.cy < dloop.top

    # Terminal failures stage before the common return merge.
    assert failm.cy == endm.cy
    assert failm.right < endm.left

    # Compact canvas with a normal lower margin.
    assert H - ret.bottom <= 45

    write(
        "flow_hc_baseline_ieee_v1",
        W,
        H,
        "Hosting capacity baseline sweep - IEEE",
        "\n".join(out),
    )

def build_baseline_presentation():
    W,H=1920,1080
    out=[rect_svg(R(0,0,W,H),COL["panel"],0)]
    nodes,routes={},[]
    def Rg(n,x,y,w,h): g=R(x,y,w,h); nodes[n]=g; return g
    def Dg(n,x,y,w,h): g=D(x,y,w,h); nodes[n]=g; return g
    def Cg(n,x,y,r): g=C(x,y,r); nodes[n]=g; return g
    cx=510; loop_x=12

    start=Rg("start",155,20,710,46); draw_node(out,start,COL["blue"],"run_baseline_hc: isolate net + configure deterministic HC sweep",("voltage-only worst-case snapshot analysis",),10.7,7.3)
    prep=Rg("prep",135,88,750,58); draw_node(out,prep,COL["green"],"Infer voltage class + end-of-feeder; set worst-case snapshot",("10% load; existing in-service generation at inferred rated P; Q reset to zero",),9.7,6.7)
    add_route(out,routes,"s-p",[start.a("bottom"),prep.a("top",TARGET_GAP)],ignore=("start","prep"))

    dloop=Dg("dloop",cx,205,250,42); draw_decision(out,dloop,"within HC sweep bound?",ts=9.7); add_route(out,routes,"p-l",[prep.a("bottom"),dloop.a("top",TARGET_GAP)],ignore=("prep","dloop"))
    limit=Rg("limit",720,183,260,44); draw_node(out,limit,COL["purple"],"No violation before loop exhaustion",("set hc_limit_reached",),8.2,5.9,8); add_route(out,routes,"l-limit",[dloop.a("right"),limit.a("left",TARGET_GAP)],"edge-gate",ignore=("dloop","limit")); branch_label(out,(dloop.right+limit.left)/2,dloop.cy-7,"no",COL["gate"],size=7.5)

    add=Rg("add",300,260,420,50); draw_node(out,add,COL["green"],"Add one PV step at end-of-feeder",("increment total_mw; new sgen starts at Q=0",),8.8,6.2); add_route(out,routes,"l-add",[dloop.a("bottom"),add.a("top",TARGET_GAP)],"edge-gate",ignore=("dloop","add")); branch_label(out,dloop.cx+10,dloop.bottom+11,"yes",COL["gate"],"start",7.5)
    pf=Rg("pf",300,335,420,48); draw_node(out,pf,COL["blue"],"runpp -> detect_violations",("PF exception is treated as a violating step",),8.8,6.2); add_route(out,routes,"add-pf",[add.a("bottom"),pf.a("top",TARGET_GAP)],ignore=("add","pf"))
    dv=Dg("dv",cx,445,230,40); draw_decision(out,dv,"violating step?",ts=9.5); add_route(out,routes,"pf-dv",[pf.a("bottom"),dv.a("top",TARGET_GAP)],ignore=("pf","dv"))
    fail=Rg("fail",45,422,270,46); draw_node(out,fail,COL["red"],"Record first violation / PF failure",("drop the last-added PV and stop sweep",),8.1,5.9,8); add_route(out,routes,"dv-f",[dv.a("left"),fail.a("right",TARGET_GAP)],"edge-hil",ignore=("dv","fail")); branch_label(out,(dv.left+fail.right)/2,dv.cy-7,"yes",COL["hil"],size=7.4)
    clean=Rg("clean",300,505,420,48); draw_node(out,clean,COL["purple"],"Accept step as current hosting capacity",("hc_mw=total_mw; append sweep point",),8.8,6.2); add_route(out,routes,"dv-c",[dv.a("bottom"),clean.a("top",TARGET_GAP)],ignore=("dv","clean")); branch_label(out,dv.cx+10,dv.bottom+11,"no",anchor="start",size=7.4)
    add_route(out,routes,"clean-loop",[clean.a("left"),(loop_x,clean.cy),(loop_x,dloop.cy),dloop.a("left",TARGET_GAP)],"edge-loop",ignore=("clean","dloop")); branch_label(out,loop_x+8,clean.cy-8,"next PV step",COL["loop"],"start",7.2)

    em=Cg("em",cx,610,9); out.append(circle_svg(em,COL["neutral"])); add_route(out,routes,"fail-em",[fail.a("bottom"),(fail.cx,em.cy),em.a("left",TARGET_GAP)],"edge-hil",ignore=("fail","em")); add_route(out,routes,"limit-em",[limit.a("bottom"),(limit.cx,em.cy),em.a("right",TARGET_GAP)],"edge-gate",ignore=("limit","em"))
    ret=Rg("ret",210,645,600,62); draw_node(out,ret,COL["purple"],"Return HCResult + last violation-free network",("first violating baseline increment has been removed",),9.5,6.6); add_route(out,routes,"em-r",[em.a("bottom"),ret.a("top",TARGET_GAP)],ignore=("em","ret"))

    px,pw=1080,800; panels=[R(px,24,pw,280),R(px,326,pw,310),R(px,658,pw,380)]
    for p in panels: out.append(rect_svg(p,COL["white"],16,COL["detail_border"],1.2))
    out.append(label(px+24,57,"Deterministic HC setup",COL["text"],17,700,"start"))
    lines=[
        "Both HC cases operate on a deep copy, so the network supplied by the benchmark runner is not modified.",
        "The distribution level is inferred from the modal bus voltage: >1 kV selects MV parameters and <=1 kV selects LV parameters.",
        "The end-of-feeder target is the reachable distribution-level bus with the greatest topological distance from the slack.",
        "Before the sweep, load P/Q is reduced to 10% and existing in-service generators are placed at inferred rated active power.",
        "The analysis evaluates voltage violations only; every power flow forces voltage-dependent loads off and uses Newton-Raphson.",
    ]
    for i,t in enumerate(lines): out.append(label(px+24,94+42*i,t,COL["text"],10.7,600,"start"))
    out.append(label(px+24,359,"Incremental baseline sweep",COL["text"],17,700,"start"))
    lines=[
        "Each iteration adds one PV sgen at the same end-of-feeder bus and increases total_mw by the MV or LV step size.",
        "A solved snapshot is passed to detect_violations; a power-flow exception is handled as a violating step.",
        "The first violating step records the binding voltage location and is removed from the returned baseline network.",
        "Every clean step becomes the latest hosting capacity and contributes one {mw, max_vm_pu} point to sweep_curve.",
        "If the loop finishes without a violating step, hc_limit_reached is set and the last accepted capacity is retained.",
    ]
    for i,t in enumerate(lines): out.append(label(px+24,396+43*i,t,COL["text"],10.6,600,"start"))
    out.append(label(px+24,691,"Result semantics",COL["text"],17,700,"start"))
    lines=[
        "HCResult stores hc_mw, the first violating MW level, binding bus/voltage, step count, sweep limit status and sweep curve.",
        "Baseline HC additionally returns the deep-copied network at the last accepted PV capacity for downstream stressed analysis.",
        "MV uses 0.5 MW increments up to the configured 20 MW bound; LV uses 0.01 MW increments up to 0.5 MW.",
        "No Monte Carlo placement or annual QSTS is performed here; this module implements the deterministic snapshot method.",
    ]
    for i,t in enumerate(lines): out.append(label(px+24,728+54*i,t,COL["text"],10.7,600,"start"))

    audit_bounds(nodes,W,H,0); audit_node_overlaps(nodes); audit_routes(routes,nodes)
    assert ret.bottom < H-250
    write("flow_hc_baseline_presentation_v1",W,H,"Hosting capacity baseline sweep - presentation","\n".join(out))

def build_voltvar_ieee():
    W, H = 1180, 1950

    out = [
        rect_svg(
            R(14, 14, W - 28, H - 28),
            COL["panel"],
            16,
            COL["panel_border"],
            1.5,
        )
    ]

    nodes, routes = {}, []

    def Rg(n, x, y, w, h):
        g = R(x, y, w, h)
        nodes[n] = g
        return g

    def Dg(n, x, y, w, h):
        g = D(x, y, w, h)
        nodes[n] = g
        return g

    def Cg(n, x, y, r):
        g = C(x, y, r)
        nodes[n] = g
        return g

    cx = 620

    # Nested-loop and terminal rails are intentionally separated.
    qloop_x = 195          # inner Q(V) fixed-point loop
    no_der_rail_x = 1080   # skip Q(V) when no controlled DERs
    limit_rail_x = 1135    # normal HC while-loop exhaustion
    outer_loop_x = 1158    # accepted-step outer HC sweep loopback

    out.append(
        label(
            W / 2,
            38,
            "Hosting Capacity - Volt-Var Sweep (Case B, Q(V) fixed-point)",
            COL["text"],
            17,
            700,
        )
    )

    # ==================================================================
    # Case-B setup
    # ==================================================================

    start = Rg(
        "start",
        290, 60,
        660, 58,
    )

    draw_node(
        out,
        start,
        COL["blue"],
        "run_hc_with_volt_var(net, network_id, runpp_kwargs)",
        (
            "returns HCResult; caller network remains untouched",
        ),
        11.8,
        8.3,
    )

    setup = Rg(
        "setup",
        260, 145,
        720, 82,
    )

    draw_node(
        out,
        setup,
        COL["green"],
        "Deep-copy + deterministic HC setup",
        (
            "same voltage-class, HC_PARAMS, end-of-feeder and "
            "10%-load/max-generation snapshot as Case A",
            "build runpp kwargs with voltage_depend_loads=False",
        ),
        10.5,
        7.5,
    )

    add_route(
        out, routes,
        "s-set",
        [
            start.a("bottom"),
            setup.a("top", TARGET_GAP),
        ],
        ignore=("start", "setup"),
    )

    ctrl = Rg(
        "ctrl",
        250, 250,
        740, 84,
    )

    draw_node(
        out,
        ctrl,
        COL["green"],
        "Create dry-run VoltVarController and configure BEFORE HC sweep",
        (
            "interface=None; controller resolves only the pre-existing "
            "in-service DER fleet",
            "PV sgens added later by the HC sweep are not in "
            "ctrl.sgen_indices and receive no Volt-Var Q",
        ),
        10.2,
        7.3,
    )

    add_route(
        out, routes,
        "set-ctrl",
        [
            setup.a("bottom"),
            ctrl.a("top", TARGET_GAP),
        ],
        ignore=("setup", "ctrl"),
    )

    init = Rg(
        "init",
        290, 360,
        660, 66,
    )

    draw_node(
        out,
        init,
        COL["purple"],
        "Initialize HC + Q(V) tracking",
        (
            "all_qv_converged=True; qv_iters_max=0; step_mw and total_mw",
        ),
        10.0,
        7.2,
    )

    add_route(
        out, routes,
        "ctrl-init",
        [
            ctrl.a("bottom"),
            init.a("top", TARGET_GAP),
        ],
        ignore=("ctrl", "init"),
    )

    # ==================================================================
    # Outer deterministic HC sweep
    #
    # Initial entry and accepted-step loopback meet at sweepm. This
    # prevents the outer loopback from crossing the lower failure paths.
    # ==================================================================

    sweepm = Cg(
        "sweepm",
        cx, 445,
        10,
    )

    out.append(
        circle_svg(
            sweepm,
            COL["neutral"],
        )
    )

    add_route(
        out, routes,
        "init-sweepm",
        [
            init.a("bottom"),
            sweepm.a("top", TARGET_GAP),
        ],
        ignore=("init", "sweepm"),
    )

    dloop = Dg(
        "dloop",
        cx, 500,
        300, 60,
    )

    draw_decision(
        out,
        dloop,
        "total_mw <= params['max']?",
        ts=10.8,
    )

    add_route(
        out, routes,
        "sweepm-loop",
        [
            sweepm.a("bottom"),
            dloop.a("top", TARGET_GAP),
        ],
        ignore=("sweepm", "dloop"),
    )

    limit = Rg(
        "limit",
        860, 469,
        270, 62,
    )

    draw_node(
        out,
        limit,
        COL["purple"],
        "Sweep completed without break",
        (
            "hc_limit_reached=True",
        ),
        9.2,
        6.7,
        9,
    )

    add_route(
        out, routes,
        "loop-limit",
        [
            dloop.a("right"),
            limit.a("left", TARGET_GAP),
        ],
        "edge-gate",
        ignore=("dloop", "limit"),
    )

    branch_label(
        out,
        (dloop.right + limit.left) / 2,
        dloop.cy - 8,
        "no",
        COL["gate"],
        size=8.0,
    )

    add = Rg(
        "add",
        420, 560,
        400, 68,
    )

    draw_node(
        out,
        add,
        COL["green"],
        "Add one uncontrolled HC PV increment",
        (
            "end-of-feeder; p_mw=step_mw; q_mvar=0; increment total_mw",
        ),
        9.9,
        7.2,
    )

    add_route(
        out, routes,
        "loop-add",
        [
            dloop.a("bottom"),
            add.a("top", TARGET_GAP),
        ],
        "edge-gate",
        ignore=("dloop", "add"),
    )

    branch_label(
        out,
        dloop.cx + 15,
        dloop.bottom + 14,
        "yes - next step",
        COL["gate"],
        "start",
        8.1,
    )

    dders = Dg(
        "dders",
        cx, 675,
        250, 56,
    )

    draw_decision(
        out,
        dders,
        "ctrl.n_ders > 0?",
        ts=10.5,
    )

    add_route(
        out, routes,
        "add-dders",
        [
            add.a("bottom"),
            dders.a("top", TARGET_GAP),
        ],
        ignore=("add", "dders"),
    )

    # ==================================================================
    # Q(V) branch for pre-existing controlled DERs
    # ==================================================================

    reset = Rg(
        "reset",
        400, 725,
        440, 58,
    )

    draw_node(
        out,
        reset,
        COL["green"],
        "Reset controlled DER q_mvar = 0",
        (
            "then call _qv_converge(net, ctrl, kwargs)",
        ),
        9.7,
        7.0,
    )

    add_route(
        out, routes,
        "dders-reset",
        [
            dders.a("bottom"),
            reset.a("top", TARGET_GAP),
        ],
        "edge-gate",
        ignore=("dders", "reset"),
    )

    branch_label(
        out,
        dders.cx + 13,
        dders.bottom + 13,
        "yes",
        COL["gate"],
        "start",
        8.0,
    )

    # ==================================================================
    # _qv_converge fixed-point execution
    # ==================================================================

    qinit = Rg(
        "qinit",
        420, 810,
        400, 64,
    )

    draw_node(
        out,
        qinit,
        COL["purple"],
        "_qv_converge: q_prev=zeros; converged=False",
        (
            "iterate n_iter = 1 .. MAX_QV_ITERS (10)",
        ),
        9.2,
        6.7,
        9,
    )

    add_route(
        out, routes,
        "reset-qinit",
        [
            reset.a("bottom"),
            qinit.a("top", TARGET_GAP),
        ],
        ignore=("reset", "qinit"),
    )

    qpf = Rg(
        "qpf",
        420, 900,
        400, 58,
    )

    draw_node(
        out,
        qpf,
        COL["blue"],
        "try pp.runpp() for current Q iterate",
        (),
        9.5,
        6.8,
        9,
    )

    add_route(
        out, routes,
        "qinit-qpf",
        [
            qinit.a("bottom"),
            qpf.a("top", TARGET_GAP),
        ],
        ignore=("qinit", "qpf"),
    )

    dqpf = Dg(
        "dqpf",
        cx, 1000,
        250, 54,
    )

    draw_decision(
        out,
        dqpf,
        "inner runpp succeeded?",
        ts=9.5,
    )

    add_route(
        out, routes,
        "qpf-dqpf",
        [
            qpf.a("bottom"),
            dqpf.a("top", TARGET_GAP),
        ],
        ignore=("qpf", "dqpf"),
    )

    qcalc = Rg(
        "qcalc",
        390, 1055,
        460, 76,
    )

    draw_node(
        out,
        qcalc,
        COL["green"],
        "Compute and apply next Q iterate",
        (
            "read DER-bus vm_pu; "
            "QVCharacteristic.compute_setpoints(vm, p_installed)",
            "clamp with ctrl._clamp_to_net_limits; "
            "write controlled sgen.q_mvar",
        ),
        9.1,
        6.5,
        8,
    )

    add_route(
        out, routes,
        "dqpf-qcalc",
        [
            dqpf.a("bottom"),
            qcalc.a("top", TARGET_GAP),
        ],
        ignore=("dqpf", "qcalc"),
    )

    branch_label(
        out,
        dqpf.cx + 12,
        dqpf.bottom + 12,
        "yes",
        anchor="start",
        size=7.6,
    )

    dq = Dg(
        "dq",
        cx, 1175,
        270, 58,
    )

    draw_decision(
        out,
        dq,
        "max|q_new-q_prev| < 1e-4?",
        ts=9.1,
    )

    add_route(
        out, routes,
        "qcalc-dq",
        [
            qcalc.a("bottom"),
            dq.a("top", TARGET_GAP),
        ],
        ignore=("qcalc", "dq"),
    )

    # ------------------------------------------------------------------
    # Not converged.
    #
    # The source assigns q_prev = q_new before the next iteration.
    # Put this decision beside dq so the no-branch is short and direct.
    # ------------------------------------------------------------------

    dmoreq = Dg(
        "dmoreq",
        350, 1175,
        230, 58,
    )

    draw_decision(
        out,
        dmoreq,
        "n_iter < MAX_QV_ITERS?",
        (
            "q_prev = q_new before next iteration",
        ),
        8.8,
        6.5,
    )

    add_route(
        out, routes,
        "dq-more",
        [
            dq.a("left"),
            dmoreq.a("right", TARGET_GAP),
        ],
        ignore=("dq", "dmoreq"),
    )

    branch_label(
        out,
        (dmoreq.right + dq.left) / 2,
        dq.cy - 7,
        "no",
        size=7.5,
    )

    # Inner Q(V) iteration stays local on the left.
    add_route(
        out, routes,
        "dmoreq-qpf",
        [
            dmoreq.a("left"),
            (qloop_x, dmoreq.cy),
            (qloop_x, qpf.cy),
            qpf.a("left", TARGET_GAP),
        ],
        "edge-loop",
        ignore=("dmoreq", "qpf"),
    )

    branch_label(
        out,
        dmoreq.left - 12,
        dmoreq.cy - 7,
        "yes",
        COL["loop"],
        "end",
        7.4,
    )

    # ------------------------------------------------------------------
    # Fixed point reached.
    #
    # This return is directly above the common helper-return merge.
    # ------------------------------------------------------------------

    qok = Rg(
        "qok",
        510, 1230,
        220, 60,
    )

    draw_node(
        out,
        qok,
        COL["purple"],
        "Return True, n_iter",
        (
            "fixed point reached",
        ),
        8.2,
        5.9,
        8,
    )

    add_route(
        out, routes,
        "dq-ok",
        [
            dq.a("bottom"),
            qok.a("top", TARGET_GAP),
        ],
        "edge-gate",
        ignore=("dq", "qok"),
    )

    branch_label(
        out,
        dq.cx + 12,
        dq.bottom + 12,
        "yes",
        COL["gate"],
        "start",
        7.6,
    )

    # ------------------------------------------------------------------
    # Iteration budget exhausted.
    # ------------------------------------------------------------------

    qexhaust = Rg(
        "qexhaust",
        170, 1300,
        360, 60,
    )

    draw_node(
        out,
        qexhaust,
        COL["red"],
        "Return False, MAX_QV_ITERS",
        (
            "last Q iterate remains applied; outer sweep continues",
        ),
        8.4,
        6.0,
        8,
    )

    add_route(
        out, routes,
        "dmoreq-ex",
        [
            dmoreq.a("bottom"),
            qexhaust.a("top", TARGET_GAP),
        ],
        "edge-hil",
        ignore=("dmoreq", "qexhaust"),
    )

    branch_label(
        out,
        dmoreq.cx + 11,
        dmoreq.bottom + 11,
        "no",
        COL["hil"],
        "start",
        7.4,
    )

    # ------------------------------------------------------------------
    # Inner runpp failure.
    #
    # Put its return block on the same row as qexhaust/qretm rather than
    # using another long right-side return rail.
    # ------------------------------------------------------------------

    qfail = Rg(
        "qfail",
        830, 1300,
        220, 60,
    )

    draw_node(
        out,
        qfail,
        COL["red"],
        "Return False, n_iter",
        (
            "Q(V) PF failure; caller still attempts final PF",
        ),
        8.1,
        5.8,
        8,
    )

    qfail_input_x = qfail.cx

    add_route(
        out, routes,
        "dqpf-qfail",
        [
            dqpf.a("right"),
            (qfail_input_x, dqpf.cy),
            qfail.a("top", TARGET_GAP),
        ],
        "edge-hil",
        ignore=("dqpf", "qfail"),
    )

    branch_label(
        out,
        dqpf.right + 28,
        dqpf.cy - 7,
        "no",
        COL["hil"],
        "start",
        7.6,
    )

    # ==================================================================
    # _qv_converge return merge
    #
    # exhausted -> left
    # converged -> top
    # inner-PF failure -> right
    # ==================================================================

    qretm = Cg(
        "qretm",
        cx, 1330,
        11,
    )

    out.append(
        circle_svg(
            qretm,
            COL["neutral"],
        )
    )

    add_route(
        out, routes,
        "qex-qret",
        [
            qexhaust.a("right"),
            qretm.a("left", TARGET_GAP),
        ],
        "edge-hil",
        ignore=("qexhaust", "qretm"),
    )

    add_route(
        out, routes,
        "qok-qret",
        [
            qok.a("bottom"),
            qretm.a("top", TARGET_GAP),
        ],
        "edge-gate",
        ignore=("qok", "qretm"),
    )

    add_route(
        out, routes,
        "qfail-qret",
        [
            qfail.a("left"),
            qretm.a("right", TARGET_GAP),
        ],
        "edge-hil",
        ignore=("qfail", "qretm"),
    )

    track = Rg(
        "track",
        370, 1375,
        500, 72,
    )

    draw_node(
        out,
        track,
        COL["purple"],
        "Update Q(V) sweep tracking",
        (
            "all_qv_converged &= converged; qv_iters_max=max(...)",
            "if not converged: warn and continue with the last iterate",
        ),
        9.2,
        6.6,
        8,
    )

    add_route(
        out, routes,
        "qret-track",
        [
            qretm.a("bottom"),
            track.a("top", TARGET_GAP),
        ],
        ignore=("qretm", "track"),
    )

    # ==================================================================
    # Merge Q(V)-executed and no-controlled-DER paths
    # ==================================================================

    prem = Cg(
        "prem",
        cx, 1485,
        11,
    )

    out.append(
        circle_svg(
            prem,
            COL["neutral"],
        )
    )

    add_route(
        out, routes,
        "track-prem",
        [
            track.a("bottom"),
            prem.a("top", TARGET_GAP),
        ],
        ignore=("track", "prem"),
    )

    add_route(
        out, routes,
        "dders-no-prem",
        [
            dders.a("right"),
            (no_der_rail_x, dders.cy),
            (no_der_rail_x, prem.cy),
            prem.a("right", TARGET_GAP),
        ],
        ignore=("dders", "prem"),
    )

    branch_label(
        out,
        dders.right + 28,
        dders.cy - 8,
        "no",
        size=7.8,
    )

    # ==================================================================
    # Mandatory final PF and voltage-violation decision
    # ==================================================================

    finalpf = Rg(
        "finalpf",
        420, 1520,
        400, 58,
    )

    draw_node(
        out,
        finalpf,
        COL["blue"],
        "Final pp.runpp() with current Q applied",
        (
            "mandatory post-Q snapshot used for violation detection",
        ),
        9.8,
        7.0,
    )

    add_route(
        out, routes,
        "prem-final",
        [
            prem.a("bottom"),
            finalpf.a("top", TARGET_GAP),
        ],
        ignore=("prem", "finalpf"),
    )

    dfinal = Dg(
        "dfinal",
        cx, 1610,
        250, 56,
    )

    draw_decision(
        out,
        dfinal,
        "final runpp succeeded?",
        ts=9.7,
    )

    add_route(
        out, routes,
        "final-d",
        [
            finalpf.a("bottom"),
            dfinal.a("top", TARGET_GAP),
        ],
        ignore=("finalpf", "dfinal"),
    )

    finalfail = Rg(
        "finalfail",
        60, 1580,
        270, 60,
    )

    draw_node(
        out,
        finalfail,
        COL["red"],
        "Treat final PF failure as violation",
        (
            "record violated_at; binding=-1/NaN; break",
        ),
        8.7,
        6.3,
        8,
    )

    add_route(
        out, routes,
        "dfinal-f",
        [
            dfinal.a("left"),
            finalfail.a("right", TARGET_GAP),
        ],
        "edge-hil",
        ignore=("dfinal", "finalfail"),
    )

    branch_label(
        out,
        (dfinal.left + finalfail.right) / 2,
        dfinal.cy - 7,
        "no",
        COL["hil"],
        size=7.6,
    )

    detect = Rg(
        "detect",
        420, 1660,
        400, 54,
    )

    draw_node(
        out,
        detect,
        COL["green"],
        "detect_violations(net)",
        (),
        9.5,
        6.8,
    )

    add_route(
        out, routes,
        "dfinal-det",
        [
            dfinal.a("bottom"),
            detect.a("top", TARGET_GAP),
        ],
        ignore=("dfinal", "detect"),
    )

    branch_label(
        out,
        dfinal.cx + 12,
        dfinal.bottom + 12,
        "yes",
        anchor="start",
        size=7.6,
    )

    dv = Dg(
        "dv",
        cx, 1750,
        250, 54,
    )

    draw_decision(
        out,
        dv,
        "report.any_violations?",
        ts=9.5,
    )

    add_route(
        out, routes,
        "det-dv",
        [
            detect.a("bottom"),
            dv.a("top", TARGET_GAP),
        ],
        ignore=("detect", "dv"),
    )

    violate = Rg(
        "violate",
        60, 1720,
        270, 60,
    )

    draw_node(
        out,
        violate,
        COL["red"],
        "Record violating step + break",
        (
            "extract binding; append sweep point",
        ),
        8.6,
        6.2,
        8,
    )

    add_route(
        out, routes,
        "dv-v",
        [
            dv.a("left"),
            violate.a("right", TARGET_GAP),
        ],
        "edge-hil",
        ignore=("dv", "violate"),
    )

    branch_label(
        out,
        (dv.left + violate.right) / 2,
        dv.cy - 7,
        "yes",
        COL["hil"],
        size=7.5,
    )

    # Shifted slightly left so the limit and outer-loop perimeter rails
    # have clear visual separation from the block.
    clean = Rg(
        "clean",
        835, 1720,
        270, 60,
    )

    draw_node(
        out,
        clean,
        COL["purple"],
        "Accept step and continue",
        (
            "hc_mw=total_mw; append sweep point",
        ),
        8.6,
        6.2,
        8,
    )

    add_route(
        out, routes,
        "dv-c",
        [
            dv.a("right"),
            clean.a("left", TARGET_GAP),
        ],
        "edge-gate",
        ignore=("dv", "clean"),
    )

    branch_label(
        out,
        (dv.right + clean.left) / 2,
        dv.cy - 7,
        "no",
        COL["gate"],
        size=7.5,
    )

    # ==================================================================
    # Terminal-path merge
    #
    # final PF failure + detected violation -> failm
    # failm + normal sweep exhaustion      -> endm
    # ==================================================================

    failm = Cg(
        "failm",
        195, 1840,
        10,
    )

    out.append(
        circle_svg(
            failm,
            COL["neutral"],
        )
    )

    # Violating-step branch approaches the failure merge locally.
    add_route(
        out, routes,
        "viol-failm",
        [
            violate.a("bottom"),
            failm.a("top", TARGET_GAP),
        ],
        "edge-hil",
        ignore=("violate", "failm"),
    )

    fail_rail_x = 35

    add_route(
        out, routes,
        "finalfail-failm",
        [
            finalfail.a("left"),
            (fail_rail_x, finalfail.cy),
            (fail_rail_x, failm.cy),
            failm.a("left", TARGET_GAP),
        ],
        "edge-hil",
        ignore=("finalfail", "failm"),
    )

    endm = Cg(
        "endm",
        cx, 1840,
        10,
    )

    out.append(
        circle_svg(
            endm,
            COL["neutral"],
        )
    )

    add_route(
        out, routes,
        "failm-end",
        [
            failm.a("right"),
            endm.a("left", TARGET_GAP),
        ],
        "edge-hil",
        ignore=("failm", "endm"),
    )

    # Normal while-loop exhaustion uses its own terminal rail.
    add_route(
        out, routes,
        "limit-end",
        [
            limit.a("right"),
            (limit_rail_x, limit.cy),
            (limit_rail_x, endm.cy),
            endm.a("right", TARGET_GAP),
        ],
        "edge-gate",
        ignore=("limit", "endm"),
    )

    # ==================================================================
    # Accepted-step outer HC loopback
    #
    # The loop now runs on the far-right perimeter and rejoins the
    # dedicated sweep-entry circle above dloop. It no longer crosses the
    # lower red terminal paths.
    # ==================================================================

    add_route(
        out, routes,
        "clean-loop",
        [
            clean.a("right"),
            (outer_loop_x, clean.cy),
            (outer_loop_x, sweepm.cy),
            sweepm.a("right", TARGET_GAP),
        ],
        "edge-loop",
        ignore=("clean", "sweepm"),
    )

    branch_label(
        out,
        outer_loop_x - 8,
        clean.cy - 9,
        "next step",
        COL["loop"],
        "end",
        7.5,
    )

    # ==================================================================
    # Return
    # ==================================================================

    ret = Rg(
        "ret",
        430, 1885,
        380, 44,
    )

    draw_node(
        out,
        ret,
        COL["purple"],
        "return HCResult",
        (),
        9.0,
    )

    add_route(
        out, routes,
        "end-ret",
        [
            endm.a("bottom"),
            ret.a("top", TARGET_GAP),
        ],
        ignore=("endm", "ret"),
    )

    # ==================================================================
    # Geometry checks
    # ==================================================================

    audit_bounds(
        nodes,
        W,
        H,
        8,
    )

    audit_node_overlaps(
        nodes
    )

    audit_routes(
        routes,
        nodes,
    )

    # Inner Q(V) loop stays local to the left.
    assert qloop_x < dmoreq.left

    # Right-side paths now have an explicit inside-to-outside hierarchy:
    # Q(V) work -> no-DER bypass -> limit terminal -> outer HC loop.
    assert no_der_rail_x > qfail.right + TARGET_GAP
    assert no_der_rail_x < limit_rail_x < outer_loop_x < W - 14

    # All three _qv_converge return outcomes use distinct anchors.
    assert qexhaust.cy == qretm.cy == qfail.cy
    assert abs(qok.cx - qretm.cx) < 1e-9

    # Q(V)-executed and no-controlled-DER paths meet before final PF.
    assert track.bottom < prem.top

    # Terminal failures stage locally before the common HC-result merge.
    assert failm.cy == endm.cy
    assert failm.right < endm.left

    # Outer HC loop rejoins above the while decision, avoiding the lower
    # terminal routing band entirely.
    assert sweepm.cy < dloop.top
    assert outer_loop_x > limit.right + TARGET_GAP

    # Keep the final result comfortably inside the canvas.
    assert H - ret.bottom <= 25

    write(
        "flow_hc_voltvar_ieee_v1",
        W,
        H,
        "Hosting capacity Volt-Var sweep - IEEE",
        "\n".join(out),
    )

def build_voltvar_presentation():
    W, H = 1920, 1080

    out = [
        rect_svg(
            R(0, 0, W, H),
            COL["panel"],
            0,
        )
    ]

    nodes, routes = {}, []

    def Rg(n, x, y, w, h):
        g = R(x, y, w, h)
        nodes[n] = g
        return g

    def Dg(n, x, y, w, h):
        g = D(x, y, w, h)
        nodes[n] = g
        return g

    def Cg(n, x, y, r):
        g = C(x, y, r)
        nodes[n] = g
        return g

    cx = 505

    # Distinct routing rails.
    no_der_rail_x = 950
    outer_loop_x = 1010
    finalfail_rail_x = 18

    # ==================================================================
    # Case-B setup
    # ==================================================================

    start = Rg(
        "start",
        150, 18,
        710, 46,
    )

    draw_node(
        out,
        start,
        COL["blue"],
        "run_hc_with_volt_var: deterministic HC sweep + local Q(V)",
        (
            "same worst-case placement method as baseline case",
        ),
        10.6,
        7.2,
    )

    prep = Rg(
        "prep",
        135, 88,
        750, 58,
    )

    draw_node(
        out,
        prep,
        COL["green"],
        "Deep-copy; infer HC parameters/end-of-feeder; set 10% load + max existing generation",
        (),
        9.4,
        6.5,
    )

    ctrl = Rg(
        "ctrl",
        135, 170,
        750, 60,
    )

    draw_node(
        out,
        ctrl,
        COL["green"],
        "Configure dry-run VoltVarController before adding HC PV",
        (
            "only pre-existing DERs are controlled; incremental HC sgens remain Q=0",
        ),
        9.4,
        6.5,
    )

    add_route(
        out, routes,
        "s-p",
        [
            start.a("bottom"),
            prep.a("top", TARGET_GAP),
        ],
        ignore=("start", "prep"),
    )

    add_route(
        out, routes,
        "p-c",
        [
            prep.a("bottom"),
            ctrl.a("top", TARGET_GAP),
        ],
        ignore=("prep", "ctrl"),
    )

    # ==================================================================
    # Outer HC sweep entry
    #
    # Initial entry and accepted clean steps meet here.
    # ==================================================================

    sweepm = Cg(
        "sweepm",
        cx, 258,
        8,
    )

    out.append(
        circle_svg(
            sweepm,
            COL["neutral"],
        )
    )

    add_route(
        out, routes,
        "ctrl-sweepm",
        [
            ctrl.a("bottom"),
            sweepm.a("top", TARGET_GAP),
        ],
        ignore=("ctrl", "sweepm"),
    )

    dloop = Dg(
        "dloop",
        cx, 305,
        245, 40,
    )

    draw_decision(
        out,
        dloop,
        "within HC sweep bound?",
        ts=9.5,
    )

    add_route(
        out, routes,
        "sweepm-loop",
        [
            sweepm.a("bottom"),
            dloop.a("top", TARGET_GAP),
        ],
        ignore=("sweepm", "dloop"),
    )

    limit = Rg(
        "limit",
        720, 283,
        260, 44,
    )

    draw_node(
        out,
        limit,
        COL["purple"],
        "No violation before loop exhaustion",
        (
            "set hc_limit_reached",
        ),
        8.1,
        5.8,
        8,
    )

    add_route(
        out, routes,
        "loop-limit",
        [
            dloop.a("right"),
            limit.a("left", TARGET_GAP),
        ],
        "edge-gate",
        ignore=("dloop", "limit"),
    )

    branch_label(
        out,
        (dloop.right + limit.left) / 2,
        dloop.cy - 7,
        "no",
        COL["gate"],
        size=7.4,
    )

    add = Rg(
        "add",
        295, 350,
        420, 48,
    )

    draw_node(
        out,
        add,
        COL["green"],
        "Add one uncontrolled PV increment at end-of-feeder",
        (
            "increment total_mw; new HC sgen starts at Q=0",
        ),
        8.7,
        6.1,
    )

    add_route(
        out, routes,
        "loop-add",
        [
            dloop.a("bottom"),
            add.a("top", TARGET_GAP),
        ],
        "edge-gate",
        ignore=("dloop", "add"),
    )

    branch_label(
        out,
        dloop.cx + 10,
        dloop.bottom + 11,
        "yes",
        COL["gate"],
        "start",
        7.4,
    )

    # ==================================================================
    # Optional Q(V) fixed-point execution
    # ==================================================================

    dders = Dg(
        "dders",
        cx, 430,
        230, 38,
    )

    draw_decision(
        out,
        dders,
        "controlled DERs present?",
        ts=9.3,
    )

    add_route(
        out, routes,
        "add-dders",
        [
            add.a("bottom"),
            dders.a("top", TARGET_GAP),
        ],
        ignore=("add", "dders"),
    )

    qv = Rg(
        "qv",
        270, 468,
        470, 66,
    )

    draw_node(
        out,
        qv,
        COL["green"],
        "Reset controlled Q -> Q(V) fixed-point + tracking",
        (
            "up to 10 iterations: runpp -> Q(V) -> clamp -> apply",
            "update qv_converged/qv_iters_max; warn and retain last iterate if needed",
        ),
        8.6,
        6.0,
    )

    add_route(
        out, routes,
        "dders-qv",
        [
            dders.a("bottom"),
            qv.a("top", TARGET_GAP),
        ],
        "edge-gate",
        ignore=("dders", "qv"),
    )

    branch_label(
        out,
        dders.cx + 10,
        dders.bottom + 10,
        "yes",
        COL["gate"],
        "start",
        7.3,
    )

    prem = Cg(
        "prem",
        cx, 565,
        8,
    )

    out.append(
        circle_svg(
            prem,
            COL["neutral"],
        )
    )

    # Q(V)-executed path enters from top.
    add_route(
        out, routes,
        "qv-prem",
        [
            qv.a("bottom"),
            prem.a("top", TARGET_GAP),
        ],
        ignore=("qv", "prem"),
    )

    # No controlled DERs bypass the Q(V) block.
    add_route(
        out, routes,
        "dders-no",
        [
            dders.a("right"),
            (no_der_rail_x, dders.cy),
            (no_der_rail_x, prem.cy),
            prem.a("right", TARGET_GAP),
        ],
        ignore=("dders", "prem"),
    )

    branch_label(
        out,
        dders.right + 24,
        dders.cy - 7,
        "no",
        size=7.3,
    )

    # ==================================================================
    # Mandatory final power flow
    # ==================================================================

    final = Rg(
        "final",
        295, 600,
        420, 50,
    )

    draw_node(
        out,
        final,
        COL["blue"],
        "Mandatory final runpp with current Q",
        (
            "post-Q snapshot used for the HC violation decision",
        ),
        8.7,
        6.1,
    )

    add_route(
        out, routes,
        "prem-final",
        [
            prem.a("bottom"),
            final.a("top", TARGET_GAP),
        ],
        ignore=("prem", "final"),
    )

    # ==================================================================
    # Final-PF success is an explicit implementation decision
    # ==================================================================

    dfinal = Dg(
        "dfinal",
        cx, 690,
        230, 40,
    )

    draw_decision(
        out,
        dfinal,
        "final runpp succeeded?",
        ts=9.1,
    )

    add_route(
        out, routes,
        "final-dfinal",
        [
            final.a("bottom"),
            dfinal.a("top", TARGET_GAP),
        ],
        ignore=("final", "dfinal"),
    )

    # Centre-align failure block with its decision.
    finalfail = Rg(
        "finalfail",
        40, 667,
        270, 46,
    )

    draw_node(
        out,
        finalfail,
        COL["red"],
        "Treat final PF failure as violation",
        (
            "record violated_at; no binding report; stop sweep",
        ),
        7.9,
        5.7,
        8,
    )

    add_route(
        out, routes,
        "dfinal-fail",
        [
            dfinal.a("left"),
            finalfail.a("right", TARGET_GAP),
        ],
        "edge-hil",
        ignore=("dfinal", "finalfail"),
    )

    branch_label(
        out,
        (dfinal.left + finalfail.right) / 2,
        dfinal.cy - 7,
        "no",
        COL["hil"],
        size=7.3,
    )

    detect = Rg(
        "detect",
        295, 735,
        420, 46,
    )

    draw_node(
        out,
        detect,
        COL["green"],
        "report = detect_violations(net)",
        (
            "inspect the solved post-Q snapshot",
        ),
        8.6,
        6.0,
    )

    add_route(
        out, routes,
        "dfinal-detect",
        [
            dfinal.a("bottom"),
            detect.a("top", TARGET_GAP),
        ],
        ignore=("dfinal", "detect"),
    )

    branch_label(
        out,
        dfinal.cx + 10,
        dfinal.bottom + 11,
        "yes",
        anchor="start",
        size=7.3,
    )

    # ==================================================================
    # Solved-PF voltage violation decision
    # ==================================================================

    dv = Dg(
        "dv",
        cx, 825,
        230, 40,
    )

    draw_decision(
        out,
        dv,
        "report.any_violations?",
        ts=9.2,
    )

    add_route(
        out, routes,
        "detect-dv",
        [
            detect.a("bottom"),
            dv.a("top", TARGET_GAP),
        ],
        ignore=("detect", "dv"),
    )

    fail = Rg(
        "fail",
        40, 802,
        270, 46,
    )

    draw_node(
        out,
        fail,
        COL["red"],
        "Record violating step and stop",
        (
            "extract binding bus/vm and append violating sweep point",
        ),
        7.9,
        5.7,
        8,
    )

    add_route(
        out, routes,
        "dv-fail",
        [
            dv.a("left"),
            fail.a("right", TARGET_GAP),
        ],
        "edge-hil",
        ignore=("dv", "fail"),
    )

    branch_label(
        out,
        (dv.left + fail.right) / 2,
        dv.cy - 7,
        "yes",
        COL["hil"],
        size=7.3,
    )

    clean = Rg(
        "clean",
        295, 860,
        420, 50,
    )

    draw_node(
        out,
        clean,
        COL["purple"],
        "Accept step + append sweep point",
        (
            "hc_mw=total_mw; append {mw, max_vm_pu}",
        ),
        8.5,
        6.0,
    )

    add_route(
        out, routes,
        "dv-clean",
        [
            dv.a("bottom"),
            clean.a("top", TARGET_GAP),
        ],
        ignore=("dv", "clean"),
    )

    branch_label(
        out,
        dv.cx + 10,
        dv.bottom + 11,
        "no",
        COL["gate"],
        "start",
        7.3,
    )

    # ==================================================================
    # Failure staging
    #
    # Actual voltage violation drops straight into failm.
    # Final-PF failure enters from the left.
    # ==================================================================

    failm = Cg(
        "failm",
        fail.cx,
        950,
        8,
    )

    out.append(
        circle_svg(
            failm,
            COL["neutral"],
        )
    )

    add_route(
        out, routes,
        "fail-failm",
        [
            fail.a("bottom"),
            failm.a("top", TARGET_GAP),
        ],
        "edge-hil",
        ignore=("fail", "failm"),
    )

    add_route(
        out, routes,
        "finalfail-failm",
        [
            finalfail.a("left"),
            (finalfail_rail_x, finalfail.cy),
            (finalfail_rail_x, failm.cy),
            failm.a("left", TARGET_GAP),
        ],
        "edge-hil",
        ignore=("finalfail", "failm"),
    )

    # ==================================================================
    # Common terminal merge
    #
    # violation/final-PF failure -> failm -> endm
    # normal while exhaustion     -> endm
    # ==================================================================

    endm = Cg(
        "endm",
        cx, 950,
        8,
    )

    out.append(
        circle_svg(
            endm,
            COL["neutral"],
        )
    )

    add_route(
        out, routes,
        "failm-endm",
        [
            failm.a("right"),
            endm.a("left", TARGET_GAP),
        ],
        "edge-hil",
        ignore=("failm", "endm"),
    )

    # Keep the sweep-limit terminal path inside the outer loop rail.
    # This avoids a route-route crossing at the top-right.
    add_route(
        out, routes,
        "limit-endm",
        [
            limit.a("bottom"),
            (limit.cx, endm.cy),
            endm.a("right", TARGET_GAP),
        ],
        "edge-gate",
        ignore=("limit", "endm"),
    )

    # ==================================================================
    # Clean-step outer HC loopback
    #
    # Runs on the outer-right perimeter and returns to sweepm.
    # It does not cross either lower failure route.
    # ==================================================================

    add_route(
        out, routes,
        "clean-loop",
        [
            clean.a("right"),
            (outer_loop_x, clean.cy),
            (outer_loop_x, sweepm.cy),
            sweepm.a("right", TARGET_GAP),
        ],
        "edge-loop",
        ignore=("clean", "sweepm"),
    )

    branch_label(
        out,
        outer_loop_x - 8,
        clean.cy - 8,
        "next PV step",
        COL["loop"],
        "end",
        7.2,
    )

    # ==================================================================
    # Return
    # ==================================================================

    ret = Rg(
        "ret",
        230, 985,
        550, 54,
    )

    draw_node(
        out,
        ret,
        COL["purple"],
        "Return Volt-Var HCResult",
        (
            "hc_mw + binding/limit metrics + qv_converged and qv_iters_max",
        ),
        9.1,
        6.4,
    )

    add_route(
        out, routes,
        "endm-ret",
        [
            endm.a("bottom"),
            ret.a("top", TARGET_GAP),
        ],
        ignore=("endm", "ret"),
    )

    # ==================================================================
    # Audience-facing explanatory panels
    # ==================================================================

    px, pw = 1080, 800

    panels = [
        R(px, 24, pw, 280),
        R(px, 326, pw, 330),
        R(px, 678, pw, 360),
    ]

    for p in panels:
        out.append(
            rect_svg(
                p,
                COL["white"],
                16,
                COL["detail_border"],
                1.2,
            )
        )

    out.append(
        label(
            px + 24,
            57,
            "Volt-Var HC setup",
            COL["text"],
            17,
            700,
            "start",
        )
    )

    lines = [
        "Case B uses the same deep-copy, voltage-class, end-of-feeder and deterministic worst-case snapshot preparation as baseline HC.",
        "VoltVarController is configured in dry-run mode before the HC sweep begins, using the same local Q(V) and inverter-limit logic as Scenario 4.",
        "Because the controller resolves its sgen_indices at configuration time, PV capacity added later by the HC sweep is not Q-controlled.",
        "Only the pre-existing controlled DER fleet supplies reactive support while incremental PV is tested at unity power factor.",
    ]

    for i, t in enumerate(lines):
        out.append(
            label(
                px + 24,
                94 + 48 * i,
                t,
                COL["text"],
                10.7,
                600,
                "start",
            )
        )

    out.append(
        label(
            px + 24,
            359,
            "Q(V) fixed-point inside each HC step",
            COL["text"],
            17,
            700,
            "start",
        )
    )

    lines = [
        "When controlled DERs exist, their Q is reset to zero and _qv_converge iterates up to MAX_QV_ITERS=10.",
        "Each inner iteration runs a power flow, reads DER-bus voltages, computes Q(V), clamps to network/inverter limits and writes the new Q.",
        "Convergence requires max|q_new-q_prev| < 1e-4 MVAr; non-convergence is logged and the last iterate is retained.",
        "qv_converged and qv_iters_max are updated immediately after the inner function returns, before the final power flow.",
        "The caller then always performs a final runpp; if no controlled DERs exist, the inner Q(V) loop is simply skipped.",
    ]

    for i, t in enumerate(lines):
        out.append(
            label(
                px + 24,
                396 + 45 * i,
                t,
                COL["text"],
                10.6,
                600,
                "start",
            )
        )

    out.append(
        label(
            px + 24,
            711,
            "Result semantics",
            COL["text"],
            17,
            700,
            "start",
        )
    )

    lines = [
        "A final-PF failure terminates the sweep without a voltage binding report; a solved violating snapshot records the binding bus and voltage.",
        "Every clean snapshot becomes the latest hc_mw and contributes one {mw, max_vm_pu} point to sweep_curve.",
        "Unlike baseline HC, Case B returns HCResult only; its deep-copied working network remains internal to the analysis.",
        "The benchmark runner compares Case B hc_mw against baseline hc_mw to quantify the hosting-capacity gain from the existing controllable DER fleet.",
    ]

    for i, t in enumerate(lines):
        out.append(
            label(
                px + 24,
                748 + 54 * i,
                t,
                COL["text"],
                10.7,
                600,
                "start",
            )
        )

    # ==================================================================
    # Geometry checks
    # ==================================================================

    audit_bounds(nodes, W, H, 0)
    audit_node_overlaps(nodes)
    audit_routes(routes, nodes)

    # Side failure branches are intentionally horizontal.
    assert abs(finalfail.cy - dfinal.cy) < 1e-9
    assert abs(fail.cy - dv.cy) < 1e-9

    # Voltage-violation termination is vertical into its local merge.
    assert abs(failm.cx - fail.cx) < 1e-9

    # No-DER bypass remains inside the outer HC loopback.
    assert no_der_rail_x < outer_loop_x
    assert outer_loop_x > limit.right + TARGET_GAP
    assert outer_loop_x < 1080

    # Outer loop rejoins before the sweep decision.
    assert sweepm.cy < dloop.top

    # Full presentation height is now used deliberately.
    assert ret.bottom > 1030
    assert ret.bottom < H - 30

    write(
        "flow_hc_voltvar_presentation_v1",
        W,
        H,
        "Hosting capacity Volt-Var sweep - presentation",
        "\n".join(out),
    )

def main():
    build_baseline_ieee()
    build_baseline_presentation()
    build_voltvar_ieee()
    build_voltvar_presentation()
    print("wrote hosting_capacity_flowcharts_v1")


if __name__ == "__main__":
    main()
