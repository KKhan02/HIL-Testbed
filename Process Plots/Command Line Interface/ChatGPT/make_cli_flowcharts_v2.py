from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import html
import math
import xml.etree.ElementTree as ET

import cairosvg

OUT = Path(__file__).resolve().parent / "cli_flowcharts_v2"
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
        # Deterministic side-branch L route: horizontal first, then vertical.
        pts = [pts[0], (pts[1][0], pts[0][1]), pts[1]]
    pts = _clean_points(pts)
    routes.append((name, pts, set(ignore)))
    out.append(path(pts, cls, marker))




# =============================================================================
# Diagram 1: __main__.py top-level CLI entry
# =============================================================================

def build_entry_ieee():
    W, H = 1080, 1165
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

    cx = 560

    out.append(
        label(
            W / 2,
            38,
            "HIL CLI - Entry, Preset Recovery and Executor Dispatch",
            COL["text"],
            17,
            700,
        )
    )

    # ==================================================================
    # CLI entry
    # ==================================================================

    start = Rg(
        "start",
        250, 62,
        620, 58,
    )

    draw_node(
        out,
        start,
        COL["blue"],
        "python -m <cli package> -> __main__.main()",
        (
            "plan = None",
        ),
        12.1,
        8.4,
    )

    dpreset = Dg(
        "dpreset",
        cx, 175,
        300, 62,
    )

    draw_decision(
        out,
        dpreset,
        "Load a saved preset instead of wizard?",
        ts=10.8,
    )

    add_route(
        out, routes,
        "start-preset",
        [
            start.a("bottom"),
            dpreset.a("top", TARGET_GAP),
        ],
        ignore=("start", "dpreset"),
    )

    # ==================================================================
    # Optional saved-preset recovery
    # ==================================================================

    load = Rg(
        "load",
        735, 140,
        300, 70,
    )

    draw_node(
        out,
        load,
        COL["green"],
        "_load_preset()",
        (
            "read JSON -> RunPlan.from_dict",
            "semantic warnings may require confirmation",
        ),
        9.7,
        7.1,
        9,
    )

    add_route(
        out, routes,
        "preset-load",
        [
            dpreset.a("right"),
            load.a("left", TARGET_GAP),
        ],
        "edge-gate",
        ignore=("dpreset", "load"),
    )

    branch_label(
        out,
        (dpreset.right + load.left) / 2,
        dpreset.cy - 8,
        "yes",
        COL["gate"],
        size=8.1,
    )

    dplan = Dg(
        "dplan",
        cx, 300,
        260, 58,
    )

    draw_decision(
        out,
        dplan,
        "plan is None?",
        ts=10.8,
    )

    # No preset selected -> plan is still None.
    add_route(
        out, routes,
        "preset-no",
        [
            dpreset.a("bottom"),
            dplan.a("top", TARGET_GAP),
        ],
        ignore=("dpreset", "dplan"),
    )

    branch_label(
        out,
        dpreset.cx + 15,
        dpreset.bottom + 15,
        "no",
        anchor="start",
        size=8.1,
    )

    # Preset result approaches the decision directly from the right.
    # No unnecessary intermediate dogleg.
    add_route(
        out, routes,
        "load-dplan",
        [
            load.a("bottom"),
            (load.cx, dplan.cy),
            dplan.a("right", TARGET_GAP),
        ],
        "edge-gate",
        ignore=("load", "dplan"),
    )

    # ==================================================================
    # Wizard fallback
    # ==================================================================

    wiz = Rg(
        "wiz",
        250, 360,
        620, 66,
    )

    draw_node(
        out,
        wiz,
        COL["green"],
        "run_wizard()",
        (
            "collect a complete RunPlan through the 9-step wizard",
        ),
        11.0,
        7.8,
    )

    add_route(
        out, routes,
        "dplan-wiz",
        [
            dplan.a("bottom"),
            wiz.a("top", TARGET_GAP),
        ],
        "edge-gate",
        ignore=("dplan", "wiz"),
    )

    branch_label(
        out,
        dplan.cx + 15,
        dplan.bottom + 15,
        "yes",
        COL["gate"],
        "start",
        8.1,
    )

    merge = Cg(
        "merge",
        cx, 470,
        11,
    )

    out.append(
        circle_svg(
            merge,
            COL["neutral"],
        )
    )

    add_route(
        out, routes,
        "wiz-merge",
        [
            wiz.a("bottom"),
            merge.a("top", TARGET_GAP),
        ],
        ignore=("wiz", "merge"),
    )

    # A usable loaded plan bypasses the wizard.
    add_route(
        out, routes,
        "plan-ready",
        [
            dplan.a("left"),
            (180, dplan.cy),
            (180, merge.cy),
            merge.a("left", TARGET_GAP),
        ],
        ignore=("dplan", "merge"),
    )

    branch_label(
        out,
        dplan.left - 28,
        dplan.cy - 8,
        "no",
        COL["gate"],
        "end",
        8.1,
    )

    # ==================================================================
    # Preview and confirmation
    # ==================================================================

    preview = Rg(
        "preview",
        250, 505,
        620, 64,
    )

    draw_node(
        out,
        preview,
        COL["purple"],
        "print_run_plan(plan) - called twice consecutively",
        (
            "current __main__.py behavior before the Proceed prompt",
        ),
        10.3,
        7.4,
    )

    add_route(
        out, routes,
        "merge-preview",
        [
            merge.a("bottom"),
            preview.a("top", TARGET_GAP),
        ],
        ignore=("merge", "preview"),
    )

    dgo = Dg(
        "dgo",
        cx, 630,
        240, 58,
    )

    draw_decision(
        out,
        dgo,
        "Proceed?",
        ts=11.0,
    )

    add_route(
        out, routes,
        "preview-go",
        [
            preview.a("bottom"),
            dgo.a("top", TARGET_GAP),
        ],
        ignore=("preview", "dgo"),
    )

    # User cancellation is a normal exit-0 path.
    cancel = Rg(
        "cancel",
        55, 598,
        240, 64,
    )

    draw_node(
        out,
        cancel,
        COL["neutral"],
        "Cancel CLI",
        (
            "print Cancelled; sys.exit(0)",
        ),
        9.5,
        7.0,
        9,
    )

    add_route(
        out, routes,
        "go-cancel",
        [
            dgo.a("left"),
            cancel.a("right", TARGET_GAP),
        ],
        "edge-dark",
        ignore=("dgo", "cancel"),
    )

    branch_label(
        out,
        (dgo.left + cancel.right) / 2,
        dgo.cy - 8,
        "no",
        COL["text"],
        size=8.0,
    )

    # ==================================================================
    # Optional preset persistence
    # ==================================================================

    dsave = Dg(
        "dsave",
        cx, 750,
        260, 58,
    )

    draw_decision(
        out,
        dsave,
        "Save as preset?",
        ts=10.8,
    )

    add_route(
        out, routes,
        "go-save",
        [
            dgo.a("bottom"),
            dsave.a("top", TARGET_GAP),
        ],
        "edge-gate",
        ignore=("dgo", "dsave"),
    )

    branch_label(
        out,
        dgo.cx + 14,
        dgo.bottom + 15,
        "yes",
        COL["gate"],
        "start",
        8.0,
    )

    save = Rg(
        "save",
        750, 718,
        270, 64,
    )

    draw_node(
        out,
        save,
        COL["purple"],
        "_save_preset(plan)",
        (
            "prompt name; non-empty name -> write presets/<name>.json",
        ),
        8.9,
        6.5,
        9,
    )

    add_route(
        out, routes,
        "save-yes",
        [
            dsave.a("right"),
            save.a("left", TARGET_GAP),
        ],
        "edge-gate",
        ignore=("dsave", "save"),
    )

    branch_label(
        out,
        (dsave.right + save.left) / 2,
        dsave.cy - 8,
        "yes",
        COL["gate"],
        size=8.0,
    )

    sm = Cg(
        "sm",
        cx, 845,
        11,
    )

    out.append(
        circle_svg(
            sm,
            COL["neutral"],
        )
    )

    add_route(
        out, routes,
        "save-no",
        [
            dsave.a("bottom"),
            sm.a("top", TARGET_GAP),
        ],
        ignore=("dsave", "sm"),
    )

    branch_label(
        out,
        dsave.cx + 14,
        dsave.bottom + 15,
        "no",
        anchor="start",
        size=8.0,
    )

    # Simplified save-completion route.
    add_route(
        out, routes,
        "save-sm",
        [
            save.a("bottom"),
            (save.cx, sm.cy),
            sm.a("right", TARGET_GAP),
        ],
        "edge-gate",
        ignore=("save", "sm"),
    )

    # ==================================================================
    # Actual outer try/except executor boundary
    #
    # There is no Boolean "executor call returns normally?" decision in
    # __main__.py. A normal executor return is passed directly to
    # sys.exit(); the two side paths are exception handlers.
    # ==================================================================

    exe = Rg(
        "exe",
        300, 885,
        520, 70,
    )

    draw_node(
        out,
        exe,
        COL["blue"],
        "try: sys.exit(executor.execute(plan))",
        (
            "normal executor return is passed directly to sys.exit",
        ),
        10.6,
        7.4,
    )

    add_route(
        out, routes,
        "sm-exe",
        [
            sm.a("bottom"),
            exe.a("top", TARGET_GAP),
        ],
        ignore=("sm", "exe"),
    )

    # Outer KeyboardInterrupt handler.
    kbi = Rg(
        "kbi",
        30, 887,
        220, 66,
    )

    draw_node(
        out,
        kbi,
        COL["red"],
        "KeyboardInterrupt",
        (
            "print Interrupted; exit 130",
        ),
        9.1,
        6.6,
        9,
    )

    add_route(
        out, routes,
        "exe-kbi",
        [
            exe.a("left"),
            kbi.a("right", TARGET_GAP),
        ],
        "edge-hil",
        ignore=("exe", "kbi"),
    )

    branch_label(
        out,
        (exe.left + kbi.right) / 2,
        exe.cy - 8,
        "KeyboardInterrupt",
        COL["hil"],
        size=7.3,
    )

    # Unexpected exception escaping the executor's typed handling.
    bug = Rg(
        "bug",
        870, 886,
        180, 68,
    )

    draw_node(
        out,
        bug,
        COL["red"],
        "Unexpected wrapper error",
        (
            "print_error_message; exit 1",
        ),
        8.3,
        6.0,
        8,
    )

    add_route(
        out, routes,
        "exe-bug",
        [
            exe.a("right"),
            bug.a("left", TARGET_GAP),
        ],
        "edge-hil",
        ignore=("exe", "bug"),
    )

    branch_label(
        out,
        (exe.right + bug.left) / 2,
        exe.cy - 8,
        "other exception",
        COL["hil"],
        size=7.2,
    )

    # Normal executor return terminates through sys.exit(returned_code).
    ret = Rg(
        "ret",
        330, 1025,
        460, 62,
    )

    draw_node(
        out,
        ret,
        COL["purple"],
        "Process exits with returned executor ExitCode",
        (
            "0, 2-8 or 130 depending on executor outcome",
        ),
        10.0,
        7.1,
    )

    add_route(
        out, routes,
        "exe-ret",
        [
            exe.a("bottom"),
            ret.a("top", TARGET_GAP),
        ],
        ignore=("exe", "ret"),
    )

    branch_label(
        out,
        exe.cx + 14,
        exe.bottom + 15,
        "normal return",
        anchor="start",
        size=7.7,
    )

    # ==================================================================
    # Geometry checks
    # ==================================================================

    audit_bounds(nodes, W, H, 8)
    audit_node_overlaps(nodes)
    audit_routes(routes, nodes)

    assert abs(load.cy - dpreset.cy) < 1e-9
    assert abs(cancel.cy - dgo.cy) < 1e-9
    assert abs(save.cy - dsave.cy) < 1e-9

    # Exception handlers are genuinely horizontal side branches.
    assert abs(kbi.cy - exe.cy) < 1e-9
    assert abs(bug.cy - exe.cy) < 1e-9

    assert H - ret.bottom <= 80

    write(
        "flow_cli_entry_ieee_v2",
        W,
        H,
        "HIL CLI top-level entry - IEEE",
        "\n".join(out),
    )

def build_entry_presentation():
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

    cx = 500

    # ==================================================================
    # RunPlan acquisition
    # ==================================================================

    start = Rg(
        "start",
        155, 22,
        690, 46,
    )

    draw_node(
        out,
        start,
        COL["blue"],
        "HIL CLI entry (__main__.main)",
        (
            "start one configured benchmark run",
        ),
        11.2,
        7.5,
    )

    dp = Dg(
        "dp",
        cx, 150,
        245, 42,
    )

    draw_decision(
        out,
        dp,
        "load saved preset?",
        ts=10.2,
    )

    add_route(
        out, routes,
        "s-d",
        [
            start.a("bottom"),
            dp.a("top", TARGET_GAP),
        ],
        ignore=("start", "dp"),
    )

    load = Rg(
        "load",
        700, 126,
        260, 48,
    )

    draw_node(
        out,
        load,
        COL["green"],
        "Load + validate preset",
        (
            "JSON -> RunPlan; warnings may cancel",
        ),
        8.8,
        6.4,
        8,
    )

    add_route(
        out, routes,
        "d-load",
        [
            dp.a("right"),
            load.a("left", TARGET_GAP),
        ],
        "edge-gate",
        ignore=("dp", "load"),
    )

    branch_label(
        out,
        (dp.right + load.left) / 2,
        dp.cy - 7,
        "yes",
        COL["gate"],
        size=7.6,
    )

    dn = Dg(
        "dn",
        cx, 260,
        220, 40,
    )

    draw_decision(
        out,
        dn,
        "usable plan loaded?",
        ts=9.6,
    )

    add_route(
        out, routes,
        "dp-dn",
        [
            dp.a("bottom"),
            dn.a("top", TARGET_GAP),
        ],
        ignore=("dp", "dn"),
    )

    branch_label(
        out,
        dp.cx + 11,
        dp.bottom + 12,
        "no",
        anchor="start",
        size=7.6,
    )

    # Direct L-shaped preset-result route.
    add_route(
        out, routes,
        "load-dn",
        [
            load.a("bottom"),
            (load.cx, dn.cy),
            dn.a("right", TARGET_GAP),
        ],
        "edge-gate",
        ignore=("load", "dn"),
    )

    wiz = Rg(
        "wiz",
        230, 310,
        540, 54,
    )

    draw_node(
        out,
        wiz,
        COL["green"],
        "Run 9-step wizard",
        (
            "used when no preset was selected or loading was cancelled",
        ),
        9.5,
        6.8,
    )

    add_route(
        out, routes,
        "dn-wiz",
        [
            dn.a("bottom"),
            wiz.a("top", TARGET_GAP),
        ],
        "edge-gate",
        ignore=("dn", "wiz"),
    )

    branch_label(
        out,
        dn.cx + 11,
        dn.bottom + 12,
        "no",
        COL["gate"],
        "start",
        7.6,
    )

    m = Cg(
        "m",
        cx, 400,
        8,
    )

    out.append(
        circle_svg(
            m,
            COL["neutral"],
        )
    )

    add_route(
        out, routes,
        "wiz-m",
        [
            wiz.a("bottom"),
            m.a("top", TARGET_GAP),
        ],
        ignore=("wiz", "m"),
    )

    add_route(
        out, routes,
        "loaded-m",
        [
            dn.a("left"),
            (180, dn.cy),
            (180, m.cy),
            m.a("left", TARGET_GAP),
        ],
        ignore=("dn", "m"),
    )

    branch_label(
        out,
        dn.left - 24,
        dn.cy - 7,
        "yes",
        COL["gate"],
        "end",
        7.5,
    )

    # ==================================================================
    # Confirmation and optional persistence
    # ==================================================================

    prev = Rg(
        "prev",
        200, 445,
        600, 56,
    )

    draw_node(
        out,
        prev,
        COL["purple"],
        "Preview RunPlan",
        (
            "terminal tables for run, dataset, network and limits",
        ),
        9.7,
        6.9,
    )

    add_route(
        out, routes,
        "m-prev",
        [
            m.a("bottom"),
            prev.a("top", TARGET_GAP),
        ],
        ignore=("m", "prev"),
    )

    dgo = Dg(
        "dgo",
        cx, 570,
        200, 40,
    )

    draw_decision(
        out,
        dgo,
        "proceed?",
        ts=9.8,
    )

    add_route(
        out, routes,
        "prev-go",
        [
            prev.a("bottom"),
            dgo.a("top", TARGET_GAP),
        ],
        ignore=("prev", "dgo"),
    )

    # Normal user cancellation.
    cancel = Rg(
        "cancel",
        70, 549,
        220, 42,
    )

    draw_node(
        out,
        cancel,
        COL["neutral"],
        "Cancel",
        (
            "exit 0",
        ),
        8.7,
        6.2,
        8,
    )

    add_route(
        out, routes,
        "go-c",
        [
            dgo.a("left"),
            cancel.a("right", TARGET_GAP),
        ],
        "edge-dark",
        ignore=("dgo", "cancel"),
    )

    branch_label(
        out,
        (dgo.left + cancel.right) / 2,
        dgo.cy - 6,
        "no",
        COL["text"],
        size=7.4,
    )

    dsave = Dg(
        "dsave",
        cx, 700,
        215, 40,
    )

    draw_decision(
        out,
        dsave,
        "save preset?",
        ts=9.6,
    )

    add_route(
        out, routes,
        "go-save",
        [
            dgo.a("bottom"),
            dsave.a("top", TARGET_GAP),
        ],
        "edge-gate",
        ignore=("dgo", "dsave"),
    )

    branch_label(
        out,
        dgo.cx + 10,
        dgo.bottom + 10,
        "yes",
        COL["gate"],
        "start",
        7.4,
    )

    save = Rg(
        "save",
        700, 679,
        245, 42,
    )

    draw_node(
        out,
        save,
        COL["purple"],
        "Attempt preset save",
        (
            "non-empty name -> presets/<name>.json",
        ),
        8.1,
        5.8,
        8,
    )

    add_route(
        out, routes,
        "save-y",
        [
            dsave.a("right"),
            save.a("left", TARGET_GAP),
        ],
        "edge-gate",
        ignore=("dsave", "save"),
    )

    branch_label(
        out,
        (dsave.right + save.left) / 2,
        dsave.cy - 6,
        "yes",
        COL["gate"],
        size=7.4,
    )

    sm = Cg(
        "sm",
        cx, 780,
        8,
    )

    out.append(
        circle_svg(
            sm,
            COL["neutral"],
        )
    )

    add_route(
        out, routes,
        "save-n",
        [
            dsave.a("bottom"),
            sm.a("top", TARGET_GAP),
        ],
        ignore=("dsave", "sm"),
    )

    branch_label(
        out,
        dsave.cx + 10,
        dsave.bottom + 10,
        "no",
        anchor="start",
        size=7.4,
    )

    add_route(
        out, routes,
        "save-m",
        [
            save.a("bottom"),
            (save.cx, sm.cy),
            sm.a("right", TARGET_GAP),
        ],
        "edge-gate",
        ignore=("save", "sm"),
    )

    # ==================================================================
    # Executor boundary
    #
    # Keep the presentation compact, but retain the real executable
    # exception paths. There is no fictitious decision diamond.
    # ==================================================================

    exe = Rg(
        "exe",
        255, 825,
        490, 58,
    )

    draw_node(
        out,
        exe,
        COL["blue"],
        "try: executor.execute(plan)",
        (
            "runtime validation, network/profile resolution, benchmark and publishing",
        ),
        9.5,
        6.6,
    )

    add_route(
        out, routes,
        "sm-exe",
        [
            sm.a("bottom"),
            exe.a("top", TARGET_GAP),
        ],
        ignore=("sm", "exe"),
    )

    # Sufficient horizontal shaft is retained for the branch label.
    kbi = Rg(
        "kbi",
        20, 830,
        185, 48,
    )

    draw_node(
        out,
        kbi,
        COL["red"],
        "KeyboardInterrupt",
        (
            "outer safety net -> exit 130",
        ),
        7.7,
        5.5,
        8,
    )

    add_route(
        out, routes,
        "exe-kbi",
        [
            exe.a("left"),
            kbi.a("right", TARGET_GAP),
        ],
        "edge-hil",
        ignore=("exe", "kbi"),
    )

    branch_label(
        out,
        (exe.left + kbi.right) / 2,
        exe.cy - 6,
        "interrupt",
        COL["hil"],
        size=6.9,
    )

    bug = Rg(
        "bug",
        795, 828,
        215, 52,
    )

    draw_node(
        out,
        bug,
        COL["red"],
        "Unexpected wrapper error",
        (
            "print error -> exit 1",
        ),
        7.9,
        5.7,
        8,
    )

    add_route(
        out, routes,
        "exe-bug",
        [
            exe.a("right"),
            bug.a("left", TARGET_GAP),
        ],
        "edge-hil",
        ignore=("exe", "bug"),
    )

    branch_label(
        out,
        (exe.right + bug.left) / 2,
        exe.cy - 6,
        "exception",
        COL["hil"],
        size=6.9,
    )

    ret = Rg(
        "ret",
        250, 950,
        500, 52,
    )

    draw_node(
        out,
        ret,
        COL["purple"],
        "Process exits with executor ExitCode",
        (
            "normal return -> sys.exit(code); typed codes 0, 2-8 or 130",
        ),
        8.9,
        6.2,
    )

    add_route(
        out, routes,
        "exe-ret",
        [
            exe.a("bottom"),
            ret.a("top", TARGET_GAP),
        ],
        ignore=("exe", "ret"),
    )

    branch_label(
        out,
        exe.cx + 10,
        exe.bottom + 11,
        "normal return",
        anchor="start",
        size=7.0,
    )

    # ==================================================================
    # Audience-facing explanatory panels
    # ==================================================================

    px, pw = 1080, 800

    panels = [
        R(px, 26, pw, 280),
        R(px, 330, pw, 315),
        R(px, 669, pw, 365),
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
            60,
            "RunPlan acquisition",
            COL["text"],
            17,
            700,
            "start",
        )
    )

    lines = [
        "The CLI can reconstruct a saved RunPlan or collect a new one interactively through the wizard.",
        "Saved presets are structurally rebuilt with RunPlan.from_dict and then checked against the current network/plugin paths.",
        "A rejected, missing or unreadable preset does not terminate the CLI; control falls back to the wizard.",
        "The RunPlan is the single configuration object passed into the executor.",
    ]

    for i, t in enumerate(lines):
        out.append(
            label(
                px + 24,
                98 + 48 * i,
                t,
                COL["text"],
                10.8,
                600,
                "start",
            )
        )

    out.append(
        label(
            px + 24,
            364,
            "Confirmation and persistence",
            COL["text"],
            17,
            700,
            "start",
        )
    )

    lines = [
        "The selected configuration is rendered as terminal tables before any network or profile loading occurs.",
        "The user can cancel with exit code 0 or continue to execution.",
        "Saving a preset is optional; an empty preset name simply leaves the RunPlan unsaved.",
        "Runtime reproducibility is additionally handled by the executor, which writes a copy of the RunPlan into the run output directory.",
    ]

    for i, t in enumerate(lines):
        out.append(
            label(
                px + 24,
                402 + 49 * i,
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
            703,
            "Executor boundary",
            COL["text"],
            17,
            700,
            "start",
        )
    )

    lines = [
        "__main__ delegates runtime work to executor.execute(plan) rather than loading networks or running scenarios itself.",
        "A normal executor return is passed directly to sys.exit, preserving the executor's typed process status.",
        "KeyboardInterrupt caught by the outer safety net exits with 130; an unexpected wrapper-layer exception is reported and exits with 1.",
    ]

    for i, t in enumerate(lines):
        out.append(
            label(
                px + 24,
                742 + 58 * i,
                t,
                COL["text"],
                10.8,
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

    assert abs(load.cy - dp.cy) < 1e-9
    assert abs(cancel.cy - dgo.cy) < 1e-9
    assert abs(save.cy - dsave.cy) < 1e-9

    # Exception routes are true horizontal side branches.
    assert abs(kbi.cy - exe.cy) < 1e-9
    assert abs(bug.cy - exe.cy) < 1e-9

    assert ret.bottom > 995

    write(
        "flow_cli_entry_presentation_v2",
        W,
        H,
        "HIL CLI top-level entry - presentation",
        "\n".join(out),
    )

def build_wizard_ieee():
    W, H = 1080, 860

    out = [
        rect_svg(
            R(14,14,W-28,H-28),
            COL["panel"],
            16,
            COL["panel_border"],
            1.5,
        )
    ]

    nodes, routes = {}, []

    def Rg(n,x,y,w,h):
        g = R(x,y,w,h)
        nodes[n] = g
        return g

    def Dg(n,x,y,w,h):
        g = D(x,y,w,h)
        nodes[n] = g
        return g

    def Cg(n,x,y,r):
        g = C(x,y,r)
        nodes[n] = g
        return g

    cx = 560

    # Keep the actual while-loop return rail separate from the local
    # BackRequested-handler bypass.
    loop_x = 12
    back_no_x = 35
    done_x = 1015

    out.append(
        label(
            W/2,
            38,
            "CLI Wizard - Index-Walked RunPlan Construction",
            COL["text"],
            17,
            700,
        )
    )

    # ==============================================================
    # Entry and ordered callback list
    # ==============================================================

    start = Rg(
        "start",
        250,60,
        620,58,
    )

    draw_node(
        out,
        start,
        COL["blue"],
        "run_wizard()",
        (
            "state = {}",
        ),
        12.0,
        8.3,
    )

    init = Rg(
        "init",
        210,145,
        700,112,
    )

    draw_node(
        out,
        init,
        COL["green"],
        "Define 9 wizard steps and set index = 0",
        (
            "Study -> Network -> Network modifications -> Dataset -> Time window",
            "Parameters -> Hardware -> Streaming -> Controller plugin",
            "each callback writes its result into the shared state dict",
        ),
        10.4,
        7.3,
    )

    add_route(
        out,
        routes,
        "s-init",
        [
            start.a("bottom"),
            init.a("top",TARGET_GAP),
        ],
        ignore=("start","init"),
    )

    # ==============================================================
    # Actual while condition
    # ==============================================================

    dloop = Dg(
        "dloop",
        cx,330,
        250,58,
    )

    draw_decision(
        out,
        dloop,
        "index < len(steps)?",
        ts=10.7,
    )

    add_route(
        out,
        routes,
        "init-loop",
        [
            init.a("bottom"),
            dloop.a("top",TARGET_GAP),
        ],
        ignore=("init","dloop"),
    )

    # ==============================================================
    # Current callback
    # ==============================================================

    step = Rg(
        "step",
        250,390,
        620,100,
    )

    draw_node(
        out,
        step,
        COL["green"],
        "name, fn = steps[index]; try: fn(state)",
        (
            "print Step i/9 rule, then run the step-specific prompts and validation",
            "plugin network: Dataset step inserts DatasetConfig('plugin') instead of prompting",
        ),
        10.3,
        7.3,
    )

    add_route(
        out,
        routes,
        "loop-step",
        [
            dloop.a("bottom"),
            step.a("top",TARGET_GAP),
        ],
        "edge-gate",
        ignore=("dloop","step"),
    )

    branch_label(
        out,
        dloop.cx+14,
        dloop.bottom+14,
        "yes",
        COL["gate"],
        "start",
        8.0,
    )

    # ==============================================================
    # BackRequested exception handler
    #
    # BackRequested is NOT a Boolean decision after fn(state).
    # It is the exception edge from the callback's try block.
    # ==============================================================

    dfirst = Dg(
        "dfirst",
        155,440,
        190,58,
    )

    draw_decision(
        out,
        dfirst,
        "index == 0?",
        ts=9.7,
    )

    # dfirst is centre-aligned with the callback, so this is a true
    # horizontal side branch.
    add_route(
        out,
        routes,
        "step-backreq",
        [
            step.a("left"),
            dfirst.a("right",TARGET_GAP),
        ],
        "edge-loop",
        ignore=("step","dfirst"),
    )

    branch_label(
        out,
        (step.left + dfirst.right)/2,
        step.cy-9,
        "BackRequested",
        COL["loop"],
        size=7.5,
    )

    # --------------------------------------------------------------
    # index == 0
    # --------------------------------------------------------------

    firstmsg = Rg(
        "firstmsg",
        55,490,
        200,48,
    )

    draw_node(
        out,
        firstmsg,
        COL["neutral"],
        "Print first-step notice",
        (
            "Already at the first step",
        ),
        8.2,
        5.9,
        8,
    )

    add_route(
        out,
        routes,
        "first-yes",
        [
            dfirst.a("bottom"),
            firstmsg.a("top",TARGET_GAP),
        ],
        ignore=("dfirst","firstmsg"),
    )

    branch_label(
        out,
        dfirst.cx+11,
        dfirst.bottom+11,
        "yes",
        COL["gate"],
        "start",
        7.3,
    )

    # Both branches of index == 0 converge before the common
    # index-decrement operation.
    bmerge = Cg(
        "bmerge",
        155,560,
        9,
    )

    out.append(
        circle_svg(
            bmerge,
            COL["neutral"],
        )
    )

    add_route(
        out,
        routes,
        "msg-bmerge",
        [
            firstmsg.a("bottom"),
            bmerge.a("top",TARGET_GAP),
        ],
        ignore=("firstmsg","bmerge"),
    )

    # index != 0 skips the informational message.
    # This local rail is distinct from the far-left while-loop rail.
    add_route(
        out,
        routes,
        "first-no",
        [
            dfirst.a("left"),
            (back_no_x,dfirst.cy),
            (back_no_x,bmerge.cy),
            bmerge.a("left",TARGET_GAP),
        ],
        ignore=("dfirst","bmerge"),
    )

    branch_label(
        out,
        dfirst.left-18,
        dfirst.cy-8,
        "no",
        anchor="end",
        size=7.3,
    )

    back = Rg(
        "back",
        55,596,
        200,58,
    )

    draw_node(
        out,
        back,
        COL["neutral"],
        "index = max(index - 1, 0)",
        (
            "continue -> previous step / re-ask step 1",
        ),
        8.4,
        6.0,
        8,
    )

    add_route(
        out,
        routes,
        "bmerge-back",
        [
            bmerge.a("bottom"),
            back.a("top",TARGET_GAP),
        ],
        ignore=("bmerge","back"),
    )

    # ==============================================================
    # Successful callback
    #
    # This is normal fall-through from try: fn(state), not the "no"
    # side of a fictitious BackRequested decision.
    # ==============================================================

    inc = Rg(
        "inc",
        430,535,
        260,58,
    )

    draw_node(
        out,
        inc,
        COL["neutral"],
        "index += 1",
        (
            "normal return from callback",
        ),
        9.3,
        6.7,
        9,
    )

    add_route(
        out,
        routes,
        "step-inc",
        [
            step.a("bottom"),
            inc.a("top",TARGET_GAP),
        ],
        ignore=("step","inc"),
    )

    branch_label(
        out,
        step.cx+14,
        step.bottom+14,
        "normal return",
        anchor="start",
        size=7.4,
    )

    # ==============================================================
    # Common while-loop re-entry
    #
    # Successful callbacks and handled BackRequested navigation both
    # reach one merge circle. Only one dashed loopback returns to dloop.
    # ==============================================================

    reentry = Cg(
        "reentry",
        560,625,
        10,
    )

    out.append(
        circle_svg(
            reentry,
            COL["neutral"],
        )
    )

    add_route(
        out,
        routes,
        "inc-reentry",
        [
            inc.a("bottom"),
            reentry.a("top",TARGET_GAP),
        ],
        ignore=("inc","reentry"),
    )

    # Align back.cy == reentry.cy so this becomes a straight side entry.
    add_route(
        out,
        routes,
        "back-reentry",
        [
            back.a("right"),
            reentry.a("left",TARGET_GAP),
        ],
        ignore=("back","reentry"),
    )

    # Leave the merge from the bottom so the dashed iteration rail does
    # not overlap the horizontal BackRequested-handler connection.
    add_route(
        out,
        routes,
        "reentry-loop",
        [
            reentry.a("bottom"),
            (reentry.cx,680),
            (loop_x,680),
            (loop_x,dloop.cy),
            dloop.a("left",TARGET_GAP),
        ],
        "edge-loop",
        ignore=("reentry","dloop"),
    )

    branch_label(
        out,
        loop_x+10,
        670,
        "next loop test",
        COL["loop"],
        "start",
        7.4,
    )

    # ==============================================================
    # Loop completion and RunPlan construction
    # ==============================================================

    ret = Rg(
        "ret",
        245,715,
        630,96,
    )

    draw_node(
        out,
        ret,
        COL["purple"],
        "Return RunPlan(...) from state",
        (
            "study; NetworkConfig; DatasetConfig; ParameterConfig; hardware/port",
            "stream_every_k; controller plugin; hc_stressed; time_period/time_index",
            "run_id/output_dir and other omitted fields use RunPlan dataclass defaults",
        ),
        10.1,
        7.1,
    )

    add_route(
        out,
        routes,
        "loop-ret",
        [
            dloop.a("right"),
            (done_x,dloop.cy),
            (done_x,ret.cy),
            ret.a("right",TARGET_GAP),
        ],
        "edge-gate",
        ignore=("dloop","ret"),
    )

    branch_label(
        out,
        dloop.right+30,
        dloop.cy-8,
        "no - all 9 steps complete",
        COL["gate"],
        "start",
        7.7,
    )

    # ==============================================================
    # Geometry checks
    # ==============================================================

    audit_bounds(nodes,W,H,8)
    audit_node_overlaps(nodes)
    audit_routes(routes,nodes)

    assert loop_x < back_no_x < dfirst.left

    # BackRequested edge is truly horizontal.
    assert abs(step.cy - dfirst.cy) < 1e-9

    # Back handler enters the common re-entry circle horizontally.
    assert abs(back.cy - reentry.cy) < 1e-9

    assert H-ret.bottom <= 55

    write(
        "flow_cli_wizard_ieee_v2",
        W,
        H,
        "CLI wizard RunPlan construction - IEEE",
        "\n".join(out),
    )

def build_wizard_presentation():
    W, H = 1920, 1080

    out = [
        rect_svg(
            R(0,0,W,H),
            COL["panel"],
            0,
        )
    ]

    nodes, routes = {}, []

    def Rg(n,x,y,w,h):
        g = R(x,y,w,h)
        nodes[n] = g
        return g

    def Dg(n,x,y,w,h):
        g = D(x,y,w,h)
        nodes[n] = g
        return g

    def Cg(n,x,y,r):
        g = C(x,y,r)
        nodes[n] = g
        return g

    cx = 500
    loop_x = 10
    done_x = 1000

    # ==============================================================
    # Entry and callback list
    # ==============================================================

    start = Rg(
        "start",
        155,24,
        690,48,
    )

    draw_node(
        out,
        start,
        COL["blue"],
        "run_wizard(): index-walked configuration loop",
        (
            "one RunPlan is assembled from step results",
        ),
        10.8,
        7.3,
    )

    init = Rg(
        "init",
        145,112,
        710,84,
    )

    draw_node(
        out,
        init,
        COL["green"],
        "Create state + ordered 9-step callback list",
        (
            "Study | Network | Network mods | Dataset | Time | Parameters | Hardware | Streaming | Controller plugin",
        ),
        9.5,
        6.6,
    )

    add_route(
        out,
        routes,
        "s-i",
        [
            start.a("bottom"),
            init.a("top",TARGET_GAP),
        ],
        ignore=("start","init"),
    )

    # ==============================================================
    # Actual while condition
    # ==============================================================

    dloop = Dg(
        "dloop",
        cx,300,
        230,42,
    )

    draw_decision(
        out,
        dloop,
        "another step?",
        ts=9.8,
    )

    add_route(
        out,
        routes,
        "i-loop",
        [
            init.a("bottom"),
            dloop.a("top",TARGET_GAP),
        ],
        ignore=("init","dloop"),
    )

    # ==============================================================
    # Current callback
    # ==============================================================

    step = Rg(
        "step",
        200,385,
        600,80,
    )

    draw_node(
        out,
        step,
        COL["green"],
        "Run current step callback",
        (
            "name, fn = steps[index]; print Step i/9; try: fn(state)",
            "prompts update state; plugin network skips standalone dataset prompting",
        ),
        9.1,
        6.4,
    )

    add_route(
        out,
        routes,
        "loop-step",
        [
            dloop.a("bottom"),
            step.a("top",TARGET_GAP),
        ],
        "edge-gate",
        ignore=("dloop","step"),
    )

    branch_label(
        out,
        dloop.cx+11,
        dloop.bottom+11,
        "yes",
        COL["gate"],
        "start",
        7.4,
    )

    # ==============================================================
    # BackRequested exception path
    #
    # Presentation keeps the real exception edge, but compresses the
    # internal index==0 informational branch into one handler block.
    # ==============================================================

    back = Rg(
        "back",
        35,397,
        155,56,
    )

    draw_node(
        out,
        back,
        COL["neutral"],
        "BackRequested handler",
        (
            "if first step: print notice",
            "index=max(index-1,0); continue",
        ),
        7.0,
        5.0,
        8,
    )

    # Same vertical centre as the callback, giving a clean straight arrow.
    add_route(
        out,
        routes,
        "step-back",
        [
            step.a("left"),
            back.a("right",TARGET_GAP),
        ],
        "edge-loop",
        ignore=("step","back"),
    )

    branch_label(
        out,
        (step.left + back.right)/2,
        step.cy-6,
        "BackRequested",
        COL["loop"],
        size=6.4,
    )

    # ==============================================================
    # Successful callback
    # ==============================================================

    inc = Rg(
        "inc",
        330,550,
        340,52,
    )

    draw_node(
        out,
        inc,
        COL["neutral"],
        "index += 1",
        (
            "normal callback return",
        ),
        8.7,
        6.2,
        8,
    )

    add_route(
        out,
        routes,
        "step-inc",
        [
            step.a("bottom"),
            inc.a("top",TARGET_GAP),
        ],
        ignore=("step","inc"),
    )

    branch_label(
        out,
        step.cx+10,
        step.bottom+11,
        "normal return",
        anchor="start",
        size=6.9,
    )

    # ==============================================================
    # Shared while-loop re-entry
    # ==============================================================

    reentry = Cg(
        "reentry",
        cx,700,
        8,
    )

    out.append(
        circle_svg(
            reentry,
            COL["neutral"],
        )
    )

    add_route(
        out,
        routes,
        "inc-reentry",
        [
            inc.a("bottom"),
            reentry.a("top",TARGET_GAP),
        ],
        ignore=("inc","reentry"),
    )

    add_route(
        out,
        routes,
        "back-reentry",
        [
            back.a("bottom"),
            (back.cx,reentry.cy),
            reentry.a("left",TARGET_GAP),
        ],
        ignore=("back","reentry"),
    )

    # Leave the re-entry merge from the bottom. This keeps the dashed
    # iteration rail separate from the BackRequested handler's dark route.
    add_route(
        out,
        routes,
        "reentry-loop",
        [
            reentry.a("bottom"),
            (reentry.cx,755),
            (loop_x,755),
            (loop_x,dloop.cy),
            dloop.a("left",TARGET_GAP),
        ],
        "edge-loop",
        ignore=("reentry","dloop"),
    )

    branch_label(
        out,
        loop_x+10,
        745,
        "next loop test",
        COL["loop"],
        "start",
        6.8,
    )

    # ==============================================================
    # Loop completion
    # ==============================================================

    ret = Rg(
        "ret",
        190,930,
        620,64,
    )

    draw_node(
        out,
        ret,
        COL["purple"],
        "Assemble + return RunPlan",
        (
            "typed nested configs + run-control fields; dataclass defaults supply run_id/output_dir",
        ),
        9.3,
        6.6,
    )

    add_route(
        out,
        routes,
        "loop-ret",
        [
            dloop.a("right"),
            (done_x,dloop.cy),
            (done_x,ret.cy),
            ret.a("right",TARGET_GAP),
        ],
        "edge-gate",
        ignore=("dloop","ret"),
    )

    branch_label(
        out,
        dloop.right+24,
        dloop.cy-6,
        "no - all 9 complete",
        COL["gate"],
        "start",
        7.2,
    )

    # ==============================================================
    # Audience-facing notes
    # ==============================================================

    px, pw = 1080, 800

    panels = [
        R(px,24,pw,280),
        R(px,326,pw,320),
        R(px,668,pw,370),
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
            px+24,
            57,
            "Nine configuration steps",
            COL["text"],
            17,
            700,
            "start",
        )
    )

    lines = [
        "Study chooses scenario comparison, voltage variation, hosting capacity or OPF; hosting capacity may additionally request the stressed re-benchmark.",
        "Network supports curated presets, assembled SimBench codes, custom Python factories and network-plugin YAML.",
        "Network modifications optionally inject PV sgens and flip selected switches before profile construction.",
        "Dataset is SimBench native, DWD or custom for non-plugin networks; a network plugin supplies its own profile strategy and therefore skips this prompt.",
        "The remaining steps select time window, limits/scaling/Q(V), dry-run or hardware mode, live-stream cadence and an optional controller plugin.",
    ]

    for i,t in enumerate(lines):
        out.append(
            label(
                px+24,
                94+39*i,
                t,
                COL["text"],
                10.5,
                600,
                "start",
            )
        )

    out.append(
        label(
            px+24,
            359,
            "Navigation and validation",
            COL["text"],
            17,
            700,
            "start",
        )
    )

    lines = [
        "Menus expose 0 as Back; free-text prompts use '<'. Both raise BackRequested to the central loop.",
        "BackRequested is caught around the current callback; the index is decremented with a floor at zero and the while-loop is re-tested.",
        "At index 0 the handler prints an already-at-first-step notice, then re-enters Step 1; successful callbacks increment the index.",
        "Prompt-level validation catches ranges and input syntax here; cross-field compatibility and filesystem/runtime checks remain the executor's responsibility.",
    ]

    for i,t in enumerate(lines):
        out.append(
            label(
                px+24,
                396+50*i,
                t,
                COL["text"],
                10.6,
                600,
                "start",
            )
        )

    out.append(
        label(
            px+24,
            701,
            "RunPlan data model",
            COL["text"],
            17,
            700,
            "start",
        )
    )

    lines = [
        "RunPlan nests NetworkConfig, DatasetConfig and ParameterConfig and adds study, hardware, controller-plugin, HC, time-window and output fields.",
        "The wizard supplies the selected configuration fields; dataclass defaults create the fresh run_id and default output_dir for a newly assembled RunPlan.",
        "to_dict()/from_dict() provide the JSON serialization boundary used by saved presets and the executor's reproducibility copy.",
        "The data classes are pure configuration containers; runtime decisions remain in wizard.py and executor.py.",
    ]

    for i,t in enumerate(lines):
        out.append(
            label(
                px+24,
                742+52*i,
                t,
                COL["text"],
                10.5,
                600,
                "start",
            )
        )

    # ==============================================================
    # Geometry checks
    # ==============================================================

    audit_bounds(nodes,W,H,0)
    audit_node_overlaps(nodes)
    audit_routes(routes,nodes)

    assert loop_x < back.left

    # BackRequested side route is a true horizontal connection.
    assert abs(back.cy - step.cy) < 1e-9

    assert ret.bottom > 975

    write(
        "flow_cli_wizard_presentation_v2",
        W,
        H,
        "CLI wizard RunPlan construction - presentation",
        "\n".join(out),
    )

def build_executor_ieee():
    W, H = 1360, 1460

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

    cx = 680

    out.append(
        label(
            W / 2,
            38,
            "CLI Executor - Validation, Resolution, Benchmark and Typed Exit Codes",
            COL["text"],
            17,
            700,
        )
    )

    # ==================================================================
    # Entry
    # ==================================================================

    start = Rg(
        "start",
        370, 60,
        620, 62,
    )

    draw_node(
        out,
        start,
        COL["blue"],
        "execute(plan)",
        (
            "start timer; out_dir = output_dir/run_id; configure Rich + session.log",
        ),
        11.8,
        8.2,
    )

    # ==================================================================
    # Phase 0
    # ==================================================================

    p0 = Rg(
        "p0",
        340, 150,
        680, 92,
    )

    draw_node(
        out,
        p0,
        COL["green"],
        "Phase 0 - validate configuration + apply runtime channels",
        (
            "ensure framework path; validate_plan; Q(V) overrides; violation limits",
            "hardware-port pre-check; optional controller-plugin firmware confirmation",
        ),
        10.4,
        7.4,
    )

    add_route(
        out, routes,
        "s-p0",
        [
            start.a("bottom"),
            p0.a("top", TARGET_GAP),
        ],
        ignore=("start", "p0"),
    )

    # Exactly centre-aligned with p0.
    e0 = Rg(
        "e0",
        35, 160,
        255, 72,
    )

    draw_node(
        out,
        e0,
        COL["red"],
        "Phase-0 terminal return",
        (
            "ExecutorError -> its typed exit code",
            "KeyboardInterrupt -> INTERRUPTED (130)",
        ),
        8.3,
        6.0,
        8,
    )

    add_route(
        out, routes,
        "p0-e0",
        [
            p0.a("left"),
            e0.a("right", TARGET_GAP),
        ],
        "edge-hil",
        ignore=("p0", "e0"),
    )

    branch_label(
        out,
        (p0.left + e0.right) / 2,
        p0.cy - 8,
        "caught exit",
        COL["hil"],
        size=7.3,
    )

    # ==================================================================
    # Phase 1
    # ==================================================================

    p1 = Rg(
        "p1",
        340, 280,
        680, 80,
    )

    draw_node(
        out,
        p1,
        COL["green"],
        "Phase 1 - build_net_and_profiles(plan)",
        (
            "resolve network + profiles + network_id + HC profile_factory",
            "typed loader/dataset failures; unexpected crash -> NETWORK_LOAD_ERROR",
        ),
        10.5,
        7.4,
    )

    add_route(
        out, routes,
        "p0-p1",
        [
            p0.a("bottom"),
            p1.a("top", TARGET_GAP),
        ],
        ignore=("p0", "p1"),
    )

    e1 = Rg(
        "e1",
        35, 284,
        255, 72,
    )

    draw_node(
        out,
        e1,
        COL["red"],
        "Phase-1 terminal return",
        (
            "ExecutorError -> its typed exit code",
            "unexpected crash -> NETWORK_LOAD_ERROR; interrupt -> 130",
        ),
        8.1,
        5.9,
        8,
    )

    add_route(
        out, routes,
        "p1-e1",
        [
            p1.a("left"),
            e1.a("right", TARGET_GAP),
        ],
        "edge-hil",
        ignore=("p1", "e1"),
    )

    branch_label(
        out,
        (p1.left + e1.right) / 2,
        p1.cy - 8,
        "caught exit",
        COL["hil"],
        size=7.3,
    )

    # ==================================================================
    # Phase 2
    # ==================================================================

    p2 = Rg(
        "p2",
        330, 400,
        700, 100,
    )

    draw_node(
        out,
        p2,
        COL["blue"],
        "Phase 2 - configure benchmark + publishers",
        (
            "create outer and HC-stressed PublishHandle(update_every_k=stream_every_k)",
            "build BenchmarkConfig; attach hc_publish_fn; save run_plan.json",
        ),
        10.4,
        7.3,
    )

    add_route(
        out, routes,
        "p1-p2",
        [
            p1.a("bottom"),
            p2.a("top", TARGET_GAP),
        ],
        ignore=("p1", "p2"),
    )

    e2 = Rg(
        "e2",
        35, 414,
        255, 72,
    )

    draw_node(
        out,
        e2,
        COL["red"],
        "Phase-2 terminal return",
        (
            "ExecutorError -> its typed exit code",
            "KeyboardInterrupt -> INTERRUPTED (130)",
        ),
        8.3,
        6.0,
        8,
    )

    add_route(
        out, routes,
        "p2-e2",
        [
            p2.a("left"),
            e2.a("right", TARGET_GAP),
        ],
        "edge-hil",
        ignore=("p2", "e2"),
    )

    branch_label(
        out,
        (p2.left + e2.right) / 2,
        p2.cy - 8,
        "caught exit",
        COL["hil"],
        size=7.3,
    )

    # ==================================================================
    # Pre-simulation topology/profile publication
    #
    # Failure is explicitly non-fatal.
    # ==================================================================

    prepub = Rg(
        "prepub",
        345, 540,
        670, 72,
    )

    draw_node(
        out,
        prepub,
        COL["purple"],
        "Publish topology + profiles before simulation",
        (
            "failure is warning-only; benchmark still starts",
        ),
        10.5,
        7.5,
    )

    add_route(
        out, routes,
        "p2-prepub",
        [
            p2.a("bottom"),
            prepub.a("top", TARGET_GAP),
        ],
        ignore=("p2", "prepub"),
    )

    # Neutral rather than fatal red because execution continues.
    warn = Rg(
        "warn",
        55, 545,
        235, 62,
    )

    draw_node(
        out,
        warn,
        COL["neutral"],
        "Pre-publish warning",
        (
            "exception logged; continue to benchmark",
        ),
        8.4,
        6.1,
        8,
    )

    add_route(
        out, routes,
        "pre-warn",
        [
            prepub.a("left"),
            warn.a("right", TARGET_GAP),
        ],
        "edge-hil",
        ignore=("prepub", "warn"),
    )

    branch_label(
        out,
        (prepub.left + warn.right) / 2,
        prepub.cy - 8,
        "exception",
        COL["hil"],
        size=7.4,
    )

    pm = Cg(
        "pm",
        cx, 650,
        10,
    )

    out.append(
        circle_svg(
            pm,
            COL["neutral"],
        )
    )

    add_route(
        out, routes,
        "pre-pm",
        [
            prepub.a("bottom"),
            pm.a("top", TARGET_GAP),
        ],
        ignore=("prepub", "pm"),
    )

    add_route(
        out, routes,
        "warn-pm",
        [
            warn.a("bottom"),
            (warn.cx, pm.cy),
            pm.a("left", TARGET_GAP),
        ],
        "edge-hil",
        ignore=("warn", "pm"),
    )

    # ==================================================================
    # Phase 3 - run dispatch
    #
    # Decision, runner blocks and exception blocks are all horizontally
    # aligned. Normal returns flow DOWN toward rm. Exceptions leave
    # OUTWARD toward terminal return blocks.
    # ==================================================================

    dplug = Dg(
        "dplug",
        cx, 750,
        260, 60,
    )

    draw_decision(
        out,
        dplug,
        "controller_plugin_path set?",
        ts=10.2,
    )

    add_route(
        out, routes,
        "pm-dplug",
        [
            pm.a("bottom"),
            dplug.a("top", TARGET_GAP),
        ],
        ignore=("pm", "dplug"),
    )

    preg = Rg(
        "preg",
        220, 714,
        300, 72,
    )

    draw_node(
        out,
        preg,
        COL["green"],
        "plugin_runner.register_and_run(...)",
        (
            "return_benchmark=True; port only in hardware mode",
        ),
        9.2,
        6.7,
        9,
    )

    bench = Rg(
        "bench",
        840, 714,
        300, 72,
    )

    draw_node(
        out,
        bench,
        COL["blue"],
        "benchmark_runner.run_benchmark(...)",
        (
            "standard built-in orchestration",
        ),
        9.3,
        6.8,
        9,
    )

    add_route(
        out, routes,
        "plug-reg",
        [
            dplug.a("left"),
            preg.a("right", TARGET_GAP),
        ],
        "edge-gate",
        ignore=("dplug", "preg"),
    )

    branch_label(
        out,
        (dplug.left + preg.right) / 2,
        dplug.cy - 8,
        "yes",
        COL["gate"],
        size=7.8,
    )

    add_route(
        out, routes,
        "plug-bench",
        [
            dplug.a("right"),
            bench.a("left", TARGET_GAP),
        ],
        ignore=("dplug", "bench"),
    )

    branch_label(
        out,
        (dplug.right + bench.left) / 2,
        dplug.cy - 8,
        "no",
        size=7.8,
    )

    # --------------------------------------------------------------
    # Phase-3 terminal returns
    #
    # These do NOT continue to Phase 4.
    # --------------------------------------------------------------

    plugerr = Rg(
        "plugerr",
        20, 710,
        180, 80,
    )

    draw_node(
        out,
        plugerr,
        COL["red"],
        "Plugin-path terminal return",
        (
            "definition/import -> PLUGIN_ERROR (5)",
            "RuntimeError/other crash -> SIMULATION_ERROR (7)",
            "KeyboardInterrupt -> 130",
        ),
        7.4,
        5.3,
        8,
    )

    add_route(
        out, routes,
        "reg-err",
        [
            preg.a("left"),
            plugerr.a("right", TARGET_GAP),
        ],
        "edge-hil",
        ignore=("preg", "plugerr"),
    )

    branch_label(
        out,
        (preg.left + plugerr.right) / 2,
        preg.cy - 8,
        "exception",
        COL["hil"],
        size=7.0,
    )

    bencherr = Rg(
        "bencherr",
        1160, 710,
        180, 80,
    )

    draw_node(
        out,
        bencherr,
        COL["red"],
        "Built-in path terminal return",
        (
            "definition-like types -> PLUGIN_ERROR (5)",
            "RuntimeError/other crash -> SIMULATION_ERROR (7)",
            "KeyboardInterrupt -> 130",
        ),
        7.2,
        5.2,
        8,
    )

    add_route(
        out, routes,
        "bench-err",
        [
            bench.a("right"),
            bencherr.a("left", TARGET_GAP),
        ],
        "edge-hil",
        ignore=("bench", "bencherr"),
    )

    branch_label(
        out,
        (bench.right + bencherr.left) / 2,
        bench.cy - 8,
        "exception",
        COL["hil"],
        size=7.0,
    )

    # --------------------------------------------------------------
    # Only successful Phase-3 returns merge here.
    # --------------------------------------------------------------

    rm = Cg(
        "rm",
        cx, 845,
        11,
    )

    out.append(
        circle_svg(
            rm,
            COL["neutral"],
        )
    )

    add_route(
        out, routes,
        "reg-rm",
        [
            preg.a("bottom"),
            (preg.cx, rm.cy),
            rm.a("left", TARGET_GAP),
        ],
        ignore=("preg", "rm"),
    )

    add_route(
        out, routes,
        "bench-rm",
        [
            bench.a("bottom"),
            (bench.cx, rm.cy),
            rm.a("right", TARGET_GAP),
        ],
        ignore=("bench", "rm"),
    )

    # ==================================================================
    # Phase 4 - publishing
    #
    # Unlike Phases 0-3, exceptions here alter exit_code but execution
    # CONTINUES through the final summary.
    # ==================================================================

    p4 = Rg(
        "p4",
        340, 885,
        680, 92,
    )

    draw_node(
        out,
        p4,
        COL["purple"],
        "Phase 4 - publish benchmark results",
        (
            "publish outer comparison/HC artefacts",
            "if hc_benchmark + net_hc exist: publish HC-stressed artefacts too",
        ),
        10.4,
        7.3,
    )

    add_route(
        out, routes,
        "rm-p4",
        [
            rm.a("bottom"),
            p4.a("top", TARGET_GAP),
        ],
        ignore=("rm", "p4"),
    )

    # Centre-aligned with p4.
    p4int = Rg(
        "p4int",
        55, 895,
        235, 72,
    )

    draw_node(
        out,
        p4int,
        COL["red"],
        "Publishing interrupted",
        (
            "set exit_code=INTERRUPTED (130)",
            "continue to final summary",
        ),
        8.2,
        5.9,
        8,
    )

    add_route(
        out, routes,
        "p4-int",
        [
            p4.a("left"),
            p4int.a("right", TARGET_GAP),
        ],
        "edge-hil",
        ignore=("p4", "p4int"),
    )

    branch_label(
        out,
        (p4.left + p4int.right) / 2,
        p4.cy - 8,
        "KeyboardInterrupt",
        COL["hil"],
        size=7.0,
    )

    p4err = Rg(
        "p4err",
        1070, 895,
        235, 72,
    )

    draw_node(
        out,
        p4err,
        COL["red"],
        "Publishing exception",
        (
            "set exit_code=PUBLISH_ERROR (8)",
            "continue to final summary",
        ),
        8.3,
        6.0,
        8,
    )

    add_route(
        out, routes,
        "p4-err",
        [
            p4.a("right"),
            p4err.a("left", TARGET_GAP),
        ],
        "edge-hil",
        ignore=("p4", "p4err"),
    )

    branch_label(
        out,
        (p4.right + p4err.left) / 2,
        p4.cy - 8,
        "exception",
        COL["hil"],
        size=7.2,
    )

    p4m = Cg(
        "p4m",
        cx, 1025,
        10,
    )

    out.append(
        circle_svg(
            p4m,
            COL["neutral"],
        )
    )

    # Normal publication.
    add_route(
        out, routes,
        "p4-m",
        [
            p4.a("bottom"),
            p4m.a("top", TARGET_GAP),
        ],
        ignore=("p4", "p4m"),
    )

    # Interrupted publication.
    add_route(
        out, routes,
        "p4int-m",
        [
            p4int.a("bottom"),
            (p4int.cx, p4m.cy),
            p4m.a("left", TARGET_GAP),
        ],
        "edge-hil",
        ignore=("p4int", "p4m"),
    )

    # Publishing exception.
    add_route(
        out, routes,
        "p4err-m",
        [
            p4err.a("bottom"),
            (p4err.cx, p4m.cy),
            p4m.a("right", TARGET_GAP),
        ],
        "edge-hil",
        ignore=("p4err", "p4m"),
    )

    # ==================================================================
    # Final summary and post-run scenario-failure promotion
    # ==================================================================

    summary = Rg(
        "summary",
        340, 1065,
        680, 76,
    )

    draw_node(
        out,
        summary,
        COL["purple"],
        "Build + print final summary table",
        (
            "elapsed time, scenario statuses, HC metrics, CSV/published paths",
        ),
        10.5,
        7.5,
    )

    add_route(
        out, routes,
        "m-summary",
        [
            p4m.a("bottom"),
            summary.a("top", TARGET_GAP),
        ],
        ignore=("p4m", "summary"),
    )

    dfail = Dg(
        "dfail",
        cx, 1200,
        300, 62,
    )

    draw_decision(
        out,
        dfail,
        "exit_code == OK AND real scenario failures?",
        ts=9.7,
    )

    add_route(
        out, routes,
        "sum-d",
        [
            summary.a("bottom"),
            dfail.a("top", TARGET_GAP),
        ],
        ignore=("summary", "dfail"),
    )

    mark = Rg(
        "mark",
        1000, 1167,
        300, 66,
    )

    draw_node(
        out,
        mark,
        COL["red"],
        "Promote exit status to SIMULATION_ERROR",
        (
            "comparison table remains visible",
        ),
        8.8,
        6.5,
        8,
    )

    add_route(
        out, routes,
        "d-mark",
        [
            dfail.a("right"),
            mark.a("left", TARGET_GAP),
        ],
        "edge-hil",
        ignore=("dfail", "mark"),
    )

    branch_label(
        out,
        (dfail.right + mark.left) / 2,
        dfail.cy - 8,
        "yes",
        COL["hil"],
        size=7.6,
    )

    em = Cg(
        "em",
        cx, 1290,
        10,
    )

    out.append(
        circle_svg(
            em,
            COL["neutral"],
        )
    )

    add_route(
        out, routes,
        "d-em",
        [
            dfail.a("bottom"),
            em.a("top", TARGET_GAP),
        ],
        ignore=("dfail", "em"),
    )

    branch_label(
        out,
        dfail.cx + 14,
        dfail.bottom + 14,
        "no",
        anchor="start",
        size=7.6,
    )

    add_route(
        out, routes,
        "mark-em",
        [
            mark.a("bottom"),
            (mark.cx, em.cy),
            em.a("right", TARGET_GAP),
        ],
        "edge-hil",
        ignore=("mark", "em"),
    )

    # Only statuses that can actually reach this post-summary return are
    # listed here. Codes 2-6 and Phase-3 terminal 5/7/130 return earlier.
    ret = Rg(
        "ret",
        400, 1330,
        560, 64,
    )

    draw_node(
        out,
        ret,
        COL["purple"],
        "return int(exit_code)",
        (
            "post-run path: OK=0; SIMULATION_ERROR=7; "
            "PUBLISH_ERROR=8; INTERRUPTED=130",
        ),
        9.4,
        6.7,
    )

    add_route(
        out, routes,
        "em-ret",
        [
            em.a("bottom"),
            ret.a("top", TARGET_GAP),
        ],
        ignore=("em", "ret"),
    )

    # ==================================================================
    # Geometry checks
    # ==================================================================

    audit_bounds(nodes, W, H, 8)
    audit_node_overlaps(nodes)
    audit_routes(routes, nodes)

    # Phase-0/1/2 caught exits are truly horizontal.
    assert abs(e0.cy - p0.cy) < 1e-9
    assert abs(e1.cy - p1.cy) < 1e-9
    assert abs(e2.cy - p2.cy) < 1e-9

    # Pre-publish warning is horizontally aligned.
    assert abs(warn.cy - prepub.cy) < 1e-9

    # Phase-3 dispatch and terminal exception blocks share one row.
    assert abs(preg.cy - dplug.cy) < 1e-9
    assert abs(bench.cy - dplug.cy) < 1e-9
    assert abs(plugerr.cy - preg.cy) < 1e-9
    assert abs(bencherr.cy - bench.cy) < 1e-9

    # Phase-4 continuation outcomes are horizontally aligned.
    assert abs(p4int.cy - p4.cy) < 1e-9
    assert abs(p4err.cy - p4.cy) < 1e-9

    assert H - ret.bottom <= 70

    write(
        "flow_cli_executor_ieee_v2",
        W,
        H,
        "CLI executor phases and typed exits - IEEE",
        "\n".join(out),
    )

def build_executor_presentation():
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

    cx = 520

    # ==================================================================
    # Executor entry
    # ==================================================================

    start = Rg(
        "start",
        180, 18,
        680, 44,
    )

    draw_node(
        out,
        start,
        COL["blue"],
        "executor.execute(plan)",
        (
            "typed runtime boundary between CLI configuration and benchmark framework",
        ),
        10.8,
        7.3,
    )

    # ==================================================================
    # Phase 0
    # ==================================================================

    p0 = Rg(
        "p0",
        220, 80,
        600, 54,
    )

    draw_node(
        out,
        p0,
        COL["green"],
        "Phase 0 - validate + runtime overrides",
        (
            "plan compatibility, Q(V)/limits, hardware and firmware pre-checks",
        ),
        9.2,
        6.5,
    )

    e0 = Rg(
        "e0",
        20, 84,
        170, 46,
    )

    draw_node(
        out,
        e0,
        COL["red"],
        "Phase-0 terminal",
        (
            "typed error / interrupt 130",
        ),
        7.5,
        5.4,
        8,
    )

    add_route(
        out, routes,
        "s-p0",
        [
            start.a("bottom"),
            p0.a("top", TARGET_GAP),
        ],
        ignore=("start", "p0"),
    )

    add_route(
        out, routes,
        "p0-e0",
        [
            p0.a("left"),
            e0.a("right", TARGET_GAP),
        ],
        "edge-hil",
        ignore=("p0", "e0"),
    )

    branch_label(
        out,
        (p0.left + e0.right) / 2,
        p0.cy - 6,
        "caught exit",
        COL["hil"],
        size=6.8,
    )

    # ==================================================================
    # Phase 1
    # ==================================================================

    p1 = Rg(
        "p1",
        220, 170,
        600, 54,
    )

    draw_node(
        out,
        p1,
        COL["green"],
        "Phase 1 - network + profiles",
        (
            "build_net_and_profiles -> net, profiles, network_id, profile_factory",
        ),
        9.2,
        6.5,
    )

    e1 = Rg(
        "e1",
        20, 174,
        170, 46,
    )

    draw_node(
        out,
        e1,
        COL["red"],
        "Phase-1 terminal",
        (
            "network/dataset / interrupt",
        ),
        7.4,
        5.3,
        8,
    )

    add_route(
        out, routes,
        "p0-p1",
        [
            p0.a("bottom"),
            p1.a("top", TARGET_GAP),
        ],
        ignore=("p0", "p1"),
    )

    add_route(
        out, routes,
        "p1-e1",
        [
            p1.a("left"),
            e1.a("right", TARGET_GAP),
        ],
        "edge-hil",
        ignore=("p1", "e1"),
    )

    branch_label(
        out,
        (p1.left + e1.right) / 2,
        p1.cy - 6,
        "caught exit",
        COL["hil"],
        size=6.8,
    )

    # ==================================================================
    # Phase 2
    # ==================================================================

    p2 = Rg(
        "p2",
        220, 260,
        600, 58,
    )

    draw_node(
        out,
        p2,
        COL["blue"],
        "Phase 2 - BenchmarkConfig + PublishHandles",
        (
            "outer + HC-stressed publishers; save run_plan.json",
        ),
        9.2,
        6.5,
    )

    e2 = Rg(
        "e2",
        20, 266,
        170, 46,
    )

    draw_node(
        out,
        e2,
        COL["red"],
        "Phase-2 terminal",
        (
            "typed error / interrupt 130",
        ),
        7.5,
        5.4,
        8,
    )

    add_route(
        out, routes,
        "p1-p2",
        [
            p1.a("bottom"),
            p2.a("top", TARGET_GAP),
        ],
        ignore=("p1", "p2"),
    )

    add_route(
        out, routes,
        "p2-e2",
        [
            p2.a("left"),
            e2.a("right", TARGET_GAP),
        ],
        "edge-hil",
        ignore=("p2", "e2"),
    )

    branch_label(
        out,
        (p2.left + e2.right) / 2,
        p2.cy - 6,
        "caught exit",
        COL["hil"],
        size=6.8,
    )

    # ==================================================================
    # Pre-publish snapshot
    # ==================================================================

    pre = Rg(
        "pre",
        220, 340,
        600, 54,
    )

    draw_node(
        out,
        pre,
        COL["purple"],
        "Publish topology + profiles",
        (
            "warning-only on failure; simulation still starts",
        ),
        9.2,
        6.5,
    )

    warn = Rg(
        "warn",
        20, 344,
        170, 46,
    )

    draw_node(
        out,
        warn,
        COL["neutral"],
        "Pre-publish warning",
        (
            "log + continue",
        ),
        7.5,
        5.4,
        8,
    )

    add_route(
        out, routes,
        "p2-pre",
        [
            p2.a("bottom"),
            pre.a("top", TARGET_GAP),
        ],
        ignore=("p2", "pre"),
    )

    add_route(
        out, routes,
        "pre-warn",
        [
            pre.a("left"),
            warn.a("right", TARGET_GAP),
        ],
        "edge-hil",
        ignore=("pre", "warn"),
    )

    branch_label(
        out,
        (pre.left + warn.right) / 2,
        pre.cy - 6,
        "exception",
        COL["hil"],
        size=6.8,
    )

    pm = Cg(
        "pm",
        cx, 430,
        8,
    )

    out.append(
        circle_svg(
            pm,
            COL["neutral"],
        )
    )

    add_route(
        out, routes,
        "pre-pm",
        [
            pre.a("bottom"),
            pm.a("top", TARGET_GAP),
        ],
        ignore=("pre", "pm"),
    )

    add_route(
        out, routes,
        "warn-pm",
        [
            warn.a("bottom"),
            (warn.cx, pm.cy),
            pm.a("left", TARGET_GAP),
        ],
        "edge-hil",
        ignore=("warn", "pm"),
    )

    # ==================================================================
    # Phase 3
    #
    # Runner blocks sit on the same row as the real branch decision.
    # Normal returns descend; terminal exceptions leave outward.
    # ==================================================================

    dplug = Dg(
        "dplug",
        cx, 500,
        180, 40,
    )

    draw_decision(
        out,
        dplug,
        "controller plugin?",
        ts=9.6,
    )

    add_route(
        out, routes,
        "pm-d",
        [
            pm.a("bottom"),
            dplug.a("top", TARGET_GAP),
        ],
        ignore=("pm", "dplug"),
    )

    preg = Rg(
        "preg",
        180, 478,
        230, 44,
    )

    draw_node(
        out,
        preg,
        COL["green"],
        "register_and_run",
        (
            "plugin scenario + benchmark",
        ),
        8.5,
        6.1,
        8,
    )

    bench = Rg(
        "bench",
        630, 478,
        230, 44,
    )

    draw_node(
        out,
        bench,
        COL["blue"],
        "run_benchmark",
        (
            "built-in orchestration",
        ),
        8.6,
        6.2,
        8,
    )

    add_route(
        out, routes,
        "d-pr",
        [
            dplug.a("left"),
            preg.a("right", TARGET_GAP),
        ],
        "edge-gate",
        ignore=("dplug", "preg"),
    )

    branch_label(
        out,
        (dplug.left + preg.right) / 2,
        dplug.cy - 6,
        "yes",
        COL["gate"],
        size=7.2,
    )

    add_route(
        out, routes,
        "d-br",
        [
            dplug.a("right"),
            bench.a("left", TARGET_GAP),
        ],
        ignore=("dplug", "bench"),
    )

    branch_label(
        out,
        (dplug.right + bench.left) / 2,
        dplug.cy - 6,
        "no",
        size=7.2,
    )

    # Terminal Phase-3 outcomes.
    perr = Rg(
        "perr",
        10, 476,
        150, 48,
    )

    draw_node(
        out,
        perr,
        COL["red"],
        "Plugin path failed",
        (
            "return 5 / 7 / 130",
        ),
        7.1,
        5.1,
        8,
    )

    add_route(
        out, routes,
        "pr-err",
        [
            preg.a("left"),
            perr.a("right", TARGET_GAP),
        ],
        "edge-hil",
        ignore=("preg", "perr"),
    )

    branch_label(
        out,
        (preg.left + perr.right) / 2,
        preg.cy - 6,
        "exception",
        COL["hil"],
        size=6.5,
    )

    berr = Rg(
        "berr",
        880, 476,
        170, 48,
    )

    draw_node(
        out,
        berr,
        COL["red"],
        "Built-in path failed",
        (
            "current handlers -> 5 / 7 / 130",
        ),
        6.9,
        5.0,
        8,
    )

    add_route(
        out, routes,
        "br-err",
        [
            bench.a("right"),
            berr.a("left", TARGET_GAP),
        ],
        "edge-hil",
        ignore=("bench", "berr"),
    )

    branch_label(
        out,
        (bench.right + berr.left) / 2,
        bench.cy - 6,
        "exception",
        COL["hil"],
        size=6.5,
    )

    rm = Cg(
        "rm",
        cx, 585,
        8,
    )

    out.append(
        circle_svg(
            rm,
            COL["neutral"],
        )
    )

    # Normal returns only.
    add_route(
        out, routes,
        "pr-rm",
        [
            preg.a("bottom"),
            (preg.cx, rm.cy),
            rm.a("left", TARGET_GAP),
        ],
        ignore=("preg", "rm"),
    )

    add_route(
        out, routes,
        "br-rm",
        [
            bench.a("bottom"),
            (bench.cx, rm.cy),
            rm.a("right", TARGET_GAP),
        ],
        ignore=("bench", "rm"),
    )

    # ==================================================================
    # Phase 4
    #
    # Both publishing interruption and ordinary publishing exceptions
    # continue into the same final-summary merge.
    # ==================================================================

    p4 = Rg(
        "p4",
        220, 635,
        600, 56,
    )

    draw_node(
        out,
        p4,
        COL["purple"],
        "Phase 4 - publish result artefacts",
        (
            "outer run plus HC-stressed result when present",
        ),
        9.2,
        6.5,
    )

    add_route(
        out, routes,
        "rm-p4",
        [
            rm.a("bottom"),
            p4.a("top", TARGET_GAP),
        ],
        ignore=("rm", "p4"),
    )

    pint = Rg(
        "pint",
        20, 640,
        170, 46,
    )

    draw_node(
        out,
        pint,
        COL["red"],
        "Publish interrupted",
        (
            "status=130; continue",
        ),
        7.3,
        5.2,
        8,
    )

    add_route(
        out, routes,
        "p4-int",
        [
            p4.a("left"),
            pint.a("right", TARGET_GAP),
        ],
        "edge-hil",
        ignore=("p4", "pint"),
    )

    branch_label(
        out,
        (p4.left + pint.right) / 2,
        p4.cy - 6,
        "interrupt",
        COL["hil"],
        size=6.7,
    )

    puberr = Rg(
        "puberr",
        850, 640,
        200, 46,
    )

    draw_node(
        out,
        puberr,
        COL["red"],
        "Publishing failed",
        (
            "status=PUBLISH_ERROR; continue",
        ),
        7.3,
        5.2,
        8,
    )

    add_route(
        out, routes,
        "p4-err",
        [
            p4.a("right"),
            puberr.a("left", TARGET_GAP),
        ],
        "edge-hil",
        ignore=("p4", "puberr"),
    )

    branch_label(
        out,
        (p4.right + puberr.left) / 2,
        p4.cy - 6,
        "exception",
        COL["hil"],
        size=6.7,
    )

    p4m = Cg(
        "p4m",
        cx, 735,
        8,
    )

    out.append(
        circle_svg(
            p4m,
            COL["neutral"],
        )
    )

    add_route(
        out, routes,
        "p4-p4m",
        [
            p4.a("bottom"),
            p4m.a("top", TARGET_GAP),
        ],
        ignore=("p4", "p4m"),
    )

    add_route(
        out, routes,
        "pint-p4m",
        [
            pint.a("bottom"),
            (pint.cx, p4m.cy),
            p4m.a("left", TARGET_GAP),
        ],
        "edge-hil",
        ignore=("pint", "p4m"),
    )

    add_route(
        out, routes,
        "puberr-p4m",
        [
            puberr.a("bottom"),
            (puberr.cx, p4m.cy),
            p4m.a("right", TARGET_GAP),
        ],
        "edge-hil",
        ignore=("puberr", "p4m"),
    )

    # ==================================================================
    # Final summary + scenario-failure promotion
    # ==================================================================

    summ = Rg(
        "summ",
        220, 770,
        600, 54,
    )

    draw_node(
        out,
        summ,
        COL["purple"],
        "Build + print final summary",
        (
            "scenario statuses, HC metrics and written artefacts remain visible",
        ),
        9.1,
        6.4,
    )

    add_route(
        out, routes,
        "p4m-s",
        [
            p4m.a("bottom"),
            summ.a("top", TARGET_GAP),
        ],
        ignore=("p4m", "summ"),
    )

    dfail = Dg(
        "dfail",
        cx, 865,
        260, 40,
    )

    draw_decision(
        out,
        dfail,
        "OK status + real scenario failures?",
        ts=8.8,
    )

    add_route(
        out, routes,
        "s-dfail",
        [
            summ.a("bottom"),
            dfail.a("top", TARGET_GAP),
        ],
        ignore=("summ", "dfail"),
    )

    mark = Rg(
        "mark",
        760, 844,
        250, 42,
    )

    draw_node(
        out,
        mark,
        COL["red"],
        "Promote to SIMULATION_ERROR",
        (
            "summary remains visible",
        ),
        7.8,
        5.7,
        8,
    )

    add_route(
        out, routes,
        "dfail-mark",
        [
            dfail.a("right"),
            mark.a("left", TARGET_GAP),
        ],
        "edge-hil",
        ignore=("dfail", "mark"),
    )

    branch_label(
        out,
        (dfail.right + mark.left) / 2,
        dfail.cy - 6,
        "yes",
        COL["hil"],
        size=7.0,
    )

    em = Cg(
        "em",
        cx, 925,
        8,
    )

    out.append(
        circle_svg(
            em,
            COL["neutral"],
        )
    )

    add_route(
        out, routes,
        "dfail-em",
        [
            dfail.a("bottom"),
            em.a("top", TARGET_GAP),
        ],
        ignore=("dfail", "em"),
    )

    branch_label(
        out,
        dfail.cx + 10,
        dfail.bottom + 10,
        "no",
        anchor="start",
        size=7.0,
    )

    add_route(
        out, routes,
        "mark-em",
        [
            mark.a("bottom"),
            (mark.cx, em.cy),
            em.a("right", TARGET_GAP),
        ],
        "edge-hil",
        ignore=("mark", "em"),
    )

    ret = Rg(
        "ret",
        250, 960,
        540, 50,
    )

    draw_node(
        out,
        ret,
        COL["purple"],
        "return post-run ExitCode",
        (
            "0 success; 7 scenario failures; 8 publishing failure; "
            "130 publishing interruption",
        ),
        8.8,
        6.2,
    )

    add_route(
        out, routes,
        "em-ret",
        [
            em.a("bottom"),
            ret.a("top", TARGET_GAP),
        ],
        ignore=("em", "ret"),
    )

    # ==================================================================
    # Audience-facing panels
    # ==================================================================

    px, pw = 1080, 800

    panels = [
        R(px, 24, pw, 280),
        R(px, 326, pw, 325),
        R(px, 675, pw, 363),
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
            "Pre-run safeguards",
            COL["text"],
            17,
            700,
            "start",
        )
    )

    lines = [
        "Validation occurs before a potentially multi-hour run: cross-field constraints, file paths, dataset/network compatibility and serial availability are checked early.",
        "Per-run Q(V) values are applied through volt_var_controller.set_qv_parameters so software, coordinator sizing and Arduino CFG use the same characteristic.",
        "Violation-detector thresholds are reset then rebound for the run, keeping default-driven checks consistent with the selected limits.",
        "Hardware is probe-opened only when the chosen scenario set or custom controller will actually use the serial interface.",
    ]

    for i, t in enumerate(lines):
        out.append(
            label(
                px + 24,
                94 + 47 * i,
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
            359,
            "Run and publication semantics",
            COL["text"],
            17,
            700,
            "start",
        )
    )

    lines = [
        "Network/profile resolution is isolated as its own phase and returns the HC profile factory needed for stressed re-benchmarking.",
        "Topology/profile publication before simulation is intentionally non-fatal, so an export problem does not prevent the benchmark from starting.",
        "A controller plugin routes Phase 3 through register_and_run; otherwise the executor invokes run_benchmark directly.",
        "A Phase-3 failure or interruption returns immediately; Phase-4 publication problems instead preserve the completed result and continue through the final summary.",
    ]

    for i, t in enumerate(lines):
        out.append(
            label(
                px + 24,
                396 + 50 * i,
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
            708,
            "Typed process status",
            COL["text"],
            17,
            700,
            "start",
        )
    )

    lines = [
        "CONFIG, NETWORK, DATASET, PLUGIN and HARDWARE failures terminate in their guarded pre-run phases with their typed process status.",
        "Phase-3 controller-definition/import failures map to PLUGIN_ERROR; runtime crashes map to SIMULATION_ERROR; interruption returns 130 immediately.",
        "Publishing exceptions set PUBLISH_ERROR and publishing interruption sets 130, but both still allow the completed benchmark summary to be printed.",
        "If publishing remained OK but benchmark_runner reports real scenario failures, the final post-run status is promoted from OK to SIMULATION_ERROR.",
    ]

    for i, t in enumerate(lines):
        out.append(
            label(
                px + 24,
                745 + 52 * i,
                t,
                COL["text"],
                10.6,
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

    assert abs(e0.cy - p0.cy) < 1e-9
    assert abs(e1.cy - p1.cy) < 1e-9
    assert abs(e2.cy - p2.cy) < 1e-9
    assert abs(warn.cy - pre.cy) < 1e-9

    assert abs(preg.cy - dplug.cy) < 1e-9
    assert abs(bench.cy - dplug.cy) < 1e-9
    assert abs(perr.cy - preg.cy) < 1e-9
    assert abs(berr.cy - bench.cy) < 1e-9

    assert abs(pint.cy - p4.cy) < 1e-9
    assert abs(puberr.cy - p4.cy) < 1e-9

    assert ret.bottom > 1000

    write(
        "flow_cli_executor_presentation_v2",
        W,
        H,
        "CLI executor phases and typed exits - presentation",
        "\n".join(out),
    )

def build_resolve_ieee():
    W, H = 1180, 1285
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
    nxc = 350
    pxc = 955

    out.append(
        label(
            W / 2,
            38,
            "CLI Executor - Network and Dataset Resolution",
            COL["text"],
            17,
            700,
        )
    )

    start = Rg("start", 270, 60, 640, 62)
    draw_node(
        out,
        start,
        COL["blue"],
        "build_net_and_profiles(plan)",
        ("ensure framework path; read NetworkConfig + DatasetConfig",),
        11.5,
        8.1,
    )

    dplug = Dg("dplug", cx, 175, 290, 60)
    draw_decision(out, dplug, "network source is plugin?", ts=10.5)

    add_route(
        out,
        routes,
        "s-dplug",
        [start.a("bottom"), dplug.a("top", TARGET_GAP)],
        ignore=("start", "dplug"),
    )

    # ==============================================================
    # Plugin path
    # ==============================================================

    plug = Rg("plug", 790, 140, 330, 72)
    draw_node(
        out,
        plug,
        COL["green"],
        "load_network_from_yaml(plugin_path)",
        (
            "returns net + profiles; derive network_id",
            "create same-strategy profile_factory with make_profile_factory(plugin_path)",
        ),
        8.6,
        6.1,
        8,
    )

    add_route(
        out,
        routes,
        "dplug-plug",
        [dplug.a("right"), plug.a("left", TARGET_GAP)],
        "edge-gate",
        ignore=("dplug", "plug"),
    )

    branch_label(
        out,
        (dplug.right + plug.left) / 2,
        dplug.cy - 8,
        "yes",
        COL["gate"],
        size=7.7,
    )

    pwarn = Dg("pwarn", pxc, 280, 250, 56)
    draw_decision(
        out,
        pwarn,
        "validate_network_plugin(...) returned warnings?",
        ts=8.6,
    )

    add_route(
        out,
        routes,
        "plug-pwarn",
        [plug.a("bottom"), pwarn.a("top", TARGET_GAP)],
        ignore=("plug", "pwarn"),
    )

    pconf = Dg("pconf", pxc, 365, 220, 52)
    draw_decision(out, pconf, "user proceeds?", ts=9.1)

    add_route(
        out,
        routes,
        "pw-pc",
        [pwarn.a("bottom"), pconf.a("top", TARGET_GAP)],
        "edge-gate",
        ignore=("pwarn", "pconf"),
    )

    branch_label(
        out,
        pwarn.cx + 10,
        pwarn.bottom + 11,
        "yes",
        COL["gate"],
        "start",
        7.2,
    )

    # Rejected warnings terminate this resolver call.
    perr = Rg("perr", 650, 339, 170, 52)
    draw_node(
        out,
        perr,
        COL["red"],
        "Raise PluginError",
        ("compatibility warnings rejected",),
        7.9,
        5.7,
        8,
    )

    add_route(
        out,
        routes,
        "pc-err",
        [pconf.a("left"), perr.a("right", TARGET_GAP)],
        "edge-hil",
        ignore=("pconf", "perr"),
    )

    branch_label(
        out,
        (pconf.left + perr.right) / 2,
        pconf.cy - 7,
        "no",
        COL["hil"],
        size=7.0,
    )

    # No-warning and confirmed-warning paths merge before modifications.
    pjoin = Cg("pjoin", pxc, 445, 10)
    out.append(circle_svg(pjoin, COL["neutral"]))

    add_route(
        out,
        routes,
        "pc-yes-join",
        [pconf.a("bottom"), pjoin.a("top", TARGET_GAP)],
        "edge-gate",
        ignore=("pconf", "pjoin"),
    )

    branch_label(
        out,
        pconf.cx + 10,
        pconf.bottom + 11,
        "yes",
        COL["gate"],
        "start",
        7.1,
    )

    pwarn_bypass_x = 1145

    add_route(
        out,
        routes,
        "pw-no-join",
        [
            pwarn.a("right"),
            (pwarn_bypass_x, pwarn.cy),
            (pwarn_bypass_x, pjoin.cy),
            pjoin.a("right", TARGET_GAP),
        ],
        "edge-gate",
        ignore=("pwarn", "pjoin"),
    )

    branch_label(
        out,
        pwarn.right + 20,
        pwarn.cy - 7,
        "no",
        COL["gate"],
        size=7.1,
    )

    pmod = Dg("pmod", pxc, 505, 250, 56)
    draw_decision(
        out,
        pmod,
        "network modifications changed net?",
        ts=8.7,
    )

    add_route(
        out,
        routes,
        "join-pmod",
        [pjoin.a("bottom"), pmod.a("top", TARGET_GAP)],
        ignore=("pjoin", "pmod"),
    )

    rebuild = Rg("rebuild", 800, 565, 310, 60)
    draw_node(
        out,
        rebuild,
        COL["green"],
        "profile_factory(net)",
        ("rebuild plugin profiles after modifications",),
        8.7,
        6.3,
        8,
    )

    add_route(
        out,
        routes,
        "pmod-rebuild",
        [pmod.a("bottom"), rebuild.a("top", TARGET_GAP)],
        "edge-gate",
        ignore=("pmod", "rebuild"),
    )

    branch_label(
        out,
        pmod.cx + 10,
        pmod.bottom + 11,
        "yes",
        COL["gate"],
        "start",
        7.1,
    )

    pm = Cg("pm", pxc, 655, 10)
    out.append(circle_svg(pm, COL["neutral"]))

    add_route(
        out,
        routes,
        "reb-pm",
        [rebuild.a("bottom"), pm.a("top", TARGET_GAP)],
        ignore=("rebuild", "pm"),
    )

    pmod_bypass_x = 1145

    add_route(
        out,
        routes,
        "pmod-no",
        [
            pmod.a("right"),
            (pmod_bypass_x, pmod.cy),
            (pmod_bypass_x, pm.cy),
            pm.a("right", TARGET_GAP),
        ],
        ignore=("pmod", "pm"),
    )

    branch_label(
        out,
        pmod.right + 18,
        pmod.cy - 7,
        "no",
        size=7.1,
    )

    # ==============================================================
    # Non-plugin path
    # ==============================================================

    dsrc = Dg("dsrc", nxc, 300, 240, 58)
    draw_decision(
        out,
        dsrc,
        "non-plugin network source?",
        ("preset / simbench_code / custom",),
        9.6,
        6.8,
    )

    add_route(
        out,
        routes,
        "dplug-dsrc",
        [
            dplug.a("bottom"),
            (dplug.cx, 245),
            (dsrc.cx, 245),
            dsrc.a("top", TARGET_GAP),
        ],
        ignore=("dplug", "dsrc"),
    )

    branch_label(
        out,
        dplug.cx + 14,
        dplug.bottom + 14,
        "no",
        anchor="start",
        size=7.7,
    )

    # Side alternatives share the decision's vertical centre,
    # so the preset/custom arrows are truly horizontal.
    preset = Rg("preset", 25, 266, 190, 68)
    draw_node(
        out,
        preset,
        COL["green"],
        "Preset",
        ("_preset_loaders()[name](); optional sb_code",),
        8.0,
        5.8,
        8,
    )

    custom = Rg("custom", 485, 266, 205, 68)
    draw_node(
        out,
        custom,
        COL["green"],
        "Custom Python",
        (
            "import configured factory; call it",
            "require pandapower bus/line/load/sgen tables",
        ),
        7.6,
        5.5,
        8,
    )

    sb = Rg("sb", 250, 355, 200, 68)
    draw_node(
        out,
        sb,
        COL["green"],
        "SimBench code",
        ("sb.get_simbench_net(code)",),
        8.2,
        5.9,
        8,
    )

    add_route(
        out,
        routes,
        "src-preset",
        [dsrc.a("left"), preset.a("right", TARGET_GAP)],
        "edge-gate",
        ignore=("dsrc", "preset"),
    )

    branch_label(
        out,
        (dsrc.left + preset.right) / 2,
        dsrc.cy - 8,
        "preset",
        COL["gate"],
        size=6.9,
    )

    add_route(
        out,
        routes,
        "src-custom",
        [dsrc.a("right"), custom.a("left", TARGET_GAP)],
        "edge-gate",
        ignore=("dsrc", "custom"),
    )

    branch_label(
        out,
        (dsrc.right + custom.left) / 2,
        dsrc.cy - 8,
        "custom",
        COL["gate"],
        size=6.9,
    )

    add_route(
        out,
        routes,
        "src-sb",
        [dsrc.a("bottom"), sb.a("top", TARGET_GAP)],
        "edge-gate",
        ignore=("dsrc", "sb"),
    )

    branch_label(
        out,
        dsrc.cx + 12,
        dsrc.bottom + 12,
        "simbench",
        COL["gate"],
        "start",
        6.9,
    )

    nm = Cg("nm", nxc, 455, 10)
    out.append(circle_svg(nm, COL["neutral"]))

    add_route(
        out,
        routes,
        "preset-nm",
        [
            preset.a("bottom"),
            (preset.cx, nm.cy),
            nm.a("left", TARGET_GAP),
        ],
        ignore=("preset", "nm"),
    )

    add_route(
        out,
        routes,
        "sb-nm",
        [sb.a("bottom"), nm.a("top", TARGET_GAP)],
        ignore=("sb", "nm"),
    )

    add_route(
        out,
        routes,
        "custom-nm",
        [
            custom.a("bottom"),
            (custom.cx, nm.cy),
            nm.a("right", TARGET_GAP),
        ],
        ignore=("custom", "nm"),
    )

    mods = Rg("mods", 90, 490, 520, 64)
    draw_node(
        out,
        mods,
        COL["green"],
        "Apply network modifications BEFORE profiles",
        ("inject PV sgens; flip selected switches",),
        9.6,
        6.9,
    )

    add_route(
        out,
        routes,
        "nm-mods",
        [nm.a("bottom"), mods.a("top", TARGET_GAP)],
        ignore=("nm", "mods"),
    )

    # ==============================================================
    # Non-plugin dataset dispatch
    # ==============================================================

    dds = Dg("dds", nxc, 610, 240, 58)
    draw_decision(
        out,
        dds,
        "dataset source type?",
        ("simbench_native / dwd / custom",),
        9.5,
        6.8,
    )

    add_route(
        out,
        routes,
        "mods-dds",
        [mods.a("bottom"), dds.a("top", TARGET_GAP)],
        ignore=("mods", "dds"),
    )

    d1 = Rg("d1", 25, 576, 190, 68)
    draw_node(
        out,
        d1,
        COL["green"],
        "SimBench native",
        ("requires sb_code",),
        8.1,
        5.8,
        8,
    )

    d3 = Rg("d3", 485, 576, 205, 68)
    draw_node(
        out,
        d3,
        COL["green"],
        "Custom dataset",
        (
            "custom_path is data directory",
            "file_map + col_map forwarded",
        ),
        7.7,
        5.5,
        8,
    )

    d2 = Rg("d2", 250, 665, 200, 68)
    draw_node(
        out,
        d2,
        COL["green"],
        "DWD",
        ("validate data directory / station",),
        8.1,
        5.8,
        8,
    )

    add_route(
        out,
        routes,
        "dds-d1",
        [dds.a("left"), d1.a("right", TARGET_GAP)],
        "edge-gate",
        ignore=("dds", "d1"),
    )

    branch_label(
        out,
        (dds.left + d1.right) / 2,
        dds.cy - 8,
        "simbench",
        COL["gate"],
        size=6.7,
    )

    add_route(
        out,
        routes,
        "dds-d3",
        [dds.a("right"), d3.a("left", TARGET_GAP)],
        "edge-gate",
        ignore=("dds", "d3"),
    )

    branch_label(
        out,
        (dds.right + d3.left) / 2,
        dds.cy - 8,
        "custom",
        COL["gate"],
        size=6.8,
    )

    add_route(
        out,
        routes,
        "dds-d2",
        [dds.a("bottom"), d2.a("top", TARGET_GAP)],
        "edge-gate",
        ignore=("dds", "d2"),
    )

    branch_label(
        out,
        dds.cx + 12,
        dds.bottom + 12,
        "dwd",
        COL["gate"],
        "start",
        6.9,
    )

    dm = Cg("dm", nxc, 765, 10)
    out.append(circle_svg(dm, COL["neutral"]))

    add_route(
        out,
        routes,
        "d1-dm",
        [
            d1.a("bottom"),
            (d1.cx, dm.cy),
            dm.a("left", TARGET_GAP),
        ],
        ignore=("d1", "dm"),
    )

    add_route(
        out,
        routes,
        "d2-dm",
        [d2.a("bottom"), dm.a("top", TARGET_GAP)],
        ignore=("d2", "dm"),
    )

    add_route(
        out,
        routes,
        "d3-dm",
        [
            d3.a("bottom"),
            (d3.cx, dm.cy),
            dm.a("right", TARGET_GAP),
        ],
        ignore=("d3", "dm"),
    )

    build = Rg("build", 90, 800, 520, 64)
    draw_node(
        out,
        build,
        COL["blue"],
        "build_annual_profiles(net, **builder_kwargs)",
        ("builder exception -> DatasetError",),
        9.4,
        6.7,
    )

    add_route(
        out,
        routes,
        "dm-build",
        [dm.a("bottom"), build.a("top", TARGET_GAP)],
        ignore=("dm", "build"),
    )

    factory = Rg("factory", 90, 890, 520, 68)
    draw_node(
        out,
        factory,
        COL["green"],
        "Define profile_factory(net_hc)",
        (
            "reuse builder kwargs + same selected time window for HC-stressed net",
        ),
        9.1,
        6.5,
    )

    add_route(
        out,
        routes,
        "build-f",
        [build.a("bottom"), factory.a("top", TARGET_GAP)],
        ignore=("build", "factory"),
    )

    # ==============================================================
    # Parallel final consistency
    # ==============================================================

    postn = Rg("postn", 70, 995, 560, 82)
    draw_node(
        out,
        postn,
        COL["green"],
        "Non-plugin post-build consistency",
        (
            "scale net tables + matching profile columns; apply time window",
            "check actual timestep resolution; validate focus buses",
        ),
        8.9,
        6.3,
    )

    add_route(
        out,
        routes,
        "f-postn",
        [factory.a("bottom"), postn.a("top", TARGET_GAP)],
        ignore=("factory", "postn"),
    )

    postp = Rg("postp", 760, 995, 390, 82)
    draw_node(
        out,
        postp,
        COL["green"],
        "Plugin post-build consistency",
        (
            "scale net tables + matching profile columns; apply time window",
            "check actual timestep resolution; validate focus buses",
        ),
        8.4,
        6.0,
        8,
    )

    add_route(
        out,
        routes,
        "pm-postp",
        [pm.a("bottom"), postp.a("top", TARGET_GAP)],
        ignore=("pm", "postp"),
    )

    rm = Cg("rm", 620, 1135, 11)
    out.append(circle_svg(rm, COL["neutral"]))

    add_route(
        out,
        routes,
        "postn-rm",
        [
            postn.a("bottom"),
            (postn.cx, rm.cy),
            rm.a("left", TARGET_GAP),
        ],
        ignore=("postn", "rm"),
    )

    add_route(
        out,
        routes,
        "postp-rm",
        [
            postp.a("bottom"),
            (postp.cx, rm.cy),
            rm.a("right", TARGET_GAP),
        ],
        "edge-gate",
        ignore=("postp", "rm"),
    )

    ret = Rg("ret", 300, 1175, 640, 68)
    draw_node(
        out,
        ret,
        COL["purple"],
        "return (net, profiles, network_id, profile_factory)",
        ("plugin and non-plugin branches expose one executor contract",),
        10.2,
        7.3,
    )

    add_route(
        out,
        routes,
        "rm-ret",
        [rm.a("bottom"), ret.a("top", TARGET_GAP)],
        ignore=("rm", "ret"),
    )

    audit_bounds(nodes, W, H, 8)
    audit_node_overlaps(nodes)
    audit_routes(routes, nodes)

    assert abs(preset.cy - dsrc.cy) < 1e-9
    assert abs(custom.cy - dsrc.cy) < 1e-9
    assert abs(d1.cy - dds.cy) < 1e-9
    assert abs(d3.cy - dds.cy) < 1e-9
    assert abs(postn.cy - postp.cy) < 1e-9
    assert H - ret.bottom <= 50

    write(
        "flow_cli_resolve_ieee_v2",
        W,
        H,
        "CLI network and dataset resolution - IEEE",
        "\n".join(out),
    )

def build_resolve_presentation():
    W, H = 1920, 1080
    out = [rect_svg(R(0, 0, W, H), COL["panel"], 0)]
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

    nxc = 250
    pxc = 810

    start = Rg("start", 150, 18, 700, 44)
    draw_node(
        out,
        start,
        COL["blue"],
        "build_net_and_profiles(plan)",
        ("resolve one network/profile pair before benchmark configuration",),
        10.8,
        7.3,
    )

    dplug = Dg("dplug", 500, 125, 250, 40)
    draw_decision(out, dplug, "network source is plugin?", ts=9.5)

    add_route(
        out,
        routes,
        "s-d",
        [start.a("bottom"), dplug.a("top", TARGET_GAP)],
        ignore=("start", "dplug"),
    )

    # ==============================================================
    # Non-plugin network-source dispatch
    # ==============================================================

    dsrc = Dg("dsrc", nxc, 225, 210, 40)
    draw_decision(
        out,
        dsrc,
        "non-plugin network source?",
        ts=8.8,
    )

    add_route(
        out,
        routes,
        "d-dsrc",
        [
            dplug.a("left"),
            (nxc, dplug.cy),
            dsrc.a("top", TARGET_GAP),
        ],
        "edge-gate",
        ignore=("dplug", "dsrc"),
    )

    branch_label(
        out,
        dplug.left - 22,
        dplug.cy - 6,
        "no",
        COL["gate"],
        "end",
        7.0,
    )

    # Side blocks align with the diamond centre for straight arrows.
    preset = Rg("preset", 10, 201, 130, 48)
    draw_node(
        out,
        preset,
        COL["green"],
        "Preset",
        ("loader catalogue",),
        7.4,
        5.4,
        8,
    )

    custom = Rg("custom", 410, 201, 180, 48)
    draw_node(
        out,
        custom,
        COL["green"],
        "Custom Python",
        ("import + call factory",),
        7.5,
        5.4,
        8,
    )

    sb = Rg("sb", 165, 280, 170, 48)
    draw_node(
        out,
        sb,
        COL["green"],
        "SimBench code",
        ("sb.get_simbench_net(code)",),
        7.6,
        5.5,
        8,
    )

    add_route(
        out,
        routes,
        "src-preset",
        [dsrc.a("left"), preset.a("right", TARGET_GAP)],
        "edge-gate",
        ignore=("dsrc", "preset"),
    )

    branch_label(
        out,
        (dsrc.left + preset.right) / 2,
        dsrc.cy - 6,
        "preset",
        COL["gate"],
        size=6.5,
    )

    add_route(
        out,
        routes,
        "src-custom",
        [dsrc.a("right"), custom.a("left", TARGET_GAP)],
        "edge-gate",
        ignore=("dsrc", "custom"),
    )

    branch_label(
        out,
        (dsrc.right + custom.left) / 2,
        dsrc.cy - 6,
        "custom",
        COL["gate"],
        size=6.5,
    )

    add_route(
        out,
        routes,
        "src-sb",
        [dsrc.a("bottom"), sb.a("top", TARGET_GAP)],
        "edge-gate",
        ignore=("dsrc", "sb"),
    )

    branch_label(
        out,
        dsrc.cx + 9,
        dsrc.bottom + 10,
        "simbench",
        COL["gate"],
        "start",
        6.5,
    )

    nm = Cg("nm", nxc, 350, 8)
    out.append(circle_svg(nm, COL["neutral"]))

    add_route(
        out,
        routes,
        "preset-nm",
        [
            preset.a("bottom"),
            (preset.cx, nm.cy),
            nm.a("left", TARGET_GAP),
        ],
        ignore=("preset", "nm"),
    )

    add_route(
        out,
        routes,
        "sb-nm",
        [sb.a("bottom"), nm.a("top", TARGET_GAP)],
        ignore=("sb", "nm"),
    )

    add_route(
        out,
        routes,
        "custom-nm",
        [
            custom.a("bottom"),
            (custom.cx, nm.cy),
            nm.a("right", TARGET_GAP),
        ],
        ignore=("custom", "nm"),
    )

    mods = Rg("mods", 50, 380, 400, 50)
    draw_node(
        out,
        mods,
        COL["green"],
        "Apply network modifications",
        ("DER injection + switch flips before profiles",),
        8.7,
        6.2,
        8,
    )

    add_route(
        out,
        routes,
        "nm-mods",
        [nm.a("bottom"), mods.a("top", TARGET_GAP)],
        ignore=("nm", "mods"),
    )

    # ==============================================================
    # Non-plugin dataset dispatch
    # ==============================================================

    dds = Dg("dds", nxc, 500, 200, 40)
    draw_decision(out, dds, "dataset source?", ts=9.0)

    add_route(
        out,
        routes,
        "mods-dds",
        [mods.a("bottom"), dds.a("top", TARGET_GAP)],
        ignore=("mods", "dds"),
    )

    d1 = Rg("d1", 10, 476, 130, 48)
    draw_node(
        out,
        d1,
        COL["green"],
        "SimBench native",
        ("requires sb_code",),
        7.2,
        5.3,
        8,
    )

    d3 = Rg("d3", 410, 476, 180, 48)
    draw_node(
        out,
        d3,
        COL["green"],
        "Custom data directory",
        ("file_map + col_map",),
        7.2,
        5.2,
        8,
    )

    d2 = Rg("d2", 165, 555, 170, 48)
    draw_node(
        out,
        d2,
        COL["green"],
        "DWD",
        ("directory/station pre-check",),
        7.5,
        5.4,
        8,
    )

    add_route(
        out,
        routes,
        "dds-d1",
        [dds.a("left"), d1.a("right", TARGET_GAP)],
        "edge-gate",
        ignore=("dds", "d1"),
    )

    branch_label(
        out,
        (dds.left + d1.right) / 2,
        dds.cy - 6,
        "simbench",
        COL["gate"],
        size=6.3,
    )

    add_route(
        out,
        routes,
        "dds-d3",
        [dds.a("right"), d3.a("left", TARGET_GAP)],
        "edge-gate",
        ignore=("dds", "d3"),
    )

    branch_label(
        out,
        (dds.right + d3.left) / 2,
        dds.cy - 6,
        "custom",
        COL["gate"],
        size=6.4,
    )

    add_route(
        out,
        routes,
        "dds-d2",
        [dds.a("bottom"), d2.a("top", TARGET_GAP)],
        "edge-gate",
        ignore=("dds", "d2"),
    )

    branch_label(
        out,
        dds.cx + 9,
        dds.bottom + 10,
        "dwd",
        COL["gate"],
        "start",
        6.4,
    )

    dm = Cg("dm", nxc, 625, 8)
    out.append(circle_svg(dm, COL["neutral"]))

    add_route(
        out,
        routes,
        "d1-dm",
        [
            d1.a("bottom"),
            (d1.cx, dm.cy),
            dm.a("left", TARGET_GAP),
        ],
        ignore=("d1", "dm"),
    )

    add_route(
        out,
        routes,
        "d2-dm",
        [d2.a("bottom"), dm.a("top", TARGET_GAP)],
        ignore=("d2", "dm"),
    )

    add_route(
        out,
        routes,
        "d3-dm",
        [
            d3.a("bottom"),
            (d3.cx, dm.cy),
            dm.a("right", TARGET_GAP),
        ],
        ignore=("d3", "dm"),
    )

    build = Rg("build", 45, 660, 410, 58)
    draw_node(
        out,
        build,
        COL["blue"],
        "build_annual_profiles(...) + define HC profile_factory",
        ("same builder strategy is retained for a stressed network",),
        8.3,
        5.9,
        8,
    )

    add_route(
        out,
        routes,
        "dm-build",
        [dm.a("bottom"), build.a("top", TARGET_GAP)],
        ignore=("dm", "build"),
    )

    # ==============================================================
    # Plugin path
    # ==============================================================

    plug = Rg("plug", 650, 100, 320, 50)
    draw_node(
        out,
        plug,
        COL["green"],
        "Network plugin YAML",
        (
            "load net + profiles; create same-strategy profile_factory",
        ),
        8.4,
        6.0,
        8,
    )

    add_route(
        out,
        routes,
        "d-p",
        [dplug.a("right"), plug.a("left", TARGET_GAP)],
        "edge-gate",
        ignore=("dplug", "plug"),
    )

    branch_label(
        out,
        (dplug.right + plug.left) / 2,
        dplug.cy - 6,
        "yes",
        COL["gate"],
        size=7.1,
    )

    pwarn = Dg("pwarn", pxc, 220, 230, 40)
    draw_decision(
        out,
        pwarn,
        "validator returned warnings?",
        ts=8.5,
    )

    add_route(
        out,
        routes,
        "plug-pwarn",
        [plug.a("bottom"), pwarn.a("top", TARGET_GAP)],
        ignore=("plug", "pwarn"),
    )

    pconf = Dg("pconf", pxc, 300, 200, 40)
    draw_decision(out, pconf, "user proceeds?", ts=8.8)

    add_route(
        out,
        routes,
        "pw-conf",
        [pwarn.a("bottom"), pconf.a("top", TARGET_GAP)],
        "edge-gate",
        ignore=("pwarn", "pconf"),
    )

    branch_label(
        out,
        pwarn.cx + 10,
        pwarn.bottom + 10,
        "yes",
        COL["gate"],
        "start",
        6.8,
    )

    perr = Rg("perr", 930, 279, 115, 42)
    draw_node(
        out,
        perr,
        COL["red"],
        "Raise PluginError",
        ("warnings rejected",),
        6.6,
        4.8,
        7,
    )

    add_route(
        out,
        routes,
        "conf-err",
        [pconf.a("right"), perr.a("left", TARGET_GAP)],
        "edge-hil",
        ignore=("pconf", "perr"),
    )

    branch_label(
        out,
        (pconf.right + perr.left) / 2,
        pconf.cy - 6,
        "no",
        COL["hil"],
        size=6.5,
    )

    # Warning-free and confirmed-warning paths merge first.
    pjoin = Cg("pjoin", pxc, 370, 8)
    out.append(circle_svg(pjoin, COL["neutral"]))

    add_route(
        out,
        routes,
        "conf-yes-join",
        [pconf.a("bottom"), pjoin.a("top", TARGET_GAP)],
        "edge-gate",
        ignore=("pconf", "pjoin"),
    )

    branch_label(
        out,
        pconf.cx + 9,
        pconf.bottom + 10,
        "yes",
        COL["gate"],
        "start",
        6.7,
    )

    warn_bypass_x = 1060

    add_route(
        out,
        routes,
        "pw-no-join",
        [
            pwarn.a("right"),
            (warn_bypass_x, pwarn.cy),
            (warn_bypass_x, pjoin.cy),
            pjoin.a("right", TARGET_GAP),
        ],
        "edge-gate",
        ignore=("pwarn", "pjoin"),
    )

    branch_label(
        out,
        pwarn.right + 14,
        pwarn.cy - 6,
        "no",
        COL["gate"],
        size=6.7,
    )

    pmod = Dg("pmod", pxc, 430, 230, 40)
    draw_decision(
        out,
        pmod,
        "network modifications changed net?",
        ts=8.2,
    )

    add_route(
        out,
        routes,
        "join-pmod",
        [pjoin.a("bottom"), pmod.a("top", TARGET_GAP)],
        ignore=("pjoin", "pmod"),
    )

    rebuild = Rg("rebuild", 650, 475, 320, 48)
    draw_node(
        out,
        rebuild,
        COL["green"],
        "Rebuild plugin profiles",
        ("profile_factory(net) only after a changed network",),
        8.1,
        5.8,
        8,
    )

    add_route(
        out,
        routes,
        "pmod-rebuild",
        [pmod.a("bottom"), rebuild.a("top", TARGET_GAP)],
        "edge-gate",
        ignore=("pmod", "rebuild"),
    )

    branch_label(
        out,
        pmod.cx + 9,
        pmod.bottom + 10,
        "yes",
        COL["gate"],
        "start",
        6.7,
    )

    pm = Cg("pm", pxc, 550, 8)
    out.append(circle_svg(pm, COL["neutral"]))

    add_route(
        out,
        routes,
        "reb-pm",
        [rebuild.a("bottom"), pm.a("top", TARGET_GAP)],
        ignore=("rebuild", "pm"),
    )

    pmod_bypass_x = 1030

    add_route(
        out,
        routes,
        "pmod-no",
        [
            pmod.a("right"),
            (pmod_bypass_x, pmod.cy),
            (pmod_bypass_x, pm.cy),
            pm.a("right", TARGET_GAP),
        ],
        ignore=("pmod", "pm"),
    )

    branch_label(
        out,
        pmod.right + 14,
        pmod.cy - 6,
        "no",
        size=6.7,
    )

    # ==============================================================
    # Parallel post-build consistency and common return
    # ==============================================================

    postn = Rg("postn", 45, 790, 410, 60)
    draw_node(
        out,
        postn,
        COL["green"],
        "Non-plugin post-build consistency",
        (
            "scale net + profiles; time window; resolution/focus checks",
        ),
        8.2,
        5.9,
        8,
    )

    add_route(
        out,
        routes,
        "build-postn",
        [build.a("bottom"), postn.a("top", TARGET_GAP)],
        ignore=("build", "postn"),
    )

    postp = Rg("postp", 605, 790, 410, 60)
    draw_node(
        out,
        postp,
        COL["green"],
        "Plugin post-build consistency",
        (
            "scale net + profiles; time window; resolution/focus checks",
        ),
        8.2,
        5.9,
        8,
    )

    add_route(
        out,
        routes,
        "pm-postp",
        [pm.a("bottom"), postp.a("top", TARGET_GAP)],
        ignore=("pm", "postp"),
    )

    rm = Cg("rm", 520, 885, 8)
    out.append(circle_svg(rm, COL["neutral"]))

    add_route(
        out,
        routes,
        "postn-rm",
        [
            postn.a("bottom"),
            (postn.cx, rm.cy),
            rm.a("left", TARGET_GAP),
        ],
        ignore=("postn", "rm"),
    )

    add_route(
        out,
        routes,
        "postp-rm",
        [
            postp.a("bottom"),
            (postp.cx, rm.cy),
            rm.a("right", TARGET_GAP),
        ],
        ignore=("postp", "rm"),
    )

    ret = Rg("ret", 190, 925, 660, 58)
    draw_node(
        out,
        ret,
        COL["purple"],
        "Return net + profiles + network_id + profile_factory",
        ("one executor contract for plugin and non-plugin paths",),
        9.3,
        6.6,
    )

    add_route(
        out,
        routes,
        "rm-ret",
        [rm.a("bottom"), ret.a("top", TARGET_GAP)],
        ignore=("rm", "ret"),
    )

    # ==============================================================
    # Audience-facing explanatory panels
    # ==============================================================

    px, pw = 1080, 800

    panels = [
        R(px, 24, pw, 280),
        R(px, 326, pw, 320),
        R(px, 668, pw, 370),
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
            "Network resolution",
            COL["text"],
            17,
            700,
            "start",
        )
    )

    lines = [
        "Non-plugin network resolution follows three real source branches: preset loader, assembled SimBench code, or a custom Python factory.",
        "Custom Python networks are imported from the configured file and must return an object with pandapower bus/line/load/sgen tables.",
        "Network-plugin YAML is different: load_network_from_yaml returns net + profiles, then make_profile_factory creates the same-strategy rebuild callable.",
        "Plugin compatibility warnings are shown before execution and require explicit confirmation when any are present.",
    ]

    for i, t in enumerate(lines):
        out.append(
            label(
                px + 24,
                94 + 47 * i,
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
            359,
            "Dataset resolution",
            COL["text"],
            17,
            700,
            "start",
        )
    )

    lines = [
        "Only non-plugin networks enter dataset dispatch: SimBench native, DWD, or a custom data directory with optional file/column maps.",
        "SimBench-native profiles require an available SimBench code; DWD performs a directory/station pre-check; custom requires custom_path to be a directory.",
        "All three feed profile_builder.build_annual_profiles and create a closure that can rebuild profiles for a later HC-stressed network.",
        "Plugin networks skip this dataset dispatch because their YAML strategy already constructed the initial profile set.",
    ]

    for i, t in enumerate(lines):
        out.append(
            label(
                px + 24,
                396 + 50 * i,
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
            701,
            "Post-build consistency",
            COL["text"],
            17,
            700,
            "start",
        )
    )

    lines = [
        "User-requested DER injections and switch flips are applied before profile construction or any required plugin-profile rebuild.",
        "Load/DER scaling changes both network tables and matching profile columns; the selected time window is then applied.",
        "timestep_resolution is checked against the actual profile spacing and warns on mismatch; focus_buses are validated against the loaded net.",
        "Both paths return the same four values to the benchmark-configuration phase.",
    ]

    for i, t in enumerate(lines):
        out.append(
            label(
                px + 24,
                738 + 52 * i,
                t,
                COL["text"],
                10.6,
                600,
                "start",
            )
        )

    audit_bounds(nodes, W, H, 0)
    audit_node_overlaps(nodes)
    audit_routes(routes, nodes)

    assert abs(preset.cy - dsrc.cy) < 1e-9
    assert abs(custom.cy - dsrc.cy) < 1e-9
    assert abs(d1.cy - dds.cy) < 1e-9
    assert abs(d3.cy - dds.cy) < 1e-9
    assert abs(postn.cy - postp.cy) < 1e-9
    assert ret.bottom > 975

    write(
        "flow_cli_resolve_presentation_v2",
        W,
        H,
        "CLI network and dataset resolution - presentation",
        "\n".join(out),
    )

def main():
    build_entry_ieee(); build_entry_presentation()
    build_wizard_ieee(); build_wizard_presentation()
    build_executor_ieee(); build_executor_presentation()
    build_resolve_ieee(); build_resolve_presentation()
    print(f"Wrote CLI flowcharts to {OUT}")

if __name__ == "__main__":
    main()
