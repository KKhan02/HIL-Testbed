from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import html
import xml.etree.ElementTree as ET

import cairosvg

OUT = Path(__file__).resolve().parent / "network_plugin_flowcharts_v1"
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
# Diagram 1: public loader + separate compatibility validator
# =============================================================================

def build_load_ieee():
    W, H = 1100, 1220

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

    cx = 550

    out.append(
        label(
            W / 2,
            38,
            "Network Plugin - YAML Load, Network Source and Public Validation",
            COL["text"],
            17,
            700,
        )
    )

    # ==================================================================
    # Public loader
    # ==================================================================

    start = Rg(
        "start",
        250, 62,
        600, 58,
    )

    draw_node(
        out,
        start,
        COL["blue"],
        "load_network_from_yaml(yaml_path)",
        ("resolve user path to absolute Path",),
        12.5,
        8.8,
    )

    cfg = Rg(
        "cfg",
        230, 150,
        640, 92,
    )

    draw_node(
        out,
        cfg,
        COL["green"],
        "_load_yaml_config(yaml_path)",
        (
            "parse YAML mapping; validate name/label and source; resolve source paths relative to YAML",
            "validate strategy in {simbench_native, dwd_pvlib, flat, custom}, year and custom file_map",
            "resolve optional data_dir; validate 0.5 <= v_min < v_max <= 1.5",
        ),
        11.2,
        7.7,
    )

    add_route(
        out, routes,
        "start-cfg",
        [
            start.a("bottom"),
            cfg.a("top", TARGET_GAP),
        ],
        ignore=("start", "cfg"),
    )

    # ==================================================================
    # Network-source dispatch
    # ==================================================================

    djson = Dg(
        "djson",
        cx, 300,
        250, 58,
    )

    draw_decision(
        out,
        djson,
        "cfg['source'] == 'json'?",
        ts=11.0,
    )

    add_route(
        out, routes,
        "cfg-djson",
        [
            cfg.a("bottom"),
            djson.a("top", TARGET_GAP),
        ],
        ignore=("cfg", "djson"),
    )

    jsonb = Rg(
        "jsonb",
        60, 271,
        280, 58,
    )

    draw_node(
        out,
        jsonb,
        COL["green"],
        "pandapower.from_json(path)",
        ("JSON source",),
        10.3,
        7.5,
        9,
    )

    add_route(
        out, routes,
        "djson-json",
        [
            djson.a("left"),
            jsonb.a("right", TARGET_GAP),
        ],
        "edge-gate",
        ignore=("djson", "jsonb"),
    )

    branch_label(
        out,
        (djson.left + jsonb.right) / 2,
        djson.cy - 8,
        "yes",
        COL["gate"],
        size=8.5,
    )

    dpickle = Dg(
        "dpickle",
        cx, 395,
        250, 58,
    )

    draw_decision(
        out,
        dpickle,
        "cfg['source'] == 'pickle'?",
        ts=10.8,
    )

    add_route(
        out, routes,
        "djson-no",
        [
            djson.a("bottom"),
            dpickle.a("top", TARGET_GAP),
        ],
        ignore=("djson", "dpickle"),
    )

    branch_label(
        out,
        djson.cx + 15,
        djson.bottom + 15,
        "no",
        anchor="start",
        size=8.4,
    )

    pickleb = Rg(
        "pickleb",
        60, 366,
        280, 58,
    )

    draw_node(
        out,
        pickleb,
        COL["green"],
        "pandapower.from_pickle(path)",
        ("pandapower-managed pickle source",),
        9.9,
        7.2,
        9,
    )

    add_route(
        out, routes,
        "dpickle-pickle",
        [
            dpickle.a("left"),
            pickleb.a("right", TARGET_GAP),
        ],
        "edge-gate",
        ignore=("dpickle", "pickleb"),
    )

    branch_label(
        out,
        (dpickle.left + pickleb.right) / 2,
        dpickle.cy - 8,
        "yes",
        COL["gate"],
        size=8.5,
    )

    func = Rg(
        "func",
        720, 362,
        320, 66,
    )

    draw_node(
        out,
        func,
        COL["green"],
        "Import loader file + call zero-arg function",
        (
            "_import_loader_fn(module_path, function); net = fn()",
        ),
        9.8,
        7.2,
        9,
    )

    add_route(
        out, routes,
        "dpickle-func",
        [
            dpickle.a("right"),
            func.a("left", TARGET_GAP),
        ],
        "edge-gate",
        ignore=("dpickle", "func"),
    )

    branch_label(
        out,
        (dpickle.right + func.left) / 2,
        dpickle.cy - 8,
        "no -> function",
        COL["gate"],
        size=8.0,
    )

    # ==================================================================
    # Source-result merge
    # ==================================================================

    srcm = Cg(
        "srcm",
        cx, 505,
        12,
    )

    out.append(
        circle_svg(
            srcm,
            COL["neutral"],
        )
    )

    add_route(
        out, routes,
        "json-srcm",
        [
            jsonb.a("left"),
            (30, jsonb.cy),
            (30, srcm.cy),
            srcm.a("left", TARGET_GAP),
        ],
        ignore=("jsonb", "srcm"),
    )

    add_route(
        out, routes,
        "pickle-srcm",
        [
            pickleb.a("bottom"),
            (pickleb.cx, 460),
            (srcm.cx, 460),
            srcm.a("top", TARGET_GAP),
        ],
        ignore=("pickleb", "srcm"),
    )

    add_route(
        out, routes,
        "func-srcm",
        [
            func.a("bottom"),
            (func.cx, srcm.cy),
            srcm.a("right", TARGET_GAP),
        ],
        ignore=("func", "srcm"),
    )

    # ==================================================================
    # Type validation + profile construction
    # ==================================================================

    dtype = Dg(
        "dtype",
        cx, 585,
        260, 58,
    )

    draw_decision(
        out,
        dtype,
        "result is pandapowerNet?",
        ts=10.8,
    )

    add_route(
        out, routes,
        "srcm-dtype",
        [
            srcm.a("bottom"),
            dtype.a("top", TARGET_GAP),
        ],
        ignore=("srcm", "dtype"),
    )

    bad = Rg(
        "bad",
        735, 556,
        305, 58,
    )

    draw_node(
        out,
        bad,
        COL["red"],
        "Raise TypeError",
        (
            "function/source did not produce pandapowerNet",
        ),
        9.7,
        7.0,
        9,
    )

    add_route(
        out, routes,
        "dtype-bad",
        [
            dtype.a("right"),
            bad.a("left", TARGET_GAP),
        ],
        "edge-hil",
        ignore=("dtype", "bad"),
    )

    branch_label(
        out,
        (dtype.right + bad.left) / 2,
        dtype.cy - 8,
        "no",
        COL["hil"],
        size=8.4,
    )

    strat = Rg(
        "strat",
        250, 650,
        600, 66,
    )

    draw_node(
        out,
        strat,
        COL["green"],
        "_build_profiles_for_strategy(net, cfg)",
        (
            "returns profiles + strategy actually used; detailed in Diagram 2",
        ),
        10.8,
        7.8,
    )

    add_route(
        out, routes,
        "dtype-strat",
        [
            dtype.a("bottom"),
            strat.a("top", TARGET_GAP),
        ],
        "edge-gate",
        ignore=("dtype", "strat"),
    )

    branch_label(
        out,
        dtype.cx + 15,
        dtype.bottom + 15,
        "yes",
        COL["gate"],
        "start",
        8.4,
    )

    meta = Rg(
        "meta",
        230, 748,
        640, 80,
    )

    draw_node(
        out,
        meta,
        COL["purple"],
        "Attach profiles['plugin_meta']",
        (
            "name, label, source, actual strategy, requested strategy, year",
            "v_min/v_max, notes and absolute yaml_path",
        ),
        10.7,
        7.6,
    )

    ret = Rg(
        "ret",
        270, 858,
        560, 58,
    )

    draw_node(
        out,
        ret,
        COL["purple"],
        "return (net, profiles)",
        (
            "ready for benchmark_runner.run_benchmark",
        ),
        11.1,
        8.0,
    )

    add_route(
        out, routes,
        "strat-meta",
        [
            strat.a("bottom"),
            meta.a("top", TARGET_GAP),
        ],
        ignore=("strat", "meta"),
    )

    add_route(
        out, routes,
        "meta-ret",
        [
            meta.a("bottom"),
            ret.a("top", TARGET_GAP),
        ],
        ignore=("meta", "ret"),
    )

    # ==================================================================
    # Separate public compatibility validator
    #
    # This is an association, not an execution edge from the loader.
    # The rail stays completely LEFT of ventry before entering its
    # left anchor, avoiding the previous route-through-block problem.
    # ==================================================================

    ventry = Rg(
        "ventry",
        80, 1000,
        340, 60,
    )

    draw_node(
        out,
        ventry,
        COL["blue"],
        "validate_network_plugin(net, profiles)",
        (
            "separate caller-invoked public API",
        ),
        10.3,
        7.4,
        9,
    )

    checks = Rg(
        "checks",
        460, 984,
        560, 92,
    )

    draw_node(
        out,
        checks,
        COL["green"],
        "Compatibility checks",
        (
            "sgen/DER types; MV/LV voltage level; two-/three-winding transformer presence",
            "ZIP-load shares; profile/load index alignment; missing DER profile columns; NaNs",
            "append human-readable warning strings only - no exception is raised here",
        ),
        9.7,
        6.9,
        9,
    )

    vret = Rg(
        "vret",
        480, 1112,
        520, 58,
    )

    draw_node(
        out,
        vret,
        COL["purple"],
        "return warnings: list[str]",
        (
            "empty list when all checks pass",
        ),
        10.5,
        7.5,
    )

    add_route(
        out, routes,
        "ventry-checks",
        [
            ventry.a("right"),
            checks.a("left", TARGET_GAP),
        ],
        ignore=("ventry", "checks"),
    )

    add_route(
        out, routes,
        "checks-vret",
        [
            checks.a("bottom"),
            vret.a("top", TARGET_GAP),
        ],
        ignore=("checks", "vret"),
    )

    # Corrected association route.
    validator_assoc_x = 48

    add_route(
        out, routes,
        "ret-validator-assoc",
        [
            ret.a("left"),
            (validator_assoc_x, ret.cy),
            (validator_assoc_x, ventry.cy),
            ventry.a("left", TARGET_GAP),
        ],
        "edge-assoc",
        ignore=("ret", "ventry"),
    )

    branch_label(
        out,
        validator_assoc_x + 7,
        965,
        "optional caller validation",
        COL["assoc"],
        "start",
        8.0,
    )

    # ==================================================================
    # Geometry checks
    # ==================================================================

    audit_bounds(nodes, W, H, 8)
    audit_node_overlaps(nodes)
    audit_routes(routes, nodes)

    assert abs(jsonb.cy - djson.cy) < 1e-9
    assert abs(pickleb.cy - dpickle.cy) < 1e-9
    assert abs(func.cy - dpickle.cy) < 1e-9

    # Association rail remains fully outside the validator block.
    assert validator_assoc_x < ventry.left - TARGET_GAP

    assert H - vret.bottom <= 55

    write(
        "flow_netload_load_ieee_v1",
        W,
        H,
        "Network plugin loader and validator - IEEE",
        "\n".join(out),
    )

def build_load_presentation():
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

    cx = 515

    # ==================================================================
    # Public loader
    #
    # The sequence is spread slightly more vertically than v1 so the
    # lower-left slide area is used rather than leaving a large empty
    # region after the main loader flow.
    # ==================================================================

    start = Rg(
        "start",
        150, 22,
        730, 48,
    )

    draw_node(
        out,
        start,
        COL["blue"],
        "load_network_from_yaml(yaml_path)",
        (
            "portable YAML entry point for a benchmark-ready network + profiles",
        ),
        11.4,
        7.7,
    )

    cfg = Rg(
        "cfg",
        130, 98,
        770, 64,
    )

    draw_node(
        out,
        cfg,
        COL["green"],
        "Parse and validate YAML configuration",
        (
            "source/path or loader function; strategy/year/data hooks; voltage limits",
        ),
        10.5,
        7.2,
    )

    add_route(
        out, routes,
        "start-cfg",
        [
            start.a("bottom"),
            cfg.a("top", TARGET_GAP),
        ],
        ignore=("start", "cfg"),
    )

    # ==================================================================
    # Source dispatch
    # ==================================================================

    djson = Dg(
        "djson",
        cx, 225,
        230, 42,
    )

    draw_decision(
        out,
        djson,
        "source == json?",
        ts=9.8,
    )

    add_route(
        out, routes,
        "cfg-djson",
        [
            cfg.a("bottom"),
            djson.a("top", TARGET_GAP),
        ],
        ignore=("cfg", "djson"),
    )

    jsonb = Rg(
        "jsonb",
        45, 202,
        300, 46,
    )

    draw_node(
        out,
        jsonb,
        COL["green"],
        "pp.from_json(path)",
        (
            "pandapower JSON",
        ),
        8.8,
        6.3,
        8,
    )

    add_route(
        out, routes,
        "djson-json",
        [
            djson.a("left"),
            jsonb.a("right", TARGET_GAP),
        ],
        "edge-gate",
        ignore=("djson", "jsonb"),
    )

    branch_label(
        out,
        (djson.left + jsonb.right) / 2,
        djson.cy - 7,
        "yes",
        COL["gate"],
        size=7.6,
    )

    dp = Dg(
        "dp",
        cx, 305,
        230, 42,
    )

    draw_decision(
        out,
        dp,
        "source == pickle?",
        ts=9.6,
    )

    add_route(
        out, routes,
        "djson-dp",
        [
            djson.a("bottom"),
            dp.a("top", TARGET_GAP),
        ],
        ignore=("djson", "dp"),
    )

    branch_label(
        out,
        djson.cx + 10,
        djson.bottom + 11,
        "no",
        anchor="start",
        size=7.5,
    )

    pb = Rg(
        "pb",
        45, 282,
        300, 46,
    )

    draw_node(
        out,
        pb,
        COL["green"],
        "pp.from_pickle(path)",
        (
            "pandapower-managed pickle",
        ),
        8.6,
        6.2,
        8,
    )

    fn = Rg(
        "fn",
        700, 279,
        310, 52,
    )

    draw_node(
        out,
        fn,
        COL["green"],
        "Import loader file -> call function()",
        (
            "file-location import; zero-arg loader",
        ),
        8.5,
        6.1,
        8,
    )

    add_route(
        out, routes,
        "dp-pb",
        [
            dp.a("left"),
            pb.a("right", TARGET_GAP),
        ],
        "edge-gate",
        ignore=("dp", "pb"),
    )

    branch_label(
        out,
        (dp.left + pb.right) / 2,
        dp.cy - 7,
        "yes",
        COL["gate"],
        size=7.5,
    )

    add_route(
        out, routes,
        "dp-fn",
        [
            dp.a("right"),
            fn.a("left", TARGET_GAP),
        ],
        "edge-gate",
        ignore=("dp", "fn"),
    )

    branch_label(
        out,
        (dp.right + fn.left) / 2,
        dp.cy - 7,
        "no -> function",
        COL["gate"],
        size=7.1,
    )

    # ==================================================================
    # Source merge + type gate
    # ==================================================================

    sm = Cg(
        "sm",
        cx, 395,
        9,
    )

    out.append(
        circle_svg(
            sm,
            COL["neutral"],
        )
    )

    add_route(
        out, routes,
        "json-sm",
        [
            jsonb.a("left"),
            (20, jsonb.cy),
            (20, sm.cy),
            sm.a("left", TARGET_GAP),
        ],
        ignore=("jsonb", "sm"),
    )

    add_route(
        out, routes,
        "pb-sm",
        [
            pb.a("bottom"),
            (pb.cx, 365),
            (sm.cx, 365),
            sm.a("top", TARGET_GAP),
        ],
        ignore=("pb", "sm"),
    )

    add_route(
        out, routes,
        "fn-sm",
        [
            fn.a("bottom"),
            (fn.cx, sm.cy),
            sm.a("right", TARGET_GAP),
        ],
        ignore=("fn", "sm"),
    )

    dtype = Dg(
        "dtype",
        cx, 465,
        220, 40,
    )

    draw_decision(
        out,
        dtype,
        "pandapowerNet?",
        ts=9.5,
    )

    add_route(
        out, routes,
        "sm-dtype",
        [
            sm.a("bottom"),
            dtype.a("top", TARGET_GAP),
        ],
        ignore=("sm", "dtype"),
    )

    bad = Rg(
        "bad",
        740, 443,
        250, 44,
    )

    draw_node(
        out,
        bad,
        COL["red"],
        "Raise TypeError",
        (
            "invalid network result",
        ),
        8.3,
        6.0,
        8,
    )

    add_route(
        out, routes,
        "dtype-bad",
        [
            dtype.a("right"),
            bad.a("left", TARGET_GAP),
        ],
        "edge-hil",
        ignore=("dtype", "bad"),
    )

    branch_label(
        out,
        (dtype.right + bad.left) / 2,
        dtype.cy - 7,
        "no",
        COL["hil"],
        size=7.5,
    )

    # ==================================================================
    # Profiles + metadata + loader return
    # ==================================================================

    strat = Rg(
        "strat",
        220, 525,
        590, 54,
    )

    draw_node(
        out,
        strat,
        COL["green"],
        "Build profiles using configured strategy",
        (
            "simbench_native / dwd_pvlib / flat / custom; fallback handled in Diagram 2",
        ),
        9.5,
        6.7,
    )

    add_route(
        out, routes,
        "dtype-strat",
        [
            dtype.a("bottom"),
            strat.a("top", TARGET_GAP),
        ],
        "edge-gate",
        ignore=("dtype", "strat"),
    )

    branch_label(
        out,
        dtype.cx + 10,
        dtype.bottom + 11,
        "yes",
        COL["gate"],
        "start",
        7.5,
    )

    meta = Rg(
        "meta",
        190, 610,
        650, 58,
    )

    draw_node(
        out,
        meta,
        COL["purple"],
        "Attach plugin_meta with requested + actual strategy",
        (
            "name/label/source/year/limits/notes/yaml_path",
        ),
        9.5,
        6.7,
    )

    ret = Rg(
        "ret",
        250, 700,
        530, 48,
    )

    draw_node(
        out,
        ret,
        COL["purple"],
        "return (net, profiles)",
        (
            "consumed unchanged by benchmark orchestration",
        ),
        9.8,
        6.9,
    )

    add_route(
        out, routes,
        "strat-meta",
        [
            strat.a("bottom"),
            meta.a("top", TARGET_GAP),
        ],
        ignore=("strat", "meta"),
    )

    add_route(
        out, routes,
        "meta-ret",
        [
            meta.a("bottom"),
            ret.a("top", TARGET_GAP),
        ],
        ignore=("meta", "ret"),
    )

    # ==================================================================
    # Separate validator
    #
    # Positioned lower on the slide to make deliberate use of the
    # available 1080p height while preserving visual separation between
    # the two public APIs.
    # ==================================================================

    v = Rg(
        "v",
        150, 900,
        350, 50,
    )

    draw_node(
        out,
        v,
        COL["blue"],
        "validate_network_plugin(net, profiles)",
        (
            "separate public API; caller chooses whether to proceed",
        ),
        8.6,
        6.1,
        8,
    )

    vc = Rg(
        "vc",
        540, 882,
        450, 86,
    )

    draw_node(
        out,
        vc,
        COL["green"],
        "Return compatibility warnings",
        (
            "DER/type, distribution voltage level, transformer and ZIP-load checks",
            "profile/net alignment, missing DER profile columns and NaN checks",
        ),
        8.5,
        6.0,
        8,
    )

    add_route(
        out, routes,
        "v-vc",
        [
            v.a("right"),
            vc.a("left", TARGET_GAP),
        ],
        ignore=("v", "vc"),
    )

    # Correct association geometry retained from v1.
    assoc_x = 110

    add_route(
        out, routes,
        "ret-v-assoc",
        [
            ret.a("left"),
            (assoc_x, ret.cy),
            (assoc_x, v.cy),
            v.a("left", TARGET_GAP),
        ],
        "edge-assoc",
        ignore=("ret", "v"),
    )

    # Missing label restored.
    branch_label(
        out,
        assoc_x + 8,
        825,
        "optional caller validation",
        COL["assoc"],
        "start",
        7.6,
    )

    # ==================================================================
    # Audience panels
    # ==================================================================

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
            "YAML configuration",
            COL["text"],
            17,
            700,
            "start",
        )
    )

    lines = [
        "The YAML defines a unique name/label, one network source and one profile strategy.",
        "JSON/pickle paths, function modules and optional profile data directories resolve relative to the YAML folder.",
        "Valid strategies are simbench_native, dwd_pvlib, flat and custom; custom requires file_map hooks.",
        "The configured year is checked and optional voltage limits default to 0.95/1.05 pu.",
        "Invalid fields, missing files or import errors propagate before a benchmark-ready network is returned.",
    ]

    for i, t in enumerate(lines):
        out.append(
            label(
                px + 24,
                94 + 39 * i,
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
            359,
            "Network loading and output",
            COL["text"],
            17,
            700,
            "start",
        )
    )

    lines = [
        "JSON uses pandapower.from_json; pickle uses pandapower.from_pickle; function sources import a configured Python file and call a zero-argument loader.",
        "Every source must ultimately produce a pandapowerNet before profile construction can proceed.",
        "Profile construction returns both the profile dictionary and the strategy actually used after any fallback.",
        "plugin_meta records both requested_strategy and actual strategy, keeping fallback behavior visible to downstream consumers.",
        "The public loader returns only (net, profiles); it does not itself invoke the compatibility validator.",
    ]

    for i, t in enumerate(lines):
        out.append(
            label(
                px + 24,
                396 + 43 * i,
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
            701,
            "Separate compatibility validator",
            COL["text"],
            17,
            700,
            "start",
        )
    )

    lines = [
        "validate_network_plugin is a separate caller-invoked API and returns warning strings rather than raising for compatibility findings.",
        "Checks cover DER presence/type, MV/LV distribution levels and transformer availability for the benchmark scenarios.",
        "ZIP-load shares are flagged because framework power-flow calls disable voltage-dependent load behavior.",
        "The validator also checks load-profile column alignment, missing PV/wind profile coverage and NaN values.",
        "The orchestration script decides whether warnings require user confirmation before benchmark execution.",
    ]

    for i, t in enumerate(lines):
        out.append(
            label(
                px + 24,
                738 + 50 * i,
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

    assert abs(jsonb.cy - djson.cy) < 1e-9
    assert abs(pb.cy - dp.cy) < 1e-9
    assert abs(fn.cy - dp.cy) < 1e-9

    assert assoc_x < v.left - TARGET_GAP

    # Deliberately use the lower slide rather than ending around y=920.
    assert max(v.bottom, vc.bottom) >= 950
    assert max(v.bottom, vc.bottom) < H - 70

    write(
        "flow_netload_load_presentation_v1",
        W,
        H,
        "Network plugin loader - presentation",
        "\n".join(out),
    )

# =============================================================================
# Diagram 2: profile strategy dispatcher + HC profile factory
# =============================================================================

def build_strategy_ieee():
    W, H = 1180, 1435

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

    cx = 610

    out.append(
        label(
            W / 2,
            38,
            "Network Plugin - Profile Strategy Dispatch and HC Profile Factory",
            COL["text"],
            17,
            700,
        )
    )

    # ==================================================================
    # Profile strategy selection
    # ==================================================================

    start = Rg(
        "start",
        300, 60,
        620, 58,
    )

    draw_node(
        out,
        start,
        COL["blue"],
        "_build_profiles_for_strategy(net, cfg)",
        (
            "strategy = cfg['strategy']",
        ),
        12.2,
        8.6,
    )

    dfb = Dg(
        "dfb",
        cx, 178,
        320, 66,
    )

    draw_decision(
        out,
        dfb,
        "strategy == simbench_native AND",
        (
            "no SimBench net.profiles metadata?",
        ),
        10.7,
        8.0,
    )

    add_route(
        out, routes,
        "start-dfb",
        [
            start.a("bottom"),
            dfb.a("top", TARGET_GAP),
        ],
        ignore=("start", "dfb"),
    )

    fb = Rg(
        "fb",
        795, 145,
        320, 66,
    )

    draw_node(
        out,
        fb,
        COL["dry"],
        "Warn + set strategy = dwd_pvlib",
        (
            "requested strategy remains separately recorded",
        ),
        9.6,
        7.0,
        9,
    )

    add_route(
        out, routes,
        "dfb-fb",
        [
            dfb.a("right"),
            fb.a("left", TARGET_GAP),
        ],
        "edge-dry",
        ignore=("dfb", "fb"),
    )

    branch_label(
        out,
        (dfb.right + fb.left) / 2,
        dfb.cy - 8,
        "yes",
        COL["dry"],
        size=8.3,
    )

    fm = Cg(
        "fm",
        cx, 270,
        11,
    )

    out.append(
        circle_svg(
            fm,
            COL["neutral"],
        )
    )

    add_route(
        out, routes,
        "dfb-no",
        [
            dfb.a("bottom"),
            fm.a("top", TARGET_GAP),
        ],
        ignore=("dfb", "fm"),
    )

    branch_label(
        out,
        dfb.cx + 14,
        dfb.bottom + 14,
        "no",
        anchor="start",
        size=8.3,
    )

    add_route(
        out, routes,
        "fb-fm",
        [
            fb.a("bottom"),
            (fb.cx, fm.cy),
            fm.a("right", TARGET_GAP),
        ],
        "edge-dry",
        ignore=("fb", "fm"),
    )

    dsim = Dg(
        "dsim",
        cx, 350,
        260, 58,
    )

    draw_decision(
        out,
        dsim,
        "strategy == simbench_native?",
        ts=10.6,
    )

    add_route(
        out, routes,
        "fm-dsim",
        [
            fm.a("bottom"),
            dsim.a("top", TARGET_GAP),
        ],
        ignore=("fm", "dsim"),
    )

    # ==================================================================
    # SimBench-native branch
    # ==================================================================

    sb0 = Rg(
        "sb0",
        55, 321,
        320, 58,
    )

    draw_node(
        out,
        sb0,
        COL["green"],
        "sb.get_absolute_values(net)",
        (
            "load and sgen absolute-value profiles",
        ),
        9.7,
        7.0,
        9,
    )

    add_route(
        out, routes,
        "dsim-sb0",
        [
            dsim.a("left"),
            sb0.a("right", TARGET_GAP),
        ],
        "edge-gate",
        ignore=("dsim", "sb0"),
    )

    branch_label(
        out,
        (dsim.left + sb0.right) / 2,
        dsim.cy - 8,
        "yes",
        COL["gate"],
        size=8.3,
    )

    # DatetimeIndex decision.
    dt = Dg(
        "dt",
        215, 455,
        250, 56,
    )

    draw_decision(
        out,
        dt,
        "times is DatetimeIndex?",
        ts=9.7,
    )

    add_route(
        out, routes,
        "sb0-dt",
        [
            sb0.a("bottom"),
            dt.a("top", TARGET_GAP),
        ],
        ignore=("sb0", "dt"),
    )

    # Center rebuild directly below dt.
    # dt.cx = 215 and rebuild.cx = 215, so the "no" branch is straight.
    rebuild = Rg(
        "rebuild",
        50, 525,
        330, 58,
    )

    draw_node(
        out,
        rebuild,
        COL["green"],
        "Rebuild configured-year 15-min DatetimeIndex",
        (
            "only when SimBench returned integer step index",
        ),
        8.8,
        6.4,
        8,
    )

    add_route(
        out, routes,
        "dt-rebuild",
        [
            dt.a("bottom"),
            rebuild.a("top", TARGET_GAP),
        ],
        "edge-gate",
        ignore=("dt", "rebuild"),
    )

    branch_label(
        out,
        dt.cx + 12,
        dt.bottom + 12,
        "no",
        COL["gate"],
        "start",
        7.8,
    )

    sbm = Cg(
        "sbm",
        215, 625,
        9,
    )

    out.append(
        circle_svg(
            sbm,
            COL["neutral"],
        )
    )

    # Yes path bypasses rebuild on the outside.
    dt_yes_x = 25

    add_route(
        out, routes,
        "dt-yes",
        [
            dt.a("left"),
            (dt_yes_x, dt.cy),
            (dt_yes_x, sbm.cy),
            sbm.a("left", TARGET_GAP),
        ],
        "edge-gate",
        ignore=("dt", "sbm"),
    )

    branch_label(
        out,
        dt.left - 18,
        dt.cy - 7,
        "yes",
        COL["gate"],
        "end",
        7.8,
    )

    # Rebuilt time axis joins the same merge vertically.
    add_route(
        out, routes,
        "rebuild-sbm",
        [
            rebuild.a("bottom"),
            sbm.a("top", TARGET_GAP),
        ],
        "edge-gate",
        ignore=("rebuild", "sbm"),
    )

    sbproc = Rg(
        "sbproc",
        35, 660,
        360, 92,
    )

    draw_node(
        out,
        sbproc,
        COL["green"],
        "SimBench post-processing",
        (
            "clip load >= 0; classify PV by pv|solar|lv_res and wind by wind|wp|wka",
            "zero PV at hours >=22 or <=4; clip PV/wind >=0; find_extreme_days",
            "result net_type='simbench'",
        ),
        9.0,
        6.3,
        8,
    )

    add_route(
        out, routes,
        "sbm-proc",
        [
            sbm.a("bottom"),
            sbproc.a("top", TARGET_GAP),
        ],
        ignore=("sbm", "sbproc"),
    )

    # ==================================================================
    # DWD/custom/flat strategy branches
    # ==================================================================

    ddwd = Dg(
        "ddwd",
        cx, 455,
        300, 60,
    )

    draw_decision(
        out,
        ddwd,
        "strategy in ('dwd_pvlib', 'custom')?",
        ts=9.9,
    )

    add_route(
        out, routes,
        "dsim-ddwd",
        [
            dsim.a("bottom"),
            ddwd.a("top", TARGET_GAP),
        ],
        ignore=("dsim", "ddwd"),
    )

    branch_label(
        out,
        dsim.cx + 14,
        dsim.bottom + 14,
        "no",
        anchor="start",
        size=8.3,
    )

    dwd = Rg(
        "dwd",
        790, 422,
        330, 66,
    )

    draw_node(
        out,
        dwd,
        COL["green"],
        "_build_profiles_dwd(net, cfg)",
        (
            "data_dir = cfg override or project DWD default; _dwd_safe_name(name)",
            "build_annual_profiles(..., file_map, col_map); custom uses this same path",
        ),
        9.5,
        6.8,
        9,
    )

    add_route(
        out, routes,
        "ddwd-dwd",
        [
            ddwd.a("right"),
            dwd.a("left", TARGET_GAP),
        ],
        "edge-gate",
        ignore=("ddwd", "dwd"),
    )

    branch_label(
        out,
        (ddwd.right + dwd.left) / 2,
        ddwd.cy - 8,
        "yes",
        COL["gate"],
        size=8.1,
    )

    flat = Rg(
        "flat",
        385, 535,
        450, 72,
    )

    draw_node(
        out,
        flat,
        COL["green"],
        "_build_profiles_flat(net, cfg)",
        (
            "leap-aware full-year 15-min index; constant load.p_mw and sgen.p_mw frames",
            "PV/wind type masks; empty DER groups become empty frames; find_extreme_days",
        ),
        9.6,
        6.9,
        9,
    )

    add_route(
        out, routes,
        "ddwd-flat",
        [
            ddwd.a("bottom"),
            flat.a("top", TARGET_GAP),
        ],
        ignore=("ddwd", "flat"),
    )

    branch_label(
        out,
        ddwd.cx + 14,
        ddwd.bottom + 14,
        "no -> flat",
        anchor="start",
        size=8.1,
    )

    # ==================================================================
    # Strategy output merge
    # ==================================================================

    outm = Cg(
        "outm",
        cx, 815,
        12,
    )

    out.append(
        circle_svg(
            outm,
            COL["neutral"],
        )
    )

    add_route(
        out, routes,
        "sbproc-outm",
        [
            sbproc.a("bottom"),
            (sbproc.cx, outm.cy),
            outm.a("left", TARGET_GAP),
        ],
        ignore=("sbproc", "outm"),
    )

    add_route(
        out, routes,
        "dwd-outm",
        [
            dwd.a("bottom"),
            (dwd.cx, outm.cy),
            outm.a("right", TARGET_GAP),
        ],
        ignore=("dwd", "outm"),
    )

    add_route(
        out, routes,
        "flat-outm",
        [
            flat.a("bottom"),
            (flat.cx, 785),
            (outm.cx, 785),
            outm.a("top", TARGET_GAP),
        ],
        ignore=("flat", "outm"),
    )

    ret = Rg(
        "ret",
        330, 855,
        560, 62,
    )

    draw_node(
        out,
        ret,
        COL["purple"],
        "return (profiles, strategy_used)",
        (
            "actual strategy is available for plugin_meta",
        ),
        10.7,
        7.7,
    )

    add_route(
        out, routes,
        "outm-ret",
        [
            outm.a("bottom"),
            ret.a("top", TARGET_GAP),
        ],
        ignore=("outm", "ret"),
    )

    # ==================================================================
    # HC-stressed profile factory
    #
    # Separate sequential API flow. No "shared dispatcher" annotation is
    # needed. The actual reuse is explicit in the _factory call itself.
    # ==================================================================

    f0 = Rg(
        "f0",
        275, 975,
        670, 58,
    )

    draw_node(
        out,
        f0,
        COL["blue"],
        "make_profile_factory(yaml_path)",
        (
            "resolve path; _load_yaml_config once at factory creation",
        ),
        10.8,
        7.8,
    )

    f1 = Rg(
        "f1",
        305, 1057,
        610, 58,
    )

    draw_node(
        out,
        f1,
        COL["purple"],
        "return closure _factory(net_hc)",
        (
            "BenchmarkConfig.profile_factory invokes it later on the stressed network",
        ),
        10.2,
        7.4,
    )

    f2 = Rg(
        "f2",
        285, 1139,
        650, 66,
    )

    draw_node(
        out,
        f2,
        COL["green"],
        "_factory: call _build_profiles_for_strategy(net_hc, cfg)",
        (
            "uses the configured strategy and the same simbench_native fallback behavior",
        ),
        10.2,
        7.3,
    )

    f3 = Rg(
        "f3",
        265, 1231,
        690, 80,
    )

    draw_node(
        out,
        f3,
        COL["purple"],
        "Attach HC-stressed plugin_meta",
        (
            "name += '_hc_stressed'; preserve label/source/requested strategy/year/limits/notes",
            "store strategy actually used for the stressed network",
        ),
        9.8,
        7.0,
    )

    f4 = Rg(
        "f4",
        330, 1337,
        560, 58,
    )

    draw_node(
        out,
        f4,
        COL["purple"],
        "return rebuilt profiles",
        (
            "fresh coverage for the HC-stressed network",
        ),
        10.5,
        7.5,
    )

    add_route(
        out, routes,
        "f0-f1",
        [
            f0.a("bottom"),
            f1.a("top", TARGET_GAP),
        ],
        ignore=("f0", "f1"),
    )

    add_route(
        out, routes,
        "f1-f2",
        [
            f1.a("bottom"),
            f2.a("top", TARGET_GAP),
        ],
        ignore=("f1", "f2"),
    )

    add_route(
        out, routes,
        "f2-f3",
        [
            f2.a("bottom"),
            f3.a("top", TARGET_GAP),
        ],
        ignore=("f2", "f3"),
    )

    add_route(
        out, routes,
        "f3-f4",
        [
            f3.a("bottom"),
            f4.a("top", TARGET_GAP),
        ],
        ignore=("f3", "f4"),
    )

    # ==================================================================
    # Geometry checks
    # ==================================================================

    audit_bounds(nodes, W, H, 8)
    audit_node_overlaps(nodes)
    audit_routes(routes, nodes)

    assert abs(sb0.cy - dsim.cy) < 1e-9
    assert abs(dwd.cy - ddwd.cy) < 1e-9

    # Datetime rebuild is directly below the decision.
    assert abs(rebuild.cx - dt.cx) < 1e-9
    assert abs(sbm.cx - dt.cx) < 1e-9
    assert dt_yes_x < rebuild.left - TARGET_GAP

    assert H - f4.bottom <= 45

    write(
        "flow_netload_strategy_ieee_v1",
        W,
        H,
        "Network plugin profile strategy and HC factory - IEEE",
        "\n".join(out),
    )


def build_strategy_presentation():
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

    cx = 510

    # ==================================================================
    # Strategy selection
    # ==================================================================

    start = Rg(
        "start",
        160, 18,
        700, 46,
    )

    draw_node(
        out,
        start,
        COL["blue"],
        "_build_profiles_for_strategy(net, cfg)",
        (
            "select configured profile strategy for the supplied network",
        ),
        10.8,
        7.3,
    )

    dfb = Dg(
        "dfb",
        cx, 125,
        300, 48,
    )

    draw_decision(
        out,
        dfb,
        "simbench_native requested AND metadata absent?",
        ts=9.6,
    )

    add_route(
        out, routes,
        "start-dfb",
        [
            start.a("bottom"),
            dfb.a("top", TARGET_GAP),
        ],
        ignore=("start", "dfb"),
    )

    fb = Rg(
        "fb",
        720, 101,
        280, 48,
    )

    draw_node(
        out,
        fb,
        COL["dry"],
        "Warn + use dwd_pvlib",
        (
            "requested strategy remains recorded in metadata",
        ),
        8.4,
        6.1,
        8,
    )

    add_route(
        out, routes,
        "dfb-fb",
        [
            dfb.a("right"),
            fb.a("left", TARGET_GAP),
        ],
        "edge-dry",
        ignore=("dfb", "fb"),
    )

    branch_label(
        out,
        (dfb.right + fb.left) / 2,
        dfb.cy - 7,
        "yes",
        COL["dry"],
        size=7.5,
    )

    fm = Cg(
        "fm",
        cx, 185,
        8,
    )

    out.append(
        circle_svg(
            fm,
            COL["neutral"],
        )
    )

    add_route(
        out, routes,
        "dfb-no",
        [
            dfb.a("bottom"),
            fm.a("top", TARGET_GAP),
        ],
        ignore=("dfb", "fm"),
    )

    branch_label(
        out,
        dfb.cx + 10,
        dfb.bottom + 11,
        "no",
        anchor="start",
        size=7.4,
    )

    add_route(
        out, routes,
        "fb-fm",
        [
            fb.a("bottom"),
            (fb.cx, fm.cy),
            fm.a("right", TARGET_GAP),
        ],
        "edge-dry",
        ignore=("fb", "fm"),
    )

    dsim = Dg(
        "dsim",
        cx, 245,
        230, 40,
    )

    draw_decision(
        out,
        dsim,
        "strategy == simbench_native?",
        ts=9.5,
    )

    add_route(
        out, routes,
        "fm-dsim",
        [
            fm.a("bottom"),
            dsim.a("top", TARGET_GAP),
        ],
        ignore=("fm", "dsim"),
    )

    sb = Rg(
        "sb",
        45, 217,
        320, 56,
    )

    draw_node(
        out,
        sb,
        COL["green"],
        "SimBench native profiles",
        (
            "absolute values; rebuild time index if needed; masks/night-zero/clip/extremes",
        ),
        8.3,
        5.9,
        8,
    )

    add_route(
        out, routes,
        "dsim-sb",
        [
            dsim.a("left"),
            sb.a("right", TARGET_GAP),
        ],
        "edge-gate",
        ignore=("dsim", "sb"),
    )

    branch_label(
        out,
        (dsim.left + sb.right) / 2,
        dsim.cy - 7,
        "yes",
        COL["gate"],
        size=7.4,
    )

    ddwd = Dg(
        "ddwd",
        cx, 315,
        280, 42,
    )

    draw_decision(
        out,
        ddwd,
        "dwd_pvlib OR custom?",
        ts=9.4,
    )

    add_route(
        out, routes,
        "dsim-ddwd",
        [
            dsim.a("bottom"),
            ddwd.a("top", TARGET_GAP),
        ],
        ignore=("dsim", "ddwd"),
    )

    branch_label(
        out,
        dsim.cx + 10,
        dsim.bottom + 11,
        "no",
        anchor="start",
        size=7.4,
    )

    dwd = Rg(
        "dwd",
        710, 289,
        300, 52,
    )

    draw_node(
        out,
        dwd,
        COL["green"],
        "DWD/pvlib profile path",
        (
            "build_annual_profiles; custom passes file_map/col_map hooks",
        ),
        8.5,
        6.0,
        8,
    )

    add_route(
        out, routes,
        "ddwd-dwd",
        [
            ddwd.a("right"),
            dwd.a("left", TARGET_GAP),
        ],
        "edge-gate",
        ignore=("ddwd", "dwd"),
    )

    branch_label(
        out,
        (ddwd.right + dwd.left) / 2,
        ddwd.cy - 7,
        "yes",
        COL["gate"],
        size=7.4,
    )

    flat = Rg(
        "flat",
        320, 365,
        380, 56,
    )

    draw_node(
        out,
        flat,
        COL["green"],
        "Flat full-year profiles",
        (
            "15-min constant load and PV/wind rated P; leap-aware; extremes",
        ),
        8.5,
        6.0,
        8,
    )

    add_route(
        out, routes,
        "ddwd-flat",
        [
            ddwd.a("bottom"),
            flat.a("top", TARGET_GAP),
        ],
        ignore=("ddwd", "flat"),
    )

    branch_label(
        out,
        ddwd.cx + 10,
        ddwd.bottom + 11,
        "no -> flat",
        anchor="start",
        size=7.2,
    )

    # ==================================================================
    # Strategy-result merge
    # ==================================================================

    om = Cg(
        "om",
        cx, 455,
        9,
    )

    out.append(
        circle_svg(
            om,
            COL["neutral"],
        )
    )

    add_route(
        out, routes,
        "sb-om",
        [
            sb.a("bottom"),
            (sb.cx, om.cy),
            om.a("left", TARGET_GAP),
        ],
        ignore=("sb", "om"),
    )

    add_route(
        out, routes,
        "dwd-om",
        [
            dwd.a("bottom"),
            (dwd.cx, om.cy),
            om.a("right", TARGET_GAP),
        ],
        ignore=("dwd", "om"),
    )

    add_route(
        out, routes,
        "flat-om",
        [
            flat.a("bottom"),
            om.a("top", TARGET_GAP),
        ],
        ignore=("flat", "om"),
    )

    ret = Rg(
        "ret",
        220, 485,
        580, 50,
    )

    draw_node(
        out,
        ret,
        COL["purple"],
        "return profiles + strategy actually used",
        (
            "actual strategy is stored in plugin_meta",
        ),
        9.4,
        6.7,
    )

    add_route(
        out, routes,
        "om-ret",
        [
            om.a("bottom"),
            ret.a("top", TARGET_GAP),
        ],
        ignore=("om", "ret"),
    )

    # ==================================================================
    # HC-stressed profile factory
    #
    # No "shared dispatcher" terminology or association arrow is needed.
    # The function call itself shows that the same strategy-building
    # routine is invoked for net_hc.
    #
    # Spread through the lower slide to use the available height.
    # ==================================================================

    f0 = Rg(
        "f0",
        145, 605,
        730, 50,
    )

    draw_node(
        out,
        f0,
        COL["blue"],
        "make_profile_factory(yaml): parse cfg once -> return _factory(net_hc)",
        (
            "used by the HC-stressed recursive benchmark",
        ),
        9.4,
        6.6,
    )

    f1 = Rg(
        "f1",
        165, 710,
        690, 54,
    )

    draw_node(
        out,
        f1,
        COL["green"],
        "_factory calls _build_profiles_for_strategy(net_hc, cfg)",
        (
            "configured strategy and simbench_native fallback behavior are preserved",
        ),
        9.0,
        6.4,
    )

    f2 = Rg(
        "f2",
        175, 820,
        670, 60,
    )

    draw_node(
        out,
        f2,
        COL["purple"],
        "Attach HC-stressed plugin_meta",
        (
            "name gets '_hc_stressed'; actual and requested strategy remain explicit",
        ),
        8.9,
        6.3,
    )

    f3 = Rg(
        "f3",
        220, 935,
        580, 48,
    )

    draw_node(
        out,
        f3,
        COL["purple"],
        "return rebuilt profiles",
        (
            "fresh profiles for the HC-stressed network",
        ),
        9.1,
        6.5,
    )

    add_route(
        out, routes,
        "f0-f1",
        [
            f0.a("bottom"),
            f1.a("top", TARGET_GAP),
        ],
        ignore=("f0", "f1"),
    )

    add_route(
        out, routes,
        "f1-f2",
        [
            f1.a("bottom"),
            f2.a("top", TARGET_GAP),
        ],
        ignore=("f1", "f2"),
    )

    add_route(
        out, routes,
        "f2-f3",
        [
            f2.a("bottom"),
            f3.a("top", TARGET_GAP),
        ],
        ignore=("f2", "f3"),
    )

    # ==================================================================
    # Audience panels
    # ==================================================================

    px, pw = 1080, 800

    panels = [
        R(px, 24, pw, 270),
        R(px, 316, pw, 330),
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
            "Strategy selection and fallback",
            COL["text"],
            17,
            700,
            "start",
        )
    )

    lines = [
        "Profile construction begins from cfg['strategy']; the same configuration is retained when profiles are later rebuilt for net_hc.",
        "Only simbench_native has a runtime fallback: missing net.profiles metadata logs a warning and switches the actual strategy to dwd_pvlib.",
        "requested_strategy remains unchanged in plugin_meta, while strategy records what was actually executed.",
        "After fallback handling, the code follows the exact if / elif / else dispatch shown in the execution graph.",
    ]

    for i, t in enumerate(lines):
        out.append(
            label(
                px + 24,
                94 + 43 * i,
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
            349,
            "Strategy implementations",
            COL["text"],
            17,
            700,
            "start",
        )
    )

    lines = [
        "simbench_native uses sb.get_absolute_values on the already-loaded network.",
        "If its time axis is not a DatetimeIndex, a configured-year 15-minute index is rebuilt.",
        "PV/wind masks, night-time PV zeroing, non-negative clipping and extreme-day detection are then applied.",
        "dwd_pvlib calls build_annual_profiles with the configured/default DWD data directory and a safe network name.",
        "custom is valid and uses the same DWD builder with YAML file_map/col_map hooks.",
        "flat repeats load.p_mw and PV/wind sgen.p_mw over a leap-aware full-year 15-minute index.",
    ]

    for i, t in enumerate(lines):
        out.append(
            label(
                px + 24,
                386 + 39 * i,
                t,
                COL["text"],
                10.5,
                600,
                "start",
            )
        )

    out.append(
        label(
            px + 24,
            701,
            "HC-stressed profile factory",
            COL["text"],
            17,
            700,
            "start",
        )
    )

    lines = [
        "make_profile_factory parses the YAML once when the factory is created and returns a callable for a later stressed network.",
        "When invoked with net_hc, the closure calls _build_profiles_for_strategy(net_hc, cfg) using the configured strategy.",
        "This rebuilds profile columns from the stressed network where the selected strategy supports them and preserves fallback behavior.",
        "The returned metadata adds '_hc_stressed' to the name and preserves source, requested strategy, limits, notes and YAML path.",
        "The actual stressed-network strategy is recorded separately in case simbench_native had to fall back.",
    ]

    for i, t in enumerate(lines):
        out.append(
            label(
                px + 24,
                738 + 50 * i,
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

    assert abs(sb.cy - dsim.cy) < 1e-9
    assert abs(dwd.cy - ddwd.cy) < 1e-9

    # Lower-left content deliberately occupies the available slide area.
    assert f3.bottom >= 980
    assert f3.bottom < H - 60

    write(
        "flow_netload_strategy_presentation_v1",
        W,
        H,
        "Network plugin profile strategy - presentation",
        "\n".join(out),
    )

def main():
    build_load_ieee()
    build_load_presentation()
    build_strategy_ieee()
    build_strategy_presentation()
    print(f"Wrote network plugin flowcharts to {OUT}")


if __name__ == "__main__":
    main()
