from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import html
import math
import xml.etree.ElementTree as ET

import cairosvg

OUT = Path(__file__).resolve().parent / "orchestration_flowcharts_v1"
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
# Direct script
# =============================================================================

def build_script_ieee():
    W, H = 1080, 1415

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
            "Benchmark Orchestration - Direct Script Entry",
            COL["text"],
            17,
            700,
        )
    )

    # ==================================================================
    # Script entry and global overrides
    # ==================================================================

    start = Rg(
        "start",
        250, 60,
        620, 60,
    )

    draw_node(
        out,
        start,
        COL["blue"],
        "run_benchmark_script.py",
        (
            "configure logging; parse CLI arguments",
        ),
        12.8,
        9.0,
    )

    ov = Rg(
        "ov",
        250, 144,
        620, 70,
    )

    draw_node(
        out,
        ov,
        COL["green"],
        "Apply Q(V) and violation-limit overrides",
        (
            "set_qv_parameters(); update V/thermal/angle/unbalance limits",
        ),
        11.8,
        8.5,
    )

    add_route(
        out, routes,
        "start-ov",
        [
            start.a("bottom"),
            ov.a("top", TARGET_GAP),
        ],
        ignore=("start", "ov"),
    )

    # ==================================================================
    # Network source selection
    #
    # The decision and both alternatives share the same centre-y.
    # This gives two straight horizontal branches.
    # ==================================================================

    dnet = Dg(
        "dnet",
        cx, 270,
        260, 64,
    )

    draw_decision(
        out,
        dnet,
        "--network YAML supplied?",
        ts=11.7,
    )

    add_route(
        out, routes,
        "ov-dnet",
        [
            ov.a("bottom"),
            dnet.a("top", TARGET_GAP),
        ],
        ignore=("ov", "dnet"),
    )

    # ------------------------------------------------------------------
    # No YAML -> hardcoded network
    # ------------------------------------------------------------------

    hard = Rg(
        "hard",
        80, 234,
        310, 72,
    )

    draw_node(
        out,
        hard,
        COL["green"],
        "Hardcoded network selector",
        (
            "SimBench/pandapower network chosen in script",
        ),
        10.6,
        8.0,
        10,
    )

    # ------------------------------------------------------------------
    # YAML supplied -> plugin network
    # ------------------------------------------------------------------

    plug = Rg(
        "plug",
        730, 234,
        300, 72,
    )

    draw_node(
        out,
        plug,
        COL["green"],
        "load_network_from_yaml()",
        (
            "returns plugin network + already-built profiles",
        ),
        10.5,
        7.9,
        10,
    )

    # Straight left branch.
    add_route(
        out, routes,
        "dnet-hard",
        [
            dnet.a("left"),
            hard.a("right", TARGET_GAP),
        ],
        "edge-dark",
        ignore=("dnet", "hard"),
    )

    branch_label(
        out,
        (dnet.left + hard.right) / 2,
        dnet.cy - 10,
        "no",
        anchor="middle",
        size=8.8,
    )

    # Straight right branch.
    add_route(
        out, routes,
        "dnet-plug",
        [
            dnet.a("right"),
            plug.a("left", TARGET_GAP),
        ],
        "edge-gate",
        ignore=("dnet", "plug"),
    )

    branch_label(
        out,
        (dnet.right + plug.left) / 2,
        dnet.cy - 10,
        "yes",
        COL["gate"],
        "middle",
        8.8,
    )

    # ==================================================================
    # Hardcoded network profile construction
    # ==================================================================

    prof = Rg(
        "prof",
        80, 330,
        310, 72,
    )

    draw_node(
        out,
        prof,
        COL["green"],
        "build_annual_profiles(net, ...)",
        (
            "only on non-plugin network path",
        ),
        10.6,
        8.0,
        10,
    )

    add_route(
        out, routes,
        "hard-prof",
        [
            hard.a("bottom"),
            prof.a("top", TARGET_GAP),
        ],
        ignore=("hard", "prof"),
    )

    # ==================================================================
    # Plugin network validation / confirmation
    # ==================================================================

    dwarn = Dg(
        "dwarn",
        850, 365,
        255, 62,
    )

    draw_decision(
        out,
        dwarn,
        "validation warnings AND",
        (
            "-y not supplied?",
        ),
        10.6,
        8.0,
    )

    add_route(
        out, routes,
        "plug-dwarn",
        [
            plug.a("bottom"),
            dwarn.a("top", TARGET_GAP),
        ],
        ignore=("plug", "dwarn"),
    )

    dconfirm = Dg(
        "dconfirm",
        850, 445,
        230, 56,
    )

    draw_decision(
        out,
        dconfirm,
        "user confirms?",
        ts=10.8,
    )

    add_route(
        out, routes,
        "dwarn-confirm",
        [
            dwarn.a("bottom"),
            dconfirm.a("top", TARGET_GAP),
        ],
        "edge-gate",
        ignore=("dwarn", "dconfirm"),
    )

    branch_label(
        out,
        dwarn.cx + 14,
        dwarn.bottom + 13,
        "yes",
        COL["gate"],
        "start",
        8.6,
    )

    abort = Rg(
        "abort",
        720, 494,
        260, 58,
    )

    draw_node(
        out,
        abort,
        COL["red"],
        "Abort script",
        (
            "sys.exit(1)",
        ),
        10.3,
        7.8,
        9,
    )

    # User explicitly rejects warnings -> terminal script exit.
    add_route(
        out, routes,
        "confirm-abort",
        [
            dconfirm.a("bottom"),
            abort.a("top", TARGET_GAP),
        ],
        "edge-hil",
        ignore=("dconfirm", "abort"),
    )

    branch_label(
        out,
        dconfirm.cx + 15,
        dconfirm.bottom + 13,
        "no",
        COL["hil"],
        "start",
        8.6,
    )

    # ------------------------------------------------------------------
    # Successful plugin-network merge.
    #
    # pluginm shares dconfirm.cy, so confirmation=yes is a perfectly
    # straight horizontal arrow.
    # ------------------------------------------------------------------

    pluginm = Cg(
        "pluginm",
        690, dconfirm.cy,
        10,
    )

    out.append(
        circle_svg(
            pluginm,
            COL["neutral"],
        )
    )

    # No warnings, or -y already accepts them.
    add_route(
        out, routes,
        "dwarn-no-pluginm",
        [
            dwarn.a("left"),
            (pluginm.cx, dwarn.cy),
            pluginm.a("top", TARGET_GAP),
        ],
        "edge-gate",
        ignore=("dwarn", "pluginm"),
    )

    branch_label(
        out,
        dwarn.left - 18,
        dwarn.cy - 9,
        "no",
        COL["gate"],
        "end",
        8.5,
    )

    # User explicitly confirms warnings.
    add_route(
        out, routes,
        "confirm-yes-pluginm",
        [
            dconfirm.a("left"),
            pluginm.a("right", TARGET_GAP),
        ],
        "edge-gate",
        ignore=("dconfirm", "pluginm"),
    )

    branch_label(
        out,
        (dconfirm.left + pluginm.right) / 2,
        dconfirm.cy - 9,
        "yes",
        COL["gate"],
        "middle",
        8.5,
    )

    # ==================================================================
    # Merge hardcoded and successful plugin paths
    # ==================================================================

    nmerge = Cg(
        "nmerge",
        cx, 520,
        12,
    )

    out.append(
        circle_svg(
            nmerge,
            COL["neutral"],
        )
    )

    add_route(
        out, routes,
        "prof-merge",
        [
            prof.a("bottom"),
            (prof.cx, nmerge.cy),
            nmerge.a("left", TARGET_GAP),
        ],
        ignore=("prof", "nmerge"),
    )

    add_route(
        out, routes,
        "pluginm-nmerge",
        [
            pluginm.a("bottom"),
            (pluginm.cx, nmerge.cy),
            nmerge.a("right", TARGET_GAP),
        ],
        "edge-gate",
        ignore=("pluginm", "nmerge"),
    )

    # ==================================================================
    # Optional oversizing
    #
    # Decision and both alternatives share one centre-y.
    # ==================================================================

    dos = Dg(
        "dos",
        cx, 585,
        250, 60,
    )

    draw_decision(
        out,
        dos,
        "OVERSIZE_FACTOR set?",
        ts=11.2,
    )

    add_route(
        out, routes,
        "merge-dos",
        [
            nmerge.a("bottom"),
            dos.a("top", TARGET_GAP),
        ],
        ignore=("nmerge", "dos"),
    )

    scale = Rg(
        "scale",
        70, 548,
        320, 74,
    )

    draw_node(
        out,
        scale,
        COL["green"],
        "Deep-copy + scale sgen P/sn_mva",
        (
            "scale matching PV/wind profile columns by same factor",
        ),
        10.3,
        7.7,
        10,
    )

    noscale = Rg(
        "noscale",
        730, 554,
        280, 62,
    )

    draw_node(
        out,
        noscale,
        COL["neutral"],
        "net_os = net",
        (
            "current default: OVERSIZE_FACTOR=None",
        ),
        9.8,
        7.4,
        9,
    )

    # Straight yes branch.
    add_route(
        out, routes,
        "dos-scale",
        [
            dos.a("left"),
            scale.a("right", TARGET_GAP),
        ],
        "edge-gate",
        ignore=("dos", "scale"),
    )

    branch_label(
        out,
        (dos.left + scale.right) / 2,
        dos.cy - 9,
        "yes",
        COL["gate"],
        "middle",
        8.6,
    )

    # Straight no branch.
    add_route(
        out, routes,
        "dos-noscale",
        [
            dos.a("right"),
            noscale.a("left", TARGET_GAP),
        ],
        ignore=("dos", "noscale"),
    )

    branch_label(
        out,
        (dos.right + noscale.left) / 2,
        dos.cy - 9,
        "no",
        COL["text"],
        "middle",
        8.6,
    )

    osmerge = Cg(
        "osmerge",
        cx, 665,
        11,
    )

    out.append(
        circle_svg(
            osmerge,
            COL["neutral"],
        )
    )

    add_route(
        out, routes,
        "scale-osmerge",
        [
            scale.a("bottom"),
            (scale.cx, osmerge.cy),
            osmerge.a("left", TARGET_GAP),
        ],
        ignore=("scale", "osmerge"),
    )

    add_route(
        out, routes,
        "noscale-osmerge",
        [
            noscale.a("bottom"),
            (noscale.cx, osmerge.cy),
            osmerge.a("right", TARGET_GAP),
        ],
        ignore=("noscale", "osmerge"),
    )

    # ==================================================================
    # Load scaling + benchmark/publication setup
    # ==================================================================

    loadscale = Rg(
        "loadscale",
        250, 700,
        620, 58,
    )

    draw_node(
        out,
        loadscale,
        COL["green"],
        "Apply LOAD_SCALE to net_os loads and profiles['load']",
        (
            "active script operation; LOAD_SCALE currently 1.0",
        ),
        10.8,
        7.9,
    )

    setup = Rg(
        "setup",
        220, 782,
        680, 74,
    )

    draw_node(
        out,
        setup,
        COL["blue"],
        "Create PublishHandles + BenchmarkConfig",
        (
            "outer/HC-stressed output dirs; scenario/HC/profile-factory settings",
        ),
        11.0,
        8.0,
    )

    prepub = Rg(
        "prepub",
        220, 880,
        680, 64,
    )

    draw_node(
        out,
        prepub,
        COL["purple"],
        "Scaling check + publish_topology_and_profiles",
        (
            "write network topology/profile artefacts before benchmark execution",
        ),
        10.7,
        7.8,
    )

    add_route(
        out, routes,
        "osmerge-load",
        [
            osmerge.a("bottom"),
            loadscale.a("top", TARGET_GAP),
        ],
        ignore=("osmerge", "loadscale"),
    )

    add_route(
        out, routes,
        "load-setup",
        [
            loadscale.a("bottom"),
            setup.a("top", TARGET_GAP),
        ],
        ignore=("loadscale", "setup"),
    )

    add_route(
        out, routes,
        "setup-prepub",
        [
            setup.a("bottom"),
            prepub.a("top", TARGET_GAP),
        ],
        ignore=("setup", "prepub"),
    )

    # ==================================================================
    # Controller plugin vs built-in benchmark
    #
    # Again, decision and both alternatives share one centre-y.
    # ==================================================================

    dctrl = Dg(
        "dctrl",
        cx, 1000,
        260, 62,
    )

    draw_decision(
        out,
        dctrl,
        "--controller YAML supplied?",
        ts=11.2,
    )

    add_route(
        out, routes,
        "prepub-dctrl",
        [
            prepub.a("bottom"),
            dctrl.a("top", TARGET_GAP),
        ],
        ignore=("prepub", "dctrl"),
    )

    pluginrun = Rg(
        "pluginrun",
        60, 962,
        330, 76,
    )

    draw_node(
        out,
        pluginrun,
        COL["green"],
        "plugin_runner.register_and_run",
        (
            "temporary plugin scenario + benchmark; "
            "returns custom_result, result",
        ),
        10.3,
        7.7,
        10,
    )

    bench = Rg(
        "bench",
        730, 966,
        290, 68,
    )

    draw_node(
        out,
        bench,
        COL["blue"],
        "run_benchmark(...)",
        (
            "built-in orchestration path",
        ),
        10.8,
        8.0,
        10,
    )

    # Straight yes branch.
    add_route(
        out, routes,
        "dctrl-plugin",
        [
            dctrl.a("left"),
            pluginrun.a("right", TARGET_GAP),
        ],
        "edge-gate",
        ignore=("dctrl", "pluginrun"),
    )

    branch_label(
        out,
        (dctrl.left + pluginrun.right) / 2,
        dctrl.cy - 9,
        "yes",
        COL["gate"],
        "middle",
        8.6,
    )

    # Straight no branch.
    add_route(
        out, routes,
        "dctrl-bench",
        [
            dctrl.a("right"),
            bench.a("left", TARGET_GAP),
        ],
        ignore=("dctrl", "bench"),
    )

    branch_label(
        out,
        (dctrl.right + bench.left) / 2,
        dctrl.cy - 9,
        "no",
        COL["text"],
        "middle",
        8.6,
    )

    rmerge = Cg(
        "rmerge",
        cx, 1075,
        12,
    )

    out.append(
        circle_svg(
            rmerge,
            COL["neutral"],
        )
    )

    add_route(
        out, routes,
        "plugin-rmerge",
        [
            pluginrun.a("bottom"),
            (pluginrun.cx, rmerge.cy),
            rmerge.a("left", TARGET_GAP),
        ],
        ignore=("pluginrun", "rmerge"),
    )

    add_route(
        out, routes,
        "bench-rmerge",
        [
            bench.a("bottom"),
            (bench.cx, rmerge.cy),
            rmerge.a("right", TARGET_GAP),
        ],
        ignore=("bench", "rmerge"),
    )

    # ==================================================================
    # Post-run publishing
    # ==================================================================

    postpub = Rg(
        "postpub",
        220, 1110,
        680, 62,
    )

    draw_node(
        out,
        postpub,
        COL["purple"],
        "publish_hc_and_comparison(result)",
        (
            "write outer benchmark comparison and HC artefacts",
        ),
        10.8,
        8.0,
    )

    add_route(
        out, routes,
        "rmerge-postpub",
        [
            rmerge.a("bottom"),
            postpub.a("top", TARGET_GAP),
        ],
        ignore=("rmerge", "postpub"),
    )

    dhcout = Dg(
        "dhcout",
        cx, 1220,
        300, 62,
    )

    draw_decision(
        out,
        dhcout,
        "hc_benchmark AND net_hc available?",
        ts=10.7,
    )

    add_route(
        out, routes,
        "postpub-dhcout",
        [
            postpub.a("bottom"),
            dhcout.a("top", TARGET_GAP),
        ],
        ignore=("postpub", "dhcout"),
    )

    # Centre-aligned with the decision and moved farther right.
    # This leaves enough visible arrow shaft for the `yes` label.
    pubhc = Rg(
        "pubhc",
        750, 1188,
        290, 64,
    )

    draw_node(
        out,
        pubhc,
        COL["purple"],
        "Publish HC-stressed outputs",
        (
            "publish_hc_and_comparison(result.hc_benchmark)",
        ),
        9.8,
        7.3,
        9,
    )

    add_route(
        out, routes,
        "dhcout-pubhc",
        [
            dhcout.a("right"),
            pubhc.a("left", TARGET_GAP),
        ],
        "edge-gate",
        ignore=("dhcout", "pubhc"),
    )

    branch_label(
        out,
        (dhcout.right + pubhc.left) / 2,
        dhcout.cy - 9,
        "yes",
        COL["gate"],
        "middle",
        8.6,
    )

    endmerge = Cg(
        "endmerge",
        cx, 1295,
        11,
    )

    out.append(
        circle_svg(
            endmerge,
            COL["neutral"],
        )
    )

    add_route(
        out, routes,
        "dhcout-no",
        [
            dhcout.a("bottom"),
            endmerge.a("top", TARGET_GAP),
        ],
        ignore=("dhcout", "endmerge"),
    )

    branch_label(
        out,
        dhcout.cx + 16,
        dhcout.bottom + 15,
        "no",
        anchor="start",
        size=8.5,
    )

    add_route(
        out, routes,
        "pubhc-merge",
        [
            pubhc.a("bottom"),
            (pubhc.cx, endmerge.cy),
            endmerge.a("right", TARGET_GAP),
        ],
        "edge-gate",
        ignore=("pubhc", "endmerge"),
    )

    report = Rg(
        "report",
        250, 1325,
        620, 58,
    )

    draw_node(
        out,
        report,
        COL["purple"],
        "Print comparison / CSV / HC / error summaries",
        (
            "conditional result sections are reported when present",
        ),
        10.6,
        7.8,
    )

    add_route(
        out, routes,
        "end-report",
        [
            endmerge.a("bottom"),
            report.a("top", TARGET_GAP),
        ],
        ignore=("endmerge", "report"),
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

    # Three major side-branch groups are intentionally centre-aligned.
    assert abs(hard.cy - dnet.cy) < 1e-9
    assert abs(plug.cy - dnet.cy) < 1e-9

    assert abs(scale.cy - dos.cy) < 1e-9
    assert abs(noscale.cy - dos.cy) < 1e-9

    assert abs(pluginrun.cy - dctrl.cy) < 1e-9
    assert abs(bench.cy - dctrl.cy) < 1e-9

    # Confirmation=yes enters the plugin merge horizontally.
    assert abs(pluginm.cy - dconfirm.cy) < 1e-9

    # HC publication branch is horizontal and retains enough room
    # to place its decision label on the arrow itself.
    assert abs(pubhc.cy - dhcout.cy) < 1e-9
    assert pubhc.left - dhcout.right >= 35

    # Compact lower canvas without crowding the final node.
    assert H - report.bottom <= 35

    write(
        "flow_orch_script_ieee_v1",
        W,
        H,
        "Direct benchmark script orchestration - IEEE",
        "\n".join(out),
    )

def build_script_presentation():
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
    # Entry configuration
    # ==================================================================

    start = Rg(
        "start",
        160, 18,
        680, 44,
    )

    draw_node(
        out,
        start,
        COL["blue"],
        "run_benchmark_script.py: logging + CLI arguments",
        (
            "network/controller YAML and Q(V)/limit overrides",
        ),
        11.2,
        7.5,
    )

    ov = Rg(
        "ov",
        160, 82,
        680, 46,
    )

    draw_node(
        out,
        ov,
        COL["green"],
        "Apply controller and violation-limit overrides",
        (
            "values are set before any controller/scenario is constructed",
        ),
        10.5,
        7.2,
    )

    add_route(
        out, routes,
        "s-o",
        [
            start.a("bottom"),
            ov.a("top", TARGET_GAP),
        ],
        ignore=("start", "ov"),
    )

    # ==================================================================
    # Network source selection
    #
    # Decision and both alternatives share one centreline.
    # ==================================================================

    dnet = Dg(
        "dnet",
        cx, 180,
        230, 42,
    )

    draw_decision(
        out,
        dnet,
        "--network supplied?",
        ts=10.4,
    )

    add_route(
        out, routes,
        "o-d",
        [
            ov.a("bottom"),
            dnet.a("top", TARGET_GAP),
        ],
        ignore=("ov", "dnet"),
    )

    hard = Rg(
        "hard",
        50, 155,
        300, 50,
    )

    draw_node(
        out,
        hard,
        COL["green"],
        "Hardcoded network + annual profiles",
        (
            "SimBench/pandapower selector + build_annual_profiles",
        ),
        9.1,
        6.5,
        9,
    )

    plug = Rg(
        "plug",
        650, 153,
        330, 54,
    )

    draw_node(
        out,
        plug,
        COL["green"],
        "Network plugin -> net + profiles",
        (
            "network_plugin resolves both according to YAML strategy",
        ),
        9.0,
        6.5,
        9,
    )

    # Straight no branch.
    add_route(
        out, routes,
        "d-hard",
        [
            dnet.a("left"),
            hard.a("right", TARGET_GAP),
        ],
        ignore=("dnet", "hard"),
    )

    branch_label(
        out,
        (dnet.left + hard.right) / 2,
        dnet.cy - 7,
        "no",
        size=8.0,
    )

    # Straight yes branch.
    add_route(
        out, routes,
        "d-plug",
        [
            dnet.a("right"),
            plug.a("left", TARGET_GAP),
        ],
        "edge-gate",
        ignore=("dnet", "plug"),
    )

    branch_label(
        out,
        (dnet.right + plug.left) / 2,
        dnet.cy - 7,
        "yes",
        COL["gate"],
        "middle",
        8.0,
    )

    # ==================================================================
    # Plugin-network validation and confirmation
    # ==================================================================

    dwarn = Dg(
        "dwarn",
        815, 255,
        250, 44,
    )

    draw_decision(
        out,
        dwarn,
        "validation warnings AND",
        (
            "-y not supplied?",
        ),
        9.5,
        7.2,
    )

    add_route(
        out, routes,
        "plug-warn",
        [
            plug.a("bottom"),
            dwarn.a("top", TARGET_GAP),
        ],
        ignore=("plug", "dwarn"),
    )

    dconfirm = Dg(
        "dconfirm",
        815, 320,
        230, 40,
    )

    draw_decision(
        out,
        dconfirm,
        "user confirms?",
        ts=9.6,
    )

    add_route(
        out, routes,
        "warn-confirm",
        [
            dwarn.a("bottom"),
            dconfirm.a("top", TARGET_GAP),
        ],
        "edge-gate",
        ignore=("dwarn", "dconfirm"),
    )

    branch_label(
        out,
        dwarn.cx + 11,
        dwarn.bottom + 11,
        "yes",
        COL["gate"],
        "start",
        7.5,
    )

    abort = Rg(
        "abort",
        690, 355,
        250, 44,
    )

    draw_node(
        out,
        abort,
        COL["red"],
        "Abort before benchmark",
        (
            "sys.exit(1)",
        ),
        8.5,
        6.2,
        8,
    )

    add_route(
        out, routes,
        "confirm-abort",
        [
            dconfirm.a("bottom"),
            abort.a("top", TARGET_GAP),
        ],
        "edge-hil",
        ignore=("dconfirm", "abort"),
    )

    branch_label(
        out,
        dconfirm.cx + 11,
        dconfirm.bottom + 11,
        "no",
        COL["hil"],
        "start",
        7.4,
    )

    # ------------------------------------------------------------------
    # Successful plugin-network merge.
    #
    # Same y as dconfirm so confirm=yes is completely horizontal.
    # ------------------------------------------------------------------

    pluginm = Cg(
        "pluginm",
        650, dconfirm.cy,
        8,
    )

    out.append(
        circle_svg(
            pluginm,
            COL["neutral"],
        )
    )

    # No validation confirmation required.
    add_route(
        out, routes,
        "warn-no-pluginm",
        [
            dwarn.a("left"),
            (pluginm.cx, dwarn.cy),
            pluginm.a("top", TARGET_GAP),
        ],
        "edge-gate",
        ignore=("dwarn", "pluginm"),
    )

    branch_label(
        out,
        dwarn.left - 16,
        dwarn.cy - 7,
        "no",
        COL["gate"],
        "end",
        7.4,
    )

    # User explicitly accepts warnings.
    add_route(
        out, routes,
        "confirm-yes-pluginm",
        [
            dconfirm.a("left"),
            pluginm.a("right", TARGET_GAP),
        ],
        "edge-gate",
        ignore=("dconfirm", "pluginm"),
    )

    branch_label(
        out,
        (dconfirm.left + pluginm.right) / 2,
        dconfirm.cy - 7,
        "yes",
        COL["gate"],
        "middle",
        7.4,
    )

    # ==================================================================
    # Network-path merge
    # ==================================================================

    merge = Cg(
        "merge",
        cx, 400,
        9,
    )

    out.append(
        circle_svg(
            merge,
            COL["neutral"],
        )
    )

    add_route(
        out, routes,
        "hard-m",
        [
            hard.a("bottom"),
            (hard.cx, merge.cy),
            merge.a("left", TARGET_GAP),
        ],
        ignore=("hard", "merge"),
    )

    add_route(
        out, routes,
        "pluginm-m",
        [
            pluginm.a("bottom"),
            (pluginm.cx, merge.cy),
            merge.a("right", TARGET_GAP),
        ],
        "edge-gate",
        ignore=("pluginm", "merge"),
    )

    # ==================================================================
    # Optional oversizing
    #
    # Decision and both alternatives share one centreline.
    # ==================================================================

    dos = Dg(
        "dos",
        cx, 470,
        210, 38,
    )

    draw_decision(
        out,
        dos,
        "OVERSIZE_FACTOR set?",
        ts=9.7,
    )

    add_route(
        out, routes,
        "m-dos",
        [
            merge.a("bottom"),
            dos.a("top", TARGET_GAP),
        ],
        ignore=("merge", "dos"),
    )

    scale = Rg(
        "scale",
        55, 447,
        300, 46,
    )

    draw_node(
        out,
        scale,
        COL["green"],
        "Copy + scale sgen P/sn_mva + PV/wind profiles",
        (
            "matching profile columns use the same factor",
        ),
        8.4,
        6.1,
        8,
    )

    nos = Rg(
        "nos",
        690, 449,
        275, 42,
    )

    draw_node(
        out,
        nos,
        COL["neutral"],
        "Use original net",
        (
            "current default",
        ),
        8.8,
        6.4,
        8,
    )

    # Straight yes branch.
    add_route(
        out, routes,
        "dos-scale",
        [
            dos.a("left"),
            scale.a("right", TARGET_GAP),
        ],
        "edge-gate",
        ignore=("dos", "scale"),
    )

    branch_label(
        out,
        (dos.left + scale.right) / 2,
        dos.cy - 6,
        "yes",
        COL["gate"],
        "middle",
        7.8,
    )

    # Straight no branch.
    add_route(
        out, routes,
        "dos-nos",
        [
            dos.a("right"),
            nos.a("left", TARGET_GAP),
        ],
        ignore=("dos", "nos"),
    )

    branch_label(
        out,
        (dos.right + nos.left) / 2,
        dos.cy - 6,
        "no",
        COL["text"],
        "middle",
        7.8,
    )

    m2 = Cg(
        "m2",
        cx, 540,
        8,
    )

    out.append(
        circle_svg(
            m2,
            COL["neutral"],
        )
    )

    add_route(
        out, routes,
        "scale-m2",
        [
            scale.a("bottom"),
            (scale.cx, m2.cy),
            m2.a("left", TARGET_GAP),
        ],
        ignore=("scale", "m2"),
    )

    add_route(
        out, routes,
        "nos-m2",
        [
            nos.a("bottom"),
            (nos.cx, m2.cy),
            m2.a("right", TARGET_GAP),
        ],
        ignore=("nos", "m2"),
    )

    # ==================================================================
    # Load scaling + benchmark/publication configuration
    # ==================================================================

    setup = Rg(
        "setup",
        155, 575,
        690, 58,
    )

    draw_node(
        out,
        setup,
        COL["blue"],
        "Apply load scale; create PublishHandles + BenchmarkConfig",
        (
            "configure outer benchmark, HC and optional stressed re-benchmark",
        ),
        10.1,
        7.0,
    )

    prepub = Rg(
        "prepub",
        155, 660,
        690, 50,
    )

    draw_node(
        out,
        prepub,
        COL["purple"],
        "Publish topology + profiles before benchmark",
        (
            "also print sgen scaling check",
        ),
        9.6,
        6.8,
    )

    add_route(
        out, routes,
        "m2-setup",
        [
            m2.a("bottom"),
            setup.a("top", TARGET_GAP),
        ],
        ignore=("m2", "setup"),
    )

    add_route(
        out, routes,
        "setup-pre",
        [
            setup.a("bottom"),
            prepub.a("top", TARGET_GAP),
        ],
        ignore=("setup", "prepub"),
    )

    # ==================================================================
    # Controller-plugin dispatch
    #
    # Decision and alternatives again share one centreline.
    # ==================================================================

    dctrl = Dg(
        "dctrl",
        cx, 765,
        230, 42,
    )

    draw_decision(
        out,
        dctrl,
        "--controller supplied?",
        ts=10.0,
    )

    add_route(
        out, routes,
        "pre-dctrl",
        [
            prepub.a("bottom"),
            dctrl.a("top", TARGET_GAP),
        ],
        ignore=("prepub", "dctrl"),
    )

    pr = Rg(
        "pr",
        55, 740,
        310, 50,
    )

    draw_node(
        out,
        pr,
        COL["green"],
        "register_and_run(plugin YAML)",
        (
            "plugin scenario joins configured benchmark",
        ),
        9.0,
        6.6,
        9,
    )

    br = Rg(
        "br",
        690, 742,
        275, 46,
    )

    draw_node(
        out,
        br,
        COL["blue"],
        "run_benchmark(...)",
        (
            "standard orchestration",
        ),
        9.2,
        6.8,
        9,
    )

    # Straight plugin branch.
    add_route(
        out, routes,
        "ctrl-pr",
        [
            dctrl.a("left"),
            pr.a("right", TARGET_GAP),
        ],
        "edge-gate",
        ignore=("dctrl", "pr"),
    )

    branch_label(
        out,
        (dctrl.left + pr.right) / 2,
        dctrl.cy - 7,
        "yes",
        COL["gate"],
        "middle",
        8.0,
    )

    # Straight built-in branch.
    add_route(
        out, routes,
        "ctrl-br",
        [
            dctrl.a("right"),
            br.a("left", TARGET_GAP),
        ],
        ignore=("dctrl", "br"),
    )

    branch_label(
        out,
        (dctrl.right + br.left) / 2,
        dctrl.cy - 7,
        "no",
        COL["text"],
        "middle",
        8.0,
    )

    mr = Cg(
        "mr",
        cx, 835,
        9,
    )

    out.append(
        circle_svg(
            mr,
            COL["neutral"],
        )
    )

    add_route(
        out, routes,
        "pr-mr",
        [
            pr.a("bottom"),
            (pr.cx, mr.cy),
            mr.a("left", TARGET_GAP),
        ],
        ignore=("pr", "mr"),
    )

    add_route(
        out, routes,
        "br-mr",
        [
            br.a("bottom"),
            (br.cx, mr.cy),
            mr.a("right", TARGET_GAP),
        ],
        ignore=("br", "mr"),
    )

    # ==================================================================
    # Post-run publication
    # ==================================================================

    post = Rg(
        "post",
        155, 865,
        690, 50,
    )

    draw_node(
        out,
        post,
        COL["purple"],
        "Publish outer benchmark + HC comparison artefacts",
        (
            "publish_hc_and_comparison(result)",
        ),
        9.6,
        6.8,
    )

    add_route(
        out, routes,
        "mr-post",
        [
            mr.a("bottom"),
            post.a("top", TARGET_GAP),
        ],
        ignore=("mr", "post"),
    )

    # ------------------------------------------------------------------
    # HC-stressed output publication remains explicit because this is
    # a real executable decision in the script.
    # ------------------------------------------------------------------

    dhcout = Dg(
        "dhcout",
        cx, 955,
        270, 40,
    )

    draw_decision(
        out,
        dhcout,
        "hc_benchmark AND net_hc available?",
        ts=9.2,
    )

    add_route(
        out, routes,
        "post-hcout",
        [
            post.a("bottom"),
            dhcout.a("top", TARGET_GAP),
        ],
        ignore=("post", "dhcout"),
    )

    # Centre-aligned with decision, with enough shaft for "yes".
    pubhc = Rg(
        "pubhc",
        710, 932,
        290, 46,
    )

    draw_node(
        out,
        pubhc,
        COL["purple"],
        "Publish HC-stressed outputs",
        (
            "write recursive benchmark comparison artefacts",
        ),
        8.5,
        6.2,
        8,
    )

    add_route(
        out, routes,
        "hcout-pubhc",
        [
            dhcout.a("right"),
            pubhc.a("left", TARGET_GAP),
        ],
        "edge-gate",
        ignore=("dhcout", "pubhc"),
    )

    branch_label(
        out,
        (dhcout.right + pubhc.left) / 2,
        dhcout.cy - 7,
        "yes",
        COL["gate"],
        "middle",
        7.8,
    )

    endm = Cg(
        "endm",
        cx, 1015,
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
        "hcout-no",
        [
            dhcout.a("bottom"),
            endm.a("top", TARGET_GAP),
        ],
        ignore=("dhcout", "endm"),
    )

    branch_label(
        out,
        dhcout.cx + 10,
        dhcout.bottom + 11,
        "no",
        anchor="start",
        size=7.6,
    )

    add_route(
        out, routes,
        "pubhc-endm",
        [
            pubhc.a("bottom"),
            (pubhc.cx, endm.cy),
            endm.a("right", TARGET_GAP),
        ],
        "edge-gate",
        ignore=("pubhc", "endm"),
    )

    # Final block deliberately reaches almost to the slide bottom.
    report = Rg(
        "report",
        155, 1036,
        690, 40,
    )

    draw_node(
        out,
        report,
        COL["purple"],
        "Print final comparison / CSV / HC / error summaries",
        (),
        9.2,
    )

    add_route(
        out, routes,
        "end-report",
        [
            endm.a("bottom"),
            report.a("top", TARGET_GAP),
        ],
        ignore=("endm", "report"),
    )

    # ==================================================================
    # Audience-facing explanatory panels
    # ==================================================================

    px, pw = 1080, 800

    panels = [
        R(px, 24, pw, 250),
        R(px, 296, pw, 330),
        R(px, 648, pw, 390),
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

    # ------------------------------------------------------------------
    # Panel 1
    # ------------------------------------------------------------------

    out.append(
        label(
            px + 24,
            57,
            "Entry configuration",
            COL["text"],
            17,
            700,
            "start",
        )
    )

    lines = [
        "The direct script configures logging, parses optional network/controller YAML paths and accepts runtime Q(V) and planning-limit overrides.",
        "Q(V) breakpoints and q-ratio are applied before any controller is constructed, so software and Arduino configuration use the same values.",
        "Voltage and thermal limits are updated in violation_detector before benchmark dispatch.",
        "The script creates separate publisher handles for the outer network and an optional HC-stressed network.",
        "BenchmarkConfig carries scenario selection, dry-run/serial settings, HC controls, profile factory, checkpointing and publishing handles.",
    ]

    for i, t in enumerate(lines):
        out.append(
            label(
                px + 24,
                92 + 34 * i,
                t,
                COL["text"],
                11.2,
                600,
                "start",
            )
        )

    # ------------------------------------------------------------------
    # Panel 2
    # ------------------------------------------------------------------

    out.append(
        label(
            px + 24,
            329,
            "Network and profile resolution",
            COL["text"],
            17,
            700,
            "start",
        )
    )

    lines = [
        "With --network, network_plugin loads both the network and profiles according to the YAML strategy, then validation warnings are reported.",
        "If validation warnings require confirmation and -y is absent, the user can abort before any benchmark execution.",
        "Without --network, the selected built-in network is loaded and build_annual_profiles creates the annual profile set.",
        "Generation oversizing is conditional; sgen P/sn_mva and matching PV/wind profile columns use the same scale factor.",
        "LOAD_SCALE is then applied consistently to network loads and the load profile before topology/profile publication.",
    ]

    for i, t in enumerate(lines):
        out.append(
            label(
                px + 24,
                364 + 42 * i,
                t,
                COL["text"],
                11.1,
                600,
                "start",
            )
        )

    # ------------------------------------------------------------------
    # Panel 3
    # ------------------------------------------------------------------

    out.append(
        label(
            px + 24,
            681,
            "Benchmark dispatch and outputs",
            COL["text"],
            17,
            700,
            "start",
        )
    )

    lines = [
        "Topology and profiles are published once before the benchmark so the saved model corresponds to the network entering orchestration.",
        "With --controller, plugin_runner temporarily registers the custom controller and runs it alongside BenchmarkConfig.scenarios.",
        "Without --controller, the script calls benchmark_runner.run_benchmark directly.",
        "After execution, publish_hc_and_comparison writes the outer comparison/HC artefacts.",
        "If hc_benchmark and net_hc are available, the recursive benchmark artefacts are written to the separate HC-stressed output directory.",
        "The script finally prints comparison, CSV, hosting-capacity, error and HC-stressed summaries when those result sections are available.",
    ]

    for i, t in enumerate(lines):
        out.append(
            label(
                px + 24,
                716 + 47 * i,
                t,
                COL["text"],
                11.0,
                600,
                "start",
            )
        )

    # ==================================================================
    # Geometry checks
    # ==================================================================

    audit_bounds(
        nodes,
        W,
        H,
        0,
    )

    audit_node_overlaps(
        nodes
    )

    audit_routes(
        routes,
        nodes,
    )

    # Major decision alternatives are deliberately centre-aligned.
    assert abs(hard.cy - dnet.cy) < 1e-9
    assert abs(plug.cy - dnet.cy) < 1e-9

    assert abs(scale.cy - dos.cy) < 1e-9
    assert abs(nos.cy - dos.cy) < 1e-9

    assert abs(pr.cy - dctrl.cy) < 1e-9
    assert abs(br.cy - dctrl.cy) < 1e-9

    # Confirmation=yes enters its merge horizontally.
    assert abs(pluginm.cy - dconfirm.cy) < 1e-9

    # HC-stressed publication branch is horizontal and has enough
    # visible shaft for the yes label.
    assert abs(pubhc.cy - dhcout.cy) < 1e-9
    assert pubhc.left - dhcout.right >= 35

    # The flow now intentionally occupies essentially the full
    # presentation height rather than leaving a large empty lower area.
    assert H - report.bottom <= 5

    write(
        "flow_orch_script_presentation_v1",
        W,
        H,
        "Direct benchmark script orchestration - presentation",
        "\n".join(out),
    )

# =============================================================================
# benchmark_runner.run_benchmark
# =============================================================================

def build_runner_ieee():
    W, H = 1180, 2310

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

    # Dedicated rails with distinct meanings.
    loop_x = 26
    skip_x = 48
    summary_rail_x = 1130
    loop_done_x = 1160

    out.append(
        label(
            W / 2,
            38,
            "Benchmark Runner - Scenario Isolation, HC and Result Assembly",
            COL["text"],
            17,
            700,
        )
    )

    # ==================================================================
    # Entry + default config
    # ==================================================================

    start = Rg(
        "start",
        300, 60,
        640, 58,
    )

    draw_node(
        out,
        start,
        COL["blue"],
        "run_benchmark(net, profiles, network_id, config)",
        (),
        12.4,
    )

    dcfg = Dg(
        "dcfg",
        cx, 168,
        240, 58,
    )

    draw_decision(
        out,
        dcfg,
        "config is None?",
        ts=11.4,
    )

    add_route(
        out, routes,
        "start-dcfg",
        [
            start.a("bottom"),
            dcfg.a("top", TARGET_GAP),
        ],
        ignore=("start", "dcfg"),
    )

    # cy = 168, exactly matching dcfg.cy.
    # This keeps the branch truly horizontal and the arrowhead pointing right.
    defaults = Rg(
        "defaults",
        860, 137,
        270, 62,
    )

    draw_node(
        out,
        defaults,
        COL["blue"],
        "config = BenchmarkConfig()",
        ("use runner defaults",),
        10.4,
        7.8,
        9,
    )

    add_route(
        out, routes,
        "dcfg-defaults",
        [
            dcfg.a("right"),
            defaults.a("left", TARGET_GAP),
        ],
        ignore=("dcfg", "defaults"),
    )

    branch_label(
        out,
        dcfg.right + 32,
        dcfg.cy - 8,
        "yes",
        size=8.6,
    )

    cfgm = Cg(
        "cfgm",
        cx, 238,
        11,
    )

    out.append(
        circle_svg(
            cfgm,
            COL["neutral"],
        )
    )

    add_route(
        out, routes,
        "dcfg-no",
        [
            dcfg.a("bottom"),
            cfgm.a("top", TARGET_GAP),
        ],
        ignore=("dcfg", "cfgm"),
    )

    branch_label(
        out,
        dcfg.cx + 16,
        dcfg.bottom + 15,
        "no",
        anchor="start",
        size=8.5,
    )

    add_route(
        out, routes,
        "defaults-cfgm",
        [
            defaults.a("bottom"),
            (defaults.cx, cfgm.cy),
            cfgm.a("right", TARGET_GAP),
        ],
        ignore=("defaults", "cfgm"),
    )

    # ==================================================================
    # Validation + benchmark state / network inspection
    # ==================================================================

    val = Rg(
        "val",
        300, 272,
        640, 64,
    )

    draw_node(
        out,
        val,
        COL["green"],
        "_validate_inputs(net, profiles, config)",
        (
            "profile keys/times; scenario numbers; hardware port; "
            "voltage band; HC profile factory",
        ),
        10.9,
        7.9,
    )

    inspect = Rg(
        "inspect",
        300, 360,
        640, 64,
    )

    draw_node(
        out,
        inspect,
        COL["green"],
        "Initialize benchmark state + inspect network once",
        (
            "t_start; results/errors; is_lv + in-service DER presence",
            "define partial live-CSV rewrite callback used by scenario runners",
        ),
        10.5,
        7.5,
    )

    add_route(
        out, routes,
        "cfgm-val",
        [
            cfgm.a("bottom"),
            val.a("top", TARGET_GAP),
        ],
        ignore=("cfgm", "val"),
    )

    add_route(
        out, routes,
        "val-inspect",
        [
            val.a("bottom"),
            inspect.a("top", TARGET_GAP),
        ],
        ignore=("val", "inspect"),
    )

    # ==================================================================
    # Per-scenario loop
    # ==================================================================

    loop = Dg(
        "loop",
        cx, 480,
        230, 54,
    )

    draw_decision(
        out,
        loop,
        "next n in sorted(config.scenarios)?",
        ts=10.8,
        fill=COL["neutral"],
    )

    add_route(
        out, routes,
        "inspect-loop",
        [
            inspect.a("bottom"),
            loop.a("top", TARGET_GAP),
        ],
        ignore=("inspect", "loop"),
    )

    # ------------------------------------------------------------------
    # LV compatibility gate
    # ------------------------------------------------------------------

    dskip = Dg(
        "dskip",
        cx, 570,
        300, 66,
    )

    draw_decision(
        out,
        dskip,
        "LV AND !supports_lv",
        ("AND no pre-existing DERs?",),
        10.7,
        8.0,
    )

    add_route(
        out, routes,
        "loop-dskip",
        [
            loop.a("bottom"),
            dskip.a("top", TARGET_GAP),
        ],
        ignore=("loop", "dskip"),
    )

    branch_label(
        out,
        loop.cx + 16,
        loop.bottom + 15,
        "next n",
        anchor="start",
        size=8.5,
    )

    skip = Rg(
        "skip",
        65, 537,
        260, 66,
    )

    draw_node(
        out,
        skip,
        COL["neutral"],
        "Mark scenario skipped",
        (
            "errors[n]='skipped'; results[n]=None",
        ),
        9.8,
        7.4,
        9,
    )

    add_route(
        out, routes,
        "dskip-skip",
        [
            dskip.a("left"),
            skip.a("right", TARGET_GAP),
        ],
        "edge-gate",
        ignore=("dskip", "skip"),
    )

    branch_label(
        out,
        dskip.left - 30,
        dskip.cy - 8,
        "yes",
        COL["gate"],
        "end",
        8.4,
    )

    # ------------------------------------------------------------------
    # Layer 1: already-complete final JSON shortcut
    # ------------------------------------------------------------------

    dresume = Dg(
        "dresume",
        cx, 670,
        330, 68,
    )

    draw_decision(
        out,
        dresume,
        "publisher output_dir AND",
        ("scenario JSON already exists?",),
        10.5,
        7.9,
    )

    add_route(
        out, routes,
        "dskip-resume",
        [
            dskip.a("bottom"),
            dresume.a("top", TARGET_GAP),
        ],
        ignore=("dskip", "dresume"),
    )

    branch_label(
        out,
        dskip.cx + 16,
        dskip.bottom + 15,
        "no",
        anchor="start",
        size=8.4,
    )

    # cy = 670, exactly matching dresume.cy.
    loadsum = Rg(
        "loadsum",
        820, 635,
        290, 70,
    )

    draw_node(
        out,
        loadsum,
        COL["purple"],
        "Try load payload['summary']",
        (
            "success: results[n]=summary dict; skip runner",
        ),
        9.8,
        7.4,
        9,
    )

    add_route(
        out, routes,
        "dresume-loadsum",
        [
            dresume.a("right"),
            loadsum.a("left", TARGET_GAP),
        ],
        "edge-gate",
        ignore=("dresume", "loadsum"),
    )

    branch_label(
        out,
        dresume.right + 32,
        dresume.cy - 8,
        "yes",
        COL["gate"],
        "start",
        8.4,
    )

    # Directly below loadsum. Same centre x => straight exception path.
    loadfail = Rg(
        "loadfail",
        850, 722,
        230, 58,
    )

    draw_node(
        out,
        loadfail,
        COL["red"],
        "Summary load failed",
        (
            "warning; re-run this scenario from scratch",
        ),
        9.3,
        7.0,
        9,
    )

    add_route(
        out, routes,
        "loadsum-loadfail",
        [
            loadsum.a("bottom"),
            loadfail.a("top", TARGET_GAP),
        ],
        "edge-hil",
        ignore=("loadsum", "loadfail"),
    )

    branch_label(
        out,
        loadsum.cx + 16,
        loadsum.bottom + 15,
        "exception",
        COL["hil"],
        "start",
        8.1,
    )

    # ------------------------------------------------------------------
    # No existing final JSON OR failed summary load -> execute scenario.
    # loadfail.cy == runm.cy, making the recovery route horizontal.
    # ------------------------------------------------------------------

    runm = Cg(
        "runm",
        cx, 751,
        11,
    )

    out.append(
        circle_svg(
            runm,
            COL["neutral"],
        )
    )

    add_route(
        out, routes,
        "dresume-runm",
        [
            dresume.a("bottom"),
            runm.a("top", TARGET_GAP),
        ],
        ignore=("dresume", "runm"),
    )

    branch_label(
        out,
        dresume.cx + 16,
        dresume.bottom + 15,
        "no",
        anchor="start",
        size=8.4,
    )

    add_route(
        out, routes,
        "loadfail-runm",
        [
            loadfail.a("left"),
            runm.a("right", TARGET_GAP),
        ],
        "edge-hil",
        ignore=("loadfail", "runm"),
    )

    # ==================================================================
    # Scenario execution
    #
    # The source try covers deepcopy, kwargs, runner, result handling,
    # final publication and checkpoint archival.
    # ==================================================================

    run = Rg(
        "run",
        330, 790,
        580, 76,
    )

    draw_node(
        out,
        run,
        COL["blue"],
        "deepcopy(net) + _build_kwargs + scenario runner",
        (
            "optional live_csv_rewrite_fn; "
            "result = spec.runner(net_copy, profiles, **kwargs)",
        ),
        10.6,
        7.7,
    )

    add_route(
        out, routes,
        "runm-run",
        [
            runm.a("bottom"),
            run.a("top", TARGET_GAP),
        ],
        ignore=("runm", "run"),
    )

    store = Rg(
        "store",
        330, 892,
        580, 60,
    )

    draw_node(
        out,
        store,
        COL["blue"],
        "Store successful ScenarioResult",
        (
            "log violations / convergence / elapsed time; results[n] = result",
        ),
        10.2,
        7.6,
    )

    add_route(
        out, routes,
        "run-store",
        [
            run.a("bottom"),
            store.a("top", TARGET_GAP),
        ],
        ignore=("run", "store"),
    )

    dpub = Dg(
        "dpub",
        cx, 1013,
        280, 62,
    )

    draw_decision(
        out,
        dpub,
        "publish_fn has output_dir?",
        ts=10.8,
    )

    add_route(
        out, routes,
        "store-dpub",
        [
            store.a("bottom"),
            dpub.a("top", TARGET_GAP),
        ],
        ignore=("store", "dpub"),
    )

    # cy = 1013, exactly matching dpub.cy.
    pub = Rg(
        "pub",
        820, 978,
        290, 70,
    )

    draw_node(
        out,
        pub,
        COL["purple"],
        "Publish final scenario JSON",
        (
            "archive checkpoint .jsonl -> .completed if present",
        ),
        9.5,
        7.2,
        9,
    )

    add_route(
        out, routes,
        "dpub-pub",
        [
            dpub.a("right"),
            pub.a("left", TARGET_GAP),
        ],
        "edge-gate",
        ignore=("dpub", "pub"),
    )

    branch_label(
        out,
        dpub.right + 32,
        dpub.cy - 8,
        "yes",
        COL["gate"],
        "start",
        8.3,
    )

    # Representative failure branch for the whole source-level try.
    # Aligned horizontally with the main scenario-execution block.
    runerr = Rg(
        "runerr",
        55, 790,
        230, 76,
    )

    draw_node(
        out,
        runerr,
        COL["red"],
        "Scenario try failure",
        (
            "copy/kwargs/runner/result/publish/checkpoint exception",
            "store traceback; results[n]=None; continue",
        ),
        9.0,
        6.8,
        9,
    )

    add_route(
        out, routes,
        "scenario-error",
        [
            run.a("left"),
            runerr.a("right", TARGET_GAP),
        ],
        "edge-hil",
        ignore=("run", "runerr"),
    )

    branch_label(
        out,
        run.left - 28,
        run.cy - 8,
        "exception",
        COL["hil"],
        "end",
        8.0,
    )

    # ------------------------------------------------------------------
    # Normal scenario completion merge.
    #
    # dpub=no         -> top
    # publication     -> right
    # merged stream   -> bottom
    # ------------------------------------------------------------------

    pubm = Cg(
        "pubm",
        cx, 1080,
        11,
    )

    out.append(
        circle_svg(
            pubm,
            COL["neutral"],
        )
    )

    add_route(
        out, routes,
        "dpub-no-pubm",
        [
            dpub.a("bottom"),
            pubm.a("top", TARGET_GAP),
        ],
        ignore=("dpub", "pubm"),
    )

    branch_label(
        out,
        dpub.cx + 16,
        dpub.bottom + 15,
        "no",
        anchor="start",
        size=8.3,
    )

    add_route(
        out, routes,
        "pub-pubm",
        [
            pub.a("bottom"),
            (pub.cx, pubm.cy),
            pubm.a("right", TARGET_GAP),
        ],
        "edge-gate",
        ignore=("pub", "pubm"),
    )

    # ------------------------------------------------------------------
    # Skip/failure merge.
    #
    # failm aligns with runerr.cx, so scenario failure drops straight down.
    # The skipped path exits from the LEFT of its block.
    # ------------------------------------------------------------------

    failm = Cg(
        "failm",
        runerr.cx, 1188,
        11,
    )

    out.append(
        circle_svg(
            failm,
            COL["neutral"],
        )
    )

    add_route(
        out, routes,
        "skip-failm",
        [
            skip.a("left"),
            (skip_x, skip.cy),
            (skip_x, failm.cy),
            failm.a("left", TARGET_GAP),
        ],
        "edge-gate",
        ignore=("skip", "failm"),
    )

    add_route(
        out, routes,
        "runerr-failm",
        [
            runerr.a("bottom"),
            failm.a("top", TARGET_GAP),
        ],
        "edge-hil",
        ignore=("runerr", "failm"),
    )

    # ------------------------------------------------------------------
    # Final next-scenario merge.
    #
    # normal completion      -> top
    # loaded summary success -> right
    # skip / failure         -> left
    # next iteration         -> bottom
    # ------------------------------------------------------------------

    nextm = Cg(
        "nextm",
        cx, 1188,
        12,
    )

    out.append(
        circle_svg(
            nextm,
            COL["neutral"],
        )
    )

    add_route(
        out, routes,
        "pubm-nextm",
        [
            pubm.a("bottom"),
            nextm.a("top", TARGET_GAP),
        ],
        ignore=("pubm", "nextm"),
    )

    # Successful summary load executes `continue`.
    # It bypasses the scenario runner and rejoins the next-scenario merge.
    add_route(
        out, routes,
        "loadsum-nextm",
        [
            loadsum.a("right"),
            (summary_rail_x, loadsum.cy),
            (summary_rail_x, nextm.cy),
            nextm.a("right", TARGET_GAP),
        ],
        "edge-gate",
        ignore=("loadsum", "nextm"),
    )

    branch_label(
        out,
        summary_rail_x - 8,
        loadsum.cy - 10,
        "success",
        COL["gate"],
        "end",
        7.9,
    )

    add_route(
        out, routes,
        "failm-nextm",
        [
            failm.a("right"),
            nextm.a("left", TARGET_GAP),
        ],
        ignore=("failm", "nextm"),
    )

    add_route(
        out, routes,
        "nextm-loop",
        [
            nextm.a("bottom"),
            (nextm.cx, 1220),
            (loop_x, 1220),
            (loop_x, loop.cy),
            loop.a("left", TARGET_GAP),
        ],
        "edge-loop",
        ignore=("nextm", "loop"),
    )

    branch_label(
        out,
        loop_x + 10,
        1211,
        "next scenario",
        COL["loop"],
        "start",
        8.2,
    )

    # ==================================================================
    # Exit scenario loop
    # ==================================================================

    dhc = Dg(
        "dhc",
        cx, 1275,
        240, 58,
    )

    draw_decision(
        out,
        dhc,
        "config.run_hc?",
        ts=11.0,
    )

    add_route(
        out, routes,
        "loop-dhc",
        [
            loop.a("right"),
            (loop_done_x, loop.cy),
            (loop_done_x, dhc.cy),
            dhc.a("right", TARGET_GAP),
        ],
        ignore=("loop", "dhc"),
    )

    branch_label(
        out,
        loop.right + 35,
        loop.cy - 8,
        "loop done",
        anchor="start",
        size=8.4,
    )

    # ==================================================================
    # Hosting-capacity analysis
    # ==================================================================

    hc = Rg(
        "hc",
        330, 1320,
        580, 72,
    )

    draw_node(
        out,
        hc,
        COL["green"],
        "Try hosting-capacity analysis",
        (
            "run_baseline_hc -> hc_baseline + net_hc; "
            "run_hc_with_volt_var -> hc_voltvar",
        ),
        10.2,
        7.5,
    )

    add_route(
        out, routes,
        "dhc-hc",
        [
            dhc.a("bottom"),
            hc.a("top", TARGET_GAP),
        ],
        "edge-gate",
        ignore=("dhc", "hc"),
    )

    branch_label(
        out,
        dhc.cx + 15,
        dhc.bottom + 15,
        "yes",
        COL["gate"],
        "start",
        8.3,
    )

    hcerr = Rg(
        "hcerr",
        55, 1324,
        230, 64,
    )

    draw_node(
        out,
        hcerr,
        COL["red"],
        "HC failure",
        (
            "hc_error=traceback; net_hc=None; continue benchmark",
        ),
        9.1,
        6.9,
        9,
    )

    add_route(
        out, routes,
        "hc-error",
        [
            hc.a("left"),
            hcerr.a("right", TARGET_GAP),
        ],
        "edge-hil",
        ignore=("hc", "hcerr"),
    )

    branch_label(
        out,
        hc.left - 28,
        hc.cy - 8,
        "exception",
        COL["hil"],
        "end",
        8.2,
    )

    # ------------------------------------------------------------------
    # Merge paths that do NOT produce a usable stressed network:
    #
    # config.run_hc == False
    # HC analysis exception -> net_hc=None
    # ------------------------------------------------------------------

    nohc = Cg(
        "nohc",
        300, 1430,
        11,
    )

    out.append(
        circle_svg(
            nohc,
            COL["neutral"],
        )
    )

    add_route(
        out, routes,
        "dhc-no-nohc",
        [
            dhc.a("left"),
            (300, dhc.cy),
            nohc.a("top", TARGET_GAP),
        ],
        ignore=("dhc", "nohc"),
    )

    branch_label(
        out,
        dhc.left - 28,
        dhc.cy - 8,
        "no",
        anchor="end",
        size=8.3,
    )

    add_route(
        out, routes,
        "hcerr-nohc",
        [
            hcerr.a("bottom"),
            (hcerr.cx, nohc.cy),
            nohc.a("left", TARGET_GAP),
        ],
        "edge-hil",
        ignore=("hcerr", "nohc"),
    )

    # ------------------------------------------------------------------
    # Final post-HC merge:
    #
    # successful HC -> top
    # no HC / failed HC -> left
    # ------------------------------------------------------------------

    hcm = Cg(
        "hcm",
        cx, 1468,
        11,
    )

    out.append(
        circle_svg(
            hcm,
            COL["neutral"],
        )
    )

    add_route(
        out, routes,
        "hc-hcm",
        [
            hc.a("bottom"),
            hcm.a("top", TARGET_GAP),
        ],
        "edge-gate",
        ignore=("hc", "hcm"),
    )

    add_route(
        out, routes,
        "nohc-hcm",
        [
            nohc.a("bottom"),
            (nohc.cx, hcm.cy),
            hcm.a("left", TARGET_GAP),
        ],
        ignore=("nohc", "hcm"),
    )

    # ==================================================================
    # Optional HC-stressed recursive benchmark
    # ==================================================================

    dstress = Dg(
        "dstress",
        cx, 1545,
        320, 62,
    )

    draw_decision(
        out,
        dstress,
        "run_hc_scenarios AND",
        ("net_hc is available?",),
        10.7,
        8.0,
    )

    add_route(
        out, routes,
        "hcm-stress",
        [
            hcm.a("bottom"),
            dstress.a("top", TARGET_GAP),
        ],
        ignore=("hcm", "dstress"),
    )

    profileshc = Rg(
        "profileshc",
        330, 1600,
        580, 64,
    )

    draw_node(
        out,
        profileshc,
        COL["green"],
        "profiles_hc = profile_factory(net_hc)",
        (
            "optionally publish HC topology + profiles "
            "to hc_publish_fn output_dir",
        ),
        10.1,
        7.4,
    )

    add_route(
        out, routes,
        "dstress-profiles",
        [
            dstress.a("bottom"),
            profileshc.a("top", TARGET_GAP),
        ],
        "edge-gate",
        ignore=("dstress", "profileshc"),
    )

    branch_label(
        out,
        dstress.cx + 15,
        dstress.bottom + 15,
        "yes",
        COL["gate"],
        "start",
        8.3,
    )

    cfghc = Rg(
        "cfghc",
        330, 1690,
        580, 70,
    )

    draw_node(
        out,
        cfghc,
        COL["blue"],
        "Build recursive BenchmarkConfig",
        (
            "hc_stress_scenarios or outer list; "
            "run_hc=False; run_hc_scenarios=False",
        ),
        10.0,
        7.4,
    )

    recur = Rg(
        "recur",
        330, 1785,
        580, 68,
    )

    draw_node(
        out,
        recur,
        COL["blue"],
        "hc_benchmark = run_benchmark(net_hc, profiles_hc, ...)",
        (
            "recursive stressed benchmark; further HC recursion disabled",
        ),
        10.2,
        7.5,
    )

    add_route(
        out, routes,
        "profiles-cfghc",
        [
            profileshc.a("bottom"),
            cfghc.a("top", TARGET_GAP),
        ],
        ignore=("profileshc", "cfghc"),
    )

    add_route(
        out, routes,
        "cfghc-recur",
        [
            cfghc.a("bottom"),
            recur.a("top", TARGET_GAP),
        ],
        ignore=("cfghc", "recur"),
    )

    # Covers profile generation, optional publishing, config creation
    # and the recursive benchmark try in the source.
    recurerr = Rg(
        "recurerr",
        55, 1787,
        235, 64,
    )

    draw_node(
        out,
        recurerr,
        COL["red"],
        "HC-stressed setup / run failure",
        (
            "profile/publish/config/recursive-run exception",
            "log traceback; continue result assembly",
        ),
        8.8,
        6.6,
        9,
    )

    add_route(
        out, routes,
        "stress-error",
        [
            recur.a("left"),
            recurerr.a("right", TARGET_GAP),
        ],
        "edge-hil",
        ignore=("recur", "recurerr"),
    )

    branch_label(
        out,
        recur.left - 28,
        recur.cy - 8,
        "exception",
        COL["hil"],
        "end",
        8.0,
    )

    stressm = Cg(
        "stressm",
        cx, 1888,
        11,
    )

    out.append(
        circle_svg(
            stressm,
            COL["neutral"],
        )
    )

    # Optional-section bypass kept close to the stressed section.
    add_route(
        out, routes,
        "dstress-no",
        [
            dstress.a("right"),
            (970, dstress.cy),
            (970, stressm.cy),
            stressm.a("right", TARGET_GAP),
        ],
        ignore=("dstress", "stressm"),
    )

    branch_label(
        out,
        dstress.right + 28,
        dstress.cy - 8,
        "no",
        anchor="start",
        size=8.3,
    )

    add_route(
        out, routes,
        "recur-stressm",
        [
            recur.a("bottom"),
            stressm.a("top", TARGET_GAP),
        ],
        ignore=("recur", "stressm"),
    )

    add_route(
        out, routes,
        "recurerr-stressm",
        [
            recurerr.a("bottom"),
            (recurerr.cx, stressm.cy),
            stressm.a("left", TARGET_GAP),
        ],
        "edge-hil",
        ignore=("recurerr", "stressm"),
    )

    # ==================================================================
    # Comparison + optional CSV + elapsed time + return
    # ==================================================================

    comp = Rg(
        "comp",
        300, 1925,
        640, 62,
    )

    draw_node(
        out,
        comp,
        COL["purple"],
        "Build comparison_df",
        (
            "ok / skipped / failed rows; failed numeric values remain NaN",
        ),
        10.5,
        7.8,
    )

    add_route(
        out, routes,
        "stressm-comp",
        [
            stressm.a("bottom"),
            comp.a("top", TARGET_GAP),
        ],
        ignore=("stressm", "comp"),
    )

    dcsv = Dg(
        "dcsv",
        cx, 2045,
        220, 56,
    )

    draw_decision(
        out,
        dcsv,
        "config.write_csv?",
        ts=10.9,
    )

    add_route(
        out, routes,
        "comp-dcsv",
        [
            comp.a("bottom"),
            dcsv.a("top", TARGET_GAP),
        ],
        ignore=("comp", "dcsv"),
    )

    csv = Rg(
        "csv",
        820, 2014,
        300, 62,
    )

    draw_node(
        out,
        csv,
        COL["purple"],
        "Write timestamped comparison CSV",
        (
            "csv_path = _write_csv(...)",
        ),
        9.5,
        7.2,
        9,
    )

    add_route(
        out, routes,
        "dcsv-csv",
        [
            dcsv.a("right"),
            csv.a("left", TARGET_GAP),
        ],
        "edge-gate",
        ignore=("dcsv", "csv"),
    )

    branch_label(
        out,
        dcsv.right + 30,
        dcsv.cy - 8,
        "yes",
        COL["gate"],
        "start",
        8.2,
    )

    csvm = Cg(
        "csvm",
        cx, 2110,
        11,
    )

    out.append(
        circle_svg(
            csvm,
            COL["neutral"],
        )
    )

    add_route(
        out, routes,
        "dcsv-no",
        [
            dcsv.a("bottom"),
            csvm.a("top", TARGET_GAP),
        ],
        ignore=("dcsv", "csvm"),
    )

    branch_label(
        out,
        dcsv.cx + 14,
        dcsv.bottom + 14,
        "no",
        anchor="start",
        size=8.2,
    )

    add_route(
        out, routes,
        "csv-csvm",
        [
            csv.a("bottom"),
            (csv.cx, csvm.cy),
            csvm.a("right", TARGET_GAP),
        ],
        "edge-gate",
        ignore=("csv", "csvm"),
    )

    summary = Rg(
        "summary",
        300, 2140,
        640, 58,
    )

    draw_node(
        out,
        summary,
        COL["purple"],
        "Compute total elapsed time + print benchmark summary",
        (
            "Rich console when available; plain-text fallback otherwise",
        ),
        10.2,
        7.5,
    )

    ret = Rg(
        "ret",
        300, 2220,
        640, 60,
    )

    draw_node(
        out,
        ret,
        COL["purple"],
        "return BenchmarkResult",
        (
            "results, errors, comparison_df, elapsed_s, csv_path",
            "hc_results, hc_error, hc_benchmark, net_hc",
        ),
        10.5,
        7.5,
    )

    add_route(
        out, routes,
        "csvm-summary",
        [
            csvm.a("bottom"),
            summary.a("top", TARGET_GAP),
        ],
        ignore=("csvm", "summary"),
    )

    add_route(
        out, routes,
        "summary-ret",
        [
            summary.a("bottom"),
            ret.a("top", TARGET_GAP),
        ],
        ignore=("summary", "ret"),
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

    # Distinct left-side rails.
    assert loop_x < skip_x < skip.left

    # Stored-summary continuation stays outside all right-side scenario nodes.
    assert summary_rail_x > max(
        loadsum.right,
        loadfail.right,
        pub.right,
    ) + 10

    # Summary shortcut and loop-done rails have different meanings.
    assert summary_rail_x < loop_done_x

    # Long rails stay inside the panel.
    assert loop_done_x < W - 14

    # Exact centre alignment prevents the tiny vertical final segments
    # that previously rotated horizontal branch arrowheads downward.
    assert abs(defaults.cy - dcfg.cy) < 1e-9
    assert abs(loadsum.cy - dresume.cy) < 1e-9
    assert abs(pub.cy - dpub.cy) < 1e-9

    # Scenario failure drops directly onto its merge.
    assert abs(failm.cx - runerr.cx) < 1e-9

    # Keep the canvas tight at the bottom.
    assert H - ret.bottom <= 30

    write(
        "flow_orch_runner_ieee_v1",
        W,
        H,
        "Benchmark runner orchestration - IEEE",
        "\n".join(out),
    )

def build_runner_presentation():
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

    cx = 525

    # Dedicated rails.
    loop_x = 8
    skip_x = 22
    summary_x = 1025
    loop_done_x = 1055

    # ==================================================================
    # One-time benchmark setup
    # ==================================================================

    init = Rg(
        "init",
        170, 16,
        710, 48,
    )

    draw_node(
        out,
        init,
        COL["blue"],
        "run_benchmark: default config if needed -> validate inputs -> inspect LV/DER state",
        (
            "initialize timer/results/errors and live-CSV callback "
            "before scenario isolation",
        ),
        10.6,
        7.2,
    )

    loop = Dg(
        "loop",
        cx, 112,
        240, 40,
    )

    draw_decision(
        out,
        loop,
        "next configured scenario?",
        ts=10.2,
        fill=COL["neutral"],
    )

    add_route(
        out, routes,
        "init-loop",
        [
            init.a("bottom"),
            loop.a("top", TARGET_GAP),
        ],
        ignore=("init", "loop"),
    )

    # ==================================================================
    # Per-scenario isolation / skip / resume
    # ==================================================================

    dskip = Dg(
        "dskip",
        cx, 170,
        250, 42,
    )

    draw_decision(
        out,
        dskip,
        "unsupported LV with no DER?",
        ts=9.6,
    )

    add_route(
        out, routes,
        "loop-skipd",
        [
            loop.a("bottom"),
            dskip.a("top", TARGET_GAP),
        ],
        ignore=("loop", "dskip"),
    )

    # cy = 170 exactly matches dskip.cy.
    # The branch is therefore genuinely horizontal.
    skip = Rg(
        "skip",
        35, 147,
        235, 46,
    )

    draw_node(
        out,
        skip,
        COL["neutral"],
        "Skip scenario",
        (
            "errors[n]='skipped'; results[n]=None",
        ),
        8.4,
        6.2,
        8,
    )

    add_route(
        out, routes,
        "skipd-skip",
        [
            dskip.a("left"),
            skip.a("right", TARGET_GAP),
        ],
        "edge-gate",
        ignore=("dskip", "skip"),
    )

    branch_label(
        out,
        dskip.left - 25,
        dskip.cy - 6,
        "yes",
        COL["gate"],
        size=7.7,
    )

    dresume = Dg(
        "dresume",
        cx, 232,
        265, 44,
    )

    draw_decision(
        out,
        dresume,
        "completed scenario JSON exists?",
        ts=9.5,
    )

    add_route(
        out, routes,
        "skipd-resume",
        [
            dskip.a("bottom"),
            dresume.a("top", TARGET_GAP),
        ],
        ignore=("dskip", "dresume"),
    )

    branch_label(
        out,
        dskip.cx + 11,
        dskip.bottom + 12,
        "no",
        anchor="start",
        size=7.7,
    )

    # cy = 232 exactly matches dresume.cy.
    saved = Rg(
        "saved",
        755, 209,
        250, 46,
    )

    draw_node(
        out,
        saved,
        COL["purple"],
        "Load stored summary",
        (
            "success: results[n]=summary dict; continue",
        ),
        8.4,
        6.2,
        8,
    )

    add_route(
        out, routes,
        "res-saved",
        [
            dresume.a("right"),
            saved.a("left", TARGET_GAP),
        ],
        "edge-gate",
        ignore=("dresume", "saved"),
    )

    branch_label(
        out,
        dresume.right + 26,
        dresume.cy - 6,
        "yes",
        COL["gate"],
        size=7.7,
    )

    # --------------------------------------------------------------
    # Existing JSON was found, but the stored summary cannot be read.
    #
    # This is NOT terminal. The runner warns and executes this scenario
    # normally from scratch.
    #
    # Same centre-x as `saved`, so the exception path drops vertically.
    # --------------------------------------------------------------

    loadfail = Rg(
        "loadfail",
        770, 269,
        220, 50,
    )

    draw_node(
        out,
        loadfail,
        COL["red"],
        "Stored summary unreadable",
        (
            "warn; re-run this scenario normally",
        ),
        8.1,
        6.0,
        8,
    )

    add_route(
        out, routes,
        "saved-loadfail",
        [
            saved.a("bottom"),
            loadfail.a("top", TARGET_GAP),
        ],
        "edge-hil",
        ignore=("saved", "loadfail"),
    )

    branch_label(
        out,
        saved.cx + 10,
        saved.bottom + 11,
        "exception",
        COL["hil"],
        "start",
        7.1,
    )

    # --------------------------------------------------------------
    # Merge:
    #
    # completed JSON does not exist
    # OR
    # completed JSON existed but could not be read
    #
    # -> execute scenario normally.
    # --------------------------------------------------------------

    runm = Cg(
        "runm",
        cx, 294,
        8,
    )

    out.append(
        circle_svg(
            runm,
            COL["neutral"],
        )
    )

    add_route(
        out, routes,
        "resume-no-runm",
        [
            dresume.a("bottom"),
            runm.a("top", TARGET_GAP),
        ],
        ignore=("dresume", "runm"),
    )

    branch_label(
        out,
        dresume.cx + 11,
        dresume.bottom + 12,
        "no",
        anchor="start",
        size=7.5,
    )

    # loadfail.cy == runm.cy, so this recovery route is horizontal.
    add_route(
        out, routes,
        "loadfail-runm",
        [
            loadfail.a("left"),
            runm.a("right", TARGET_GAP),
        ],
        "edge-hil",
        ignore=("loadfail", "runm"),
    )

    # ==================================================================
    # Scenario execution and final scenario publication
    # ==================================================================

    run = Rg(
        "run",
        300, 320,
        450, 58,
    )

    draw_node(
        out,
        run,
        COL["blue"],
        "deepcopy net -> build kwargs -> scenario runner",
        (
            "live CSV callback where configured; "
            "final result handling follows on success",
        ),
        9.3,
        6.6,
    )

    add_route(
        out, routes,
        "runm-run",
        [
            runm.a("bottom"),
            run.a("top", TARGET_GAP),
        ],
        ignore=("runm", "run"),
    )

    # --------------------------------------------------------------
    # Common source-level scenario try failure.
    #
    # The same try covers:
    # - deepcopy
    # - _build_kwargs
    # - scenario runner
    # - successful result handling
    # - final scenario publication
    # - checkpoint archival
    #
    # Presentation keeps one compact failure node instead of a dashed
    # try-region outline.
    # --------------------------------------------------------------

    err = Rg(
        "err",
        35, 320,
        235, 58,
    )

    draw_node(
        out,
        err,
        COL["red"],
        "Scenario try failure",
        (
            "copy / kwargs / runner / final-publish exception",
            "traceback + None result; continue",
        ),
        8.1,
        5.9,
        8,
    )

    add_route(
        out, routes,
        "run-err",
        [
            run.a("left"),
            err.a("right", TARGET_GAP),
        ],
        "edge-hil",
        ignore=("run", "err"),
    )

    branch_label(
        out,
        run.left - 24,
        run.cy - 6,
        "exception",
        COL["hil"],
        size=7.2,
    )

    dpub = Dg(
        "dpub",
        cx, 410,
        220, 40,
    )

    draw_decision(
        out,
        dpub,
        "publisher output_dir?",
        ts=9.5,
    )

    add_route(
        out, routes,
        "run-dpub",
        [
            run.a("bottom"),
            dpub.a("top", TARGET_GAP),
        ],
        ignore=("run", "dpub"),
    )

    # cy = 410 exactly matches dpub.cy.
    pub = Rg(
        "pub",
        760, 388,
        245, 44,
    )

    draw_node(
        out,
        pub,
        COL["purple"],
        "Publish final scenario",
        (
            "archive checkpoint if present",
        ),
        8.3,
        6.1,
        8,
    )

    add_route(
        out, routes,
        "dpub-pub",
        [
            dpub.a("right"),
            pub.a("left", TARGET_GAP),
        ],
        "edge-gate",
        ignore=("dpub", "pub"),
    )

    branch_label(
        out,
        dpub.right + 25,
        dpub.cy - 6,
        "yes",
        COL["gate"],
        size=7.6,
    )

    # --------------------------------------------------------------
    # Normal completion merge:
    #
    # no publishing needed -> top
    # publishing completed -> right
    # merged flow          -> bottom
    # --------------------------------------------------------------

    pubm = Cg(
        "pubm",
        cx, 465,
        8,
    )

    out.append(
        circle_svg(
            pubm,
            COL["neutral"],
        )
    )

    add_route(
        out, routes,
        "dpub-no",
        [
            dpub.a("bottom"),
            pubm.a("top", TARGET_GAP),
        ],
        ignore=("dpub", "pubm"),
    )

    branch_label(
        out,
        dpub.cx + 10,
        dpub.bottom + 11,
        "no",
        anchor="start",
        size=7.5,
    )

    add_route(
        out, routes,
        "pub-pubm",
        [
            pub.a("bottom"),
            (pub.cx, pubm.cy),
            pubm.a("right", TARGET_GAP),
        ],
        "edge-gate",
        ignore=("pub", "pubm"),
    )

    # ==================================================================
    # Staged next-scenario merge
    # ==================================================================

    # --------------------------------------------------------------
    # First merge the two left-side continue paths:
    #
    # unsupported/skipped scenario
    # failed scenario try
    # --------------------------------------------------------------

    failm = Cg(
        "failm",
        err.cx, 525,
        8,
    )

    out.append(
        circle_svg(
            failm,
            COL["neutral"],
        )
    )

    # Skip leaves directly from the LEFT side of the block.
    add_route(
        out, routes,
        "skip-failm",
        [
            skip.a("left"),
            (skip_x, skip.cy),
            (skip_x, failm.cy),
            failm.a("left", TARGET_GAP),
        ],
        "edge-gate",
        ignore=("skip", "failm"),
    )

    # Scenario failure drops vertically onto the merge.
    add_route(
        out, routes,
        "err-failm",
        [
            err.a("bottom"),
            failm.a("top", TARGET_GAP),
        ],
        "edge-hil",
        ignore=("err", "failm"),
    )

    # --------------------------------------------------------------
    # Final next-scenario merge:
    #
    # normal completion       -> top
    # stored-summary success  -> right
    # skip / failed scenario  -> left
    # next iteration          -> bottom
    # --------------------------------------------------------------

    nextm = Cg(
        "nextm",
        cx, 525,
        9,
    )

    out.append(
        circle_svg(
            nextm,
            COL["neutral"],
        )
    )

    add_route(
        out, routes,
        "pubm-nextm",
        [
            pubm.a("bottom"),
            nextm.a("top", TARGET_GAP),
        ],
        ignore=("pubm", "nextm"),
    )

    add_route(
        out, routes,
        "failm-nextm",
        [
            failm.a("right"),
            nextm.a("left", TARGET_GAP),
        ],
        ignore=("failm", "nextm"),
    )

    # Successful stored-summary read executes `continue`.
    add_route(
        out, routes,
        "saved-nextm",
        [
            saved.a("right"),
            (summary_x, saved.cy),
            (summary_x, nextm.cy),
            nextm.a("right", TARGET_GAP),
        ],
        "edge-gate",
        ignore=("saved", "nextm"),
    )

    branch_label(
        out,
        summary_x - 7,
        saved.cy - 7,
        "success",
        COL["gate"],
        "end",
        7.0,
    )

    # Loopback leaves from the bottom, not from an already-used side.
    add_route(
        out, routes,
        "nextm-loop",
        [
            nextm.a("bottom"),
            (nextm.cx, 548),
            (loop_x, 548),
            (loop_x, loop.cy),
            loop.a("left", TARGET_GAP),
        ],
        "edge-loop",
        ignore=("nextm", "loop"),
    )

    branch_label(
        out,
        loop_x + 9,
        540,
        "next scenario",
        COL["loop"],
        "start",
        7.1,
    )

    # ==================================================================
    # Exit scenario loop + hosting capacity
    # ==================================================================

    dhc = Dg(
        "dhc",
        cx, 585,
        190, 36,
    )

    draw_decision(
        out,
        dhc,
        "run_hc?",
        ts=9.3,
    )

    # Only loop completion uses this outer-right rail.
    add_route(
        out, routes,
        "loop-hc",
        [
            loop.a("right"),
            (loop_done_x, loop.cy),
            (loop_done_x, dhc.cy),
            dhc.a("right", TARGET_GAP),
        ],
        ignore=("loop", "dhc"),
    )

    branch_label(
        out,
        loop.right + 28,
        loop.cy - 6,
        "loop done",
        size=7.6,
    )

    hc = Rg(
        "hc",
        300, 620,
        450, 48,
    )

    draw_node(
        out,
        hc,
        COL["green"],
        "Baseline HC + Volt-Var HC",
        (
            "successful analysis provides HC results and net_hc",
        ),
        8.9,
        6.4,
    )

    add_route(
        out, routes,
        "dhc-hc",
        [
            dhc.a("bottom"),
            hc.a("top", TARGET_GAP),
        ],
        "edge-gate",
        ignore=("dhc", "hc"),
    )

    branch_label(
        out,
        dhc.cx + 10,
        dhc.bottom + 11,
        "yes",
        COL["gate"],
        "start",
        7.5,
    )

    # --------------------------------------------------------------
    # Explicit non-fatal HC exception branch.
    # --------------------------------------------------------------

    hcerr = Rg(
        "hcerr",
        40, 620,
        220, 48,
    )

    draw_node(
        out,
        hcerr,
        COL["red"],
        "HC failure",
        (
            "hc_error set; net_hc=None; continue",
        ),
        8.1,
        6.0,
        8,
    )

    add_route(
        out, routes,
        "hc-error",
        [
            hc.a("left"),
            hcerr.a("right", TARGET_GAP),
        ],
        "edge-hil",
        ignore=("hc", "hcerr"),
    )

    branch_label(
        out,
        hc.left - 24,
        hc.cy - 6,
        "exception",
        COL["hil"],
        size=7.1,
    )

    # --------------------------------------------------------------
    # run_hc=False and HC failure both mean there is no usable net_hc.
    # Merge those cases first.
    # --------------------------------------------------------------

    nohc = Cg(
        "nohc",
        150, 700,
        8,
    )

    out.append(
        circle_svg(
            nohc,
            COL["neutral"],
        )
    )

    # Use an outer-left rail that clears the HC-failure block.
    add_route(
        out, routes,
        "hc-no-nohc",
        [
            dhc.a("left"),
            (30, dhc.cy),
            (30, nohc.cy),
            nohc.a("left", TARGET_GAP),
        ],
        ignore=("dhc", "nohc"),
    )

    branch_label(
        out,
        dhc.left - 24,
        dhc.cy - 6,
        "no",
        size=7.5,
    )

    # hcerr.cx == nohc.cx, so this is a straight vertical drop.
    add_route(
        out, routes,
        "hcerr-nohc",
        [
            hcerr.a("bottom"),
            nohc.a("top", TARGET_GAP),
        ],
        "edge-hil",
        ignore=("hcerr", "nohc"),
    )

    # --------------------------------------------------------------
    # Successful HC and no-HC-result streams converge before deciding
    # whether an HC-stressed recursive benchmark can run.
    # --------------------------------------------------------------

    hm = Cg(
        "hm",
        cx, 700,
        8,
    )

    out.append(
        circle_svg(
            hm,
            COL["neutral"],
        )
    )

    add_route(
        out, routes,
        "hc-hm",
        [
            hc.a("bottom"),
            hm.a("top", TARGET_GAP),
        ],
        "edge-gate",
        ignore=("hc", "hm"),
    )

    add_route(
        out, routes,
        "nohc-hm",
        [
            nohc.a("right"),
            hm.a("left", TARGET_GAP),
        ],
        ignore=("nohc", "hm"),
    )

    # ==================================================================
    # Optional HC-stressed recursive benchmark
    # ==================================================================

    dstress = Dg(
        "dstress",
        cx, 752,
        250, 40,
    )

    draw_decision(
        out,
        dstress,
        "HC re-benchmark enabled + net_hc?",
        ts=9.2,
    )

    add_route(
        out, routes,
        "hm-stress",
        [
            hm.a("bottom"),
            dstress.a("top", TARGET_GAP),
        ],
        ignore=("hm", "dstress"),
    )

    recur = Rg(
        "recur",
        300, 785,
        450, 60,
    )

    draw_node(
        out,
        recur,
        COL["blue"],
        "Build HC profiles/config -> recursive run_benchmark",
        (
            "optional HC topology/profile publish; "
            "run_hc=False and run_hc_scenarios=False",
        ),
        8.6,
        6.1,
    )

    add_route(
        out, routes,
        "stress-recur",
        [
            dstress.a("bottom"),
            recur.a("top", TARGET_GAP),
        ],
        "edge-gate",
        ignore=("dstress", "recur"),
    )

    branch_label(
        out,
        dstress.cx + 10,
        dstress.bottom + 11,
        "yes",
        COL["gate"],
        "start",
        7.4,
    )

    # --------------------------------------------------------------
    # Explicit HC-stressed try failure.
    #
    # This represents failures from profile creation, optional publishing,
    # recursive config creation, or recursive run_benchmark.
    # --------------------------------------------------------------

    recurerr = Rg(
        "recurerr",
        35, 790,
        235, 50,
    )

    draw_node(
        out,
        recurerr,
        COL["red"],
        "HC-stressed run failed",
        (
            "profile / publish / config / recursive-run error",
            "log error; continue result assembly",
        ),
        7.8,
        5.8,
        8,
    )

    add_route(
        out, routes,
        "recur-error",
        [
            recur.a("left"),
            recurerr.a("right", TARGET_GAP),
        ],
        "edge-hil",
        ignore=("recur", "recurerr"),
    )

    branch_label(
        out,
        recur.left - 24,
        recur.cy - 6,
        "exception",
        COL["hil"],
        size=7.1,
    )

    # --------------------------------------------------------------
    # HC-stressed branch merge:
    #
    # successful recursive benchmark -> top
    # failed recursive setup/run     -> left
    # optional section bypass        -> right
    # --------------------------------------------------------------

    sm = Cg(
        "sm",
        cx, 875,
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
        "recur-sm",
        [
            recur.a("bottom"),
            sm.a("top", TARGET_GAP),
        ],
        ignore=("recur", "sm"),
    )

    add_route(
        out, routes,
        "recurerr-sm",
        [
            recurerr.a("bottom"),
            (recurerr.cx, sm.cy),
            sm.a("left", TARGET_GAP),
        ],
        "edge-hil",
        ignore=("recurerr", "sm"),
    )

    add_route(
        out, routes,
        "stress-no",
        [
            dstress.a("right"),
            (930, dstress.cy),
            (930, sm.cy),
            sm.a("right", TARGET_GAP),
        ],
        ignore=("dstress", "sm"),
    )

    branch_label(
        out,
        dstress.right + 24,
        dstress.cy - 6,
        "no",
        size=7.4,
    )

    # ==================================================================
    # Result assembly
    # ==================================================================

    comp = Rg(
        "comp",
        240, 902,
        570, 48,
    )

    draw_node(
        out,
        comp,
        COL["purple"],
        "Build comparison_df -> optional CSV -> print summary",
        (
            "compute total elapsed time; "
            "ok / skipped / failed rows remain distinguishable",
        ),
        9.1,
        6.4,
    )

    ret = Rg(
        "ret",
        240, 970,
        570, 54,
    )

    draw_node(
        out,
        ret,
        COL["purple"],
        "Return BenchmarkResult",
        (
            "results/errors + comparison + HC and HC-stressed outputs",
        ),
        9.2,
        6.5,
    )

    add_route(
        out, routes,
        "sm-comp",
        [
            sm.a("bottom"),
            comp.a("top", TARGET_GAP),
        ],
        ignore=("sm", "comp"),
    )

    add_route(
        out, routes,
        "comp-ret",
        [
            comp.a("bottom"),
            ret.a("top", TARGET_GAP),
        ],
        ignore=("comp", "ret"),
    )

    # ==================================================================
    # Audience-facing explanatory panels
    # ==================================================================

    px, pw = 1080, 800

    panels = [
        R(px, 24,  pw, 280),
        R(px, 326, pw, 338),
        R(px, 686, pw, 352),
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

    # ------------------------------------------------------------------
    # Panel 1
    # ------------------------------------------------------------------

    out.append(
        label(
            px + 24,
            57,
            "Scenario isolation and dispatch",
            COL["text"],
            17,
            700,
            "start",
        )
    )

    lines = [
        "Validation runs once before the loop and checks profiles, scenario IDs, hardware settings and HC prerequisites.",
        "The runner also records whether the network is LV and whether an in-service PV/wind DER fleet already exists.",
        "Configured scenarios run in ascending numeric order; each receives its own copy.deepcopy(net).",
        "_build_kwargs injects only the hardware, coordination, OPF, publishing and checkpoint options supported by that runner.",
        "An exception during isolated execution or final publication is stored as a failed result; the next scenario still runs.",
    ]

    for i, t in enumerate(lines):
        out.append(
            label(
                px + 24,
                92 + 39 * i,
                t,
                COL["text"],
                10.6,
                600,
                "start",
            )
        )

    # ------------------------------------------------------------------
    # Panel 2
    # ------------------------------------------------------------------

    out.append(
        label(
            px + 24,
            359,
            "Skip, resume and publication layers",
            COL["text"],
            17,
            700,
            "start",
        )
    )

    lines = [
        "Unsupported LV scenarios are skipped only when the LV network also has no controllable PV/wind DERs.",
        "An existing final scenario JSON can be loaded as a completed summary and that scenario is not rerun.",
        "If the stored JSON cannot be read, the runner logs a warning and reruns that scenario from scratch.",
        "When live_csv_path is configured, a callback rewrites a partial benchmark CSV during scenario execution.",
        "After success, the final scenario JSON is published and any matching checkpoint stream is archived as .completed.",
        "Skip, stored-summary success, normal success and failure all rejoin at the next-scenario loop boundary.",
    ]

    for i, t in enumerate(lines):
        out.append(
            label(
                px + 24,
                394 + 39 * i,
                t,
                COL["text"],
                10.6,
                600,
                "start",
            )
        )

    # ------------------------------------------------------------------
    # Panel 3
    # ------------------------------------------------------------------

    out.append(
        label(
            px + 24,
            719,
            "Hosting capacity and result assembly",
            COL["text"],
            17,
            700,
            "start",
        )
    )

    lines = [
        "After the scenario loop, optional HC evaluates baseline and Volt-Var hosting capacity on the untouched original network.",
        "HC failure is stored in hc_error, sets net_hc=None and does not abort the outer benchmark.",
        "If HC-stressed scenarios are enabled and net_hc exists, profile_factory builds profiles for the stressed network.",
        "The recursive config disables run_hc and run_hc_scenarios so a second HC study cannot recurse indefinitely.",
        "HC-stressed setup/run failure is logged; comparison assembly still proceeds.",
        "Finally, the runner builds comparison_df, optionally writes CSV, prints the summary and returns BenchmarkResult.",
    ]

    for i, t in enumerate(lines):
        out.append(
            label(
                px + 24,
                754 + 43 * i,
                t,
                COL["text"],
                10.5,
                600,
                "start",
            )
        )

    # ==================================================================
    # Geometry checks
    # ==================================================================

    audit_bounds(
        nodes,
        W,
        H,
        0,
    )

    audit_node_overlaps(
        nodes
    )

    audit_routes(
        routes,
        nodes,
    )

    # Horizontal decision branches are deliberately centre-aligned.
    assert abs(skip.cy - dskip.cy) < 1e-9
    assert abs(saved.cy - dresume.cy) < 1e-9
    assert abs(pub.cy - dpub.cy) < 1e-9

    # Summary load exception drops vertically.
    assert abs(loadfail.cx - saved.cx) < 1e-9

    # Scenario failure drops vertically onto the left merge.
    assert abs(failm.cx - err.cx) < 1e-9

    # Stored-summary success and true loop completion use separate rails.
    assert summary_x < loop_done_x

    # Left-side loop / skip rails remain physically distinct.
    assert loop_x < skip_x < skip.left

    # Content remains inside the 1080p presentation canvas.
    assert ret.bottom < H

    write(
        "flow_orch_runner_presentation_v1",
        W,
        H,
        "Benchmark runner orchestration - presentation",
        "\n".join(out),
    )

def main():
    build_script_ieee()
    build_script_presentation()
    build_runner_ieee()
    build_runner_presentation()
    print(f"Wrote orchestration flowcharts to {OUT}")


if __name__ == "__main__":
    main()
