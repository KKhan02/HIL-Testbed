from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import html
import xml.etree.ElementTree as ET
import cairosvg

OUT = Path(__file__).resolve().parent / "plugin_flowcharts_v4"
OUT.mkdir(parents=True, exist_ok=True)

COL = {
    "blue": "#0C447C", "green": "#0F6E56", "purple": "#3C3489",
    "red": "#993C1D", "dry": "#3B6D11", "decision": "#854F0B",
    "neutral": "#5F5E5A", "white": "#FFFFFF", "text": "#111111",
    "hil": "#D85A30", "gate": "#7A5C12", "loop": "#3D8FD9",
    "assoc": "#8A887F", "panel": "#F8FAFC", "panel_border": "#D9E1E8",
    "detail_border": "#8A887F",
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


@dataclass(frozen=True)
class C:
    cx: float; cy: float; r: float
    @property
    def left(self): return self.cx-self.r
    @property
    def right(self): return self.cx+self.r
    @property
    def top(self): return self.cy-self.r
    @property
    def bottom(self): return self.cy+self.r
    def a(self, side, gap=0.0):
        if side == "top": return (self.cx, self.top-gap)
        if side == "bottom": return (self.cx, self.bottom+gap)
        if side == "left": return (self.left-gap, self.cy)
        if side == "right": return (self.right+gap, self.cy)
        raise ValueError(side)


def rect_svg(g, fill, rx=12, stroke="none", sw=0):
    return f'<rect x="{g.x}" y="{g.y}" width="{g.w}" height="{g.h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>'

def diamond_svg(g, fill):
    return f'<polygon points="{g.cx},{g.top} {g.right},{g.cy} {g.cx},{g.bottom} {g.left},{g.cy}" fill="{fill}"/>'

def circle_svg(g, fill): return f'<circle cx="{g.cx}" cy="{g.cy}" r="{g.r}" fill="{fill}"/>'

def label(x,y,text,fill,size=13,weight=700,anchor="middle"):
    return f'<text x="{x}" y="{y}" text-anchor="{anchor}" fill="{fill}" font-size="{size}" font-weight="{weight}">{esc(text)}</text>'


def text_lines_centered(x, cy, title, subs=(), title_size=14, sub_size=10, line_gap=16, fill="#FFFFFF", anchor="middle"):
    lines=[(title,title_size,700)] + [(s,sub_size,400) for s in subs]
    total=(len(lines)-1)*line_gap
    first=cy-total/2+title_size*0.34
    out=[]
    for i,(txt,size,weight) in enumerate(lines):
        cls="node-title" if weight>=700 else "node-sub"
        opacity="" if weight>=700 else ' opacity="0.92"'
        out.append(f'<text x="{x}" y="{first+i*line_gap}" text-anchor="{anchor}" class="{cls}" font-size="{size}" fill="{fill}"{opacity}>{esc(txt)}</text>')
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
    pts=_clean_points(points)
    markers={"edge-dark":"arrowDark","edge-hil":"arrowHil","edge-gate":"arrowGate","edge-loop":"arrowLoop","edge-assoc":"arrowAssoc","edge-dry":"arrowDry"}
    d="M"+" L".join(f"{x:g} {y:g}" for x,y in pts)
    m=f' marker-end="url(#{markers[cls]})"' if marker else ""
    return f'<path d="{d}" class="{cls}"{m}/>'


def direct(src,ss,dst,ds,cls="edge-dark",gap=TARGET_GAP): return path([src.a(ss),dst.a(ds,gap)],cls)

def ortho_hv(src,ss,dst,ds,cls="edge-dark",bend_x=None,gap=TARGET_GAP):
    s=src.a(ss); t=dst.a(ds,gap); bx=(s[0]+t[0])/2 if bend_x is None else bend_x
    return path([s,(bx,s[1]),(bx,t[1]),t],cls)

def ortho_vh(src,ss,dst,ds,cls="edge-dark",bend_y=None,gap=TARGET_GAP):
    s=src.a(ss); t=dst.a(ds,gap); by=(s[1]+t[1])/2 if bend_y is None else bend_y
    return path([s,(s[0],by),(t[0],by),t],cls)


def header(w,h,title):
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">
<title>{esc(title)}</title>
<defs>
<marker id="arrowDark" markerWidth="6" markerHeight="6" refX="8" refY="4.5" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L9,4.5 L0,9 z" fill="#263238"/></marker>
<marker id="arrowHil" markerWidth="6" markerHeight="6" refX="8" refY="4.5" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L9,4.5 L0,9 z" fill="#D85A30"/></marker>
<marker id="arrowGate" markerWidth="6" markerHeight="6" refX="8" refY="4.5" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L9,4.5 L0,9 z" fill="#7A5C12"/></marker>
<marker id="arrowLoop" markerWidth="6" markerHeight="6" refX="8" refY="4.5" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L9,4.5 L0,9 z" fill="#3D8FD9"/></marker>
<marker id="arrowAssoc" markerWidth="6" markerHeight="6" refX="8" refY="4.5" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L9,4.5 L0,9 z" fill="#8A887F"/></marker>
<marker id="arrowDry" markerWidth="6" markerHeight="6" refX="8" refY="4.5" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L9,4.5 L0,9 z" fill="#6B9E26"/></marker>
</defs><style>
text {{font-family:Helvetica,Arial,sans-serif}} .node-title{{font-weight:700}} .node-sub{{font-weight:400}}
.edge-dark{{fill:none;stroke:#263238;stroke-width:2.6;stroke-linejoin:round;stroke-linecap:round}}
.edge-hil{{fill:none;stroke:#D85A30;stroke-width:2.8;stroke-linejoin:round;stroke-linecap:round}}
.edge-gate{{fill:none;stroke:#7A5C12;stroke-width:2.6;stroke-linejoin:round;stroke-linecap:round}}
.edge-loop{{fill:none;stroke:#3D8FD9;stroke-width:2.6;stroke-dasharray:7 5;stroke-linejoin:round;stroke-linecap:round}}
.edge-assoc{{fill:none;stroke:#8A887F;stroke-width:2;stroke-dasharray:5 4;stroke-linejoin:round;stroke-linecap:round}}
.edge-dry{{fill:none;stroke:#6B9E26;stroke-width:2.6;stroke-linejoin:round;stroke-linecap:round}}
</style>'''


def write(name,w,h,title,body):
    svg=header(w,h,title)+"\n"+body+"\n</svg>"
    sp=OUT/f"{name}.svg"; pp=OUT/f"{name}.pdf"; pn=OUT/f"{name}.png"
    sp.write_text(svg,encoding="utf-8")
    cairosvg.svg2pdf(bytestring=svg.encode(),write_to=str(pp))
    cairosvg.svg2png(bytestring=svg.encode(),write_to=str(pn),output_width=w*2,output_height=h*2)
    ET.parse(sp)
    assert min(sp.stat().st_size,pp.stat().st_size,pn.stat().st_size)>1000


def draw_node(out,g,fill,title,subs=(),ts=14,ss=10,rx=12):
    out += [rect_svg(g,fill,rx), text_lines_centered(g.cx,g.cy,title,subs,ts,ss,16)]

def draw_decision(out,g,title,subs=(),ts=13,ss=9.5,fill=None):
    out += [diamond_svg(g,fill or COL["decision"]), text_lines_centered(g.cx,g.cy,title,subs,ts,ss,14)]

def audit_rect_bounds(n,g,W,H,m=0): assert g.left>=m and g.top>=m and g.right<=W-m and g.bottom<=H-m, f"{n}: bounds"

def audit_diamond_bounds(n,g,W,H,m=0): assert g.left>=m and g.right<=W-m and g.top>=m and g.bottom<=H-m, f"{n}: bounds"


# ---------------------------------------------------------------------------
# Registration, routing, cleanup
# ---------------------------------------------------------------------------


def branch_label(out, x, y, text, fill=COL["text"], anchor="middle", size=9.5):
    out.append(label(x, y, text, fill, size, 700, anchor))


def build_registration_ieee():
    W, H = 980, 1865

    out = [
        rect_svg(
            R(14, 14, W - 28, H - 28),
            COL["panel"],
            16,
            COL["panel_border"],
            1.5,
        )
    ]

    rects = {}
    diamonds = {}

    def Rg(n, x, y, w, h):
        g = R(x, y, w, h)
        rects[n] = g
        return g

    def Dg(n, x, y, w, h):
        g = D(x, y, w, h)
        diamonds[n] = g
        return g

    cx = 500

    out.append(
        label(
            W / 2,
            38,
            "Custom Controller Plugin - Registration and Execution Routing",
            COL["text"],
            17,
            700,
        )
    )

    # ==================================================================
    # Plugin loading and Python function binding
    # ==================================================================

    start = Rg("start", 225, 62, 550, 58)
    draw_node(
        out,
        start,
        COL["blue"],
        "register_and_run(...)",
        ("YAML, net, profiles, network_id, config, port",),
        13.5,
        9.3,
    )

    load = Rg("load", 210, 145, 580, 78)
    draw_node(
        out,
        load,
        COL["green"],
        "load_plugin(yaml) -> cfg",
        (
            "validate required fields, plugin name, paths, kwargs, gate/clamp flags",
            "hardware:true additionally validates an existing .ino path",
        ),
        12.2,
        8.7,
    )

    imp = Rg("imp", 210, 248, 580, 62)
    draw_node(
        out,
        imp,
        COL["green"],
        "_import_controller_fn(module_path, function)",
        (
            "unique sys.modules name; execute module; require named callable",
        ),
        12.0,
        8.6,
    )

    bind = Rg("bind", 210, 335, 580, 58)
    draw_node(
        out,
        bind,
        COL["green"],
        "Bind YAML kwargs",
        ("controller_fn = partial(fn, **kwargs) or fn",),
        12.2,
        8.8,
    )

    for a, b in [
        (start, load),
        (load, imp),
        (imp, bind),
    ]:
        out.append(
            direct(a, "bottom", b, "top")
        )

    # ==================================================================
    # Software / hardware controller selection
    # ==================================================================

    dhw = Dg("dhw", cx, 455, 250, 66)
    draw_decision(
        out,
        dhw,
        'cfg["hardware"]?',
        ts=12.8,
    )

    out.append(
        direct(bind, "bottom", dhw, "top")
    )

    # --------------------------------------------------------------
    # hardware == False:
    # use the imported Python function directly.
    # --------------------------------------------------------------

    sw_plain = Rg(
        "sw_plain",
        690,
        420,
        250,
        70,
    )

    draw_node(
        out,
        sw_plain,
        COL["dry"],
        "Use Python controller mirror",
        (
            "software plugin path",
            "fn / functools.partial",
        ),
        10.7,
        8.0,
        10,
    )

    out.append(
        direct(
            dhw,
            "right",
            sw_plain,
            "left",
            "edge-dry",
        )
    )

    branch_label(
        out,
        dhw.right + 45,
        dhw.cy - 10,
        "no",
        COL["dry"],
    )

    # --------------------------------------------------------------
    # hardware == True:
    # determine whether this run can actually use serial hardware.
    # --------------------------------------------------------------

    davail = Dg(
        "davail",
        cx,
        565,
        280,
        72,
    )

    draw_decision(
        out,
        davail,
        "dry_run=False AND",
        ("serial port supplied?",),
        11.8,
        8.8,
    )

    out.append(
        direct(
            dhw,
            "bottom",
            davail,
            "top",
            "edge-hil",
        )
    )

    branch_label(
        out,
        dhw.cx + 18,
        dhw.bottom + 18,
        "yes",
        COL["hil"],
        "start",
    )

    # --------------------------------------------------------------
    # Hardware requested but unavailable in this run:
    # warn and retain the Python mirror.
    # --------------------------------------------------------------

    sw_fallback = Rg(
        "sw_fallback",
        670,
        530,
        270,
        70,
    )

    draw_node(
        out,
        sw_fallback,
        COL["dry"],
        "Warn + use Python mirror",
        (
            "hardware requested but this run is dry",
            "or no serial port was supplied",
        ),
        10.5,
        7.8,
        10,
    )

    out.append(
        direct(
            davail,
            "right",
            sw_fallback,
            "left",
            "edge-dry",
        )
    )

    branch_label(
        out,
        davail.right + 45,
        davail.cy - 10,
        "no",
        COL["dry"],
    )

    # --------------------------------------------------------------
    # Hardware is available.
    # --------------------------------------------------------------

    hw = Rg(
        "hw",
        360,
        640,
        280,
        80,
    )

    draw_node(
        out,
        hw,
        COL["red"],
        "HardwareControllerFn(port)",
        (
            "controller_fn = hw_fn",
            "first call opens/configures serial; later calls exchange V:/Q:",
        ),
        10.8,
        7.8,
        10,
    )

    out.append(
        direct(
            davail,
            "bottom",
            hw,
            "top",
            "edge-hil",
        )
    )

    branch_label(
        out,
        davail.cx + 18,
        davail.bottom + 18,
        "yes",
        COL["hil"],
        "start",
    )

    # ==================================================================
    # Software-path merge
    #
    # Do NOT route sw_plain vertically through sw_fallback.
    #
    # First combine the two software outcomes here.
    # ==================================================================

    sw_merge = C(
        sw_fallback.cx,   # 805: vertically aligned with fallback block
        625,
        11,
    )

    out.append(
        circle_svg(
            sw_merge,
            COL["neutral"],
        )
    )

    # Python-only path.
    #
    # Leave the RIGHT side of sw_plain and use an outer rail so the
    # path never passes through sw_fallback.
    software_outer_x = 958

    out.append(
        path(
            [
                sw_plain.a("right"),
                (software_outer_x, sw_plain.cy),
                (software_outer_x, sw_merge.cy),
                sw_merge.a("right", TARGET_GAP),
            ],
            "edge-dry",
        )
    )

    # Fallback Python path is vertically aligned with sw_merge.
    out.append(
        direct(
            sw_fallback,
            "bottom",
            sw_merge,
            "top",
            "edge-dry",
        )
    )

    # ==================================================================
    # Final selected-controller merge
    #
    # Hardware enters from TOP.
    # Unified software path enters from RIGHT.
    # ==================================================================

    sel = C(
        cx,
        790,
        13,
    )

    out.append(
        circle_svg(
            sel,
            COL["neutral"],
        )
    )

    # Software stream.
    out.append(
        path(
            [
                sw_merge.a("bottom"),
                (sw_merge.cx, sel.cy),
                sel.a("right", TARGET_GAP),
            ],
            "edge-dry",
        )
    )

    # Hardware stream.
    out.append(
        direct(
            hw,
            "bottom",
            sel,
            "top",
            "edge-hil",
        )
    )

    # ==================================================================
    # Benchmark configuration
    # ==================================================================

    dcfg = Dg(
        "dcfg",
        cx,
        875,
        255,
        62,
    )

    draw_decision(
        out,
        dcfg,
        "benchmark_config is None?",
        ts=11.8,
    )

    out.append(
        direct(
            sel,
            "bottom",
            dcfg,
            "top",
        )
    )

    defaults = Rg(
        "defaults",
        700,
        842,
        240,
        66,
    )

    draw_node(
        out,
        defaults,
        COL["blue"],
        "Create BenchmarkConfig()",
        ("framework defaults",),
        10.6,
        8.0,
        10,
    )

    out.append(
        direct(
            dcfg,
            "right",
            defaults,
            "left",
        )
    )

    branch_label(
        out,
        dcfg.right + 45,
        dcfg.cy - 10,
        "yes",
    )

    cfgmerge = C(
        cx,
        950,
        12,
    )

    out.append(
        circle_svg(
            cfgmerge,
            COL["neutral"],
        )
    )

    # Caller already supplied config.
    out.append(
        direct(
            dcfg,
            "bottom",
            cfgmerge,
            "top",
        )
    )

    branch_label(
        out,
        dcfg.cx + 18,
        dcfg.bottom + 18,
        "no",
        anchor="start",
    )

    # Default-created config returns from the right.
    out.append(
        path(
            [
                defaults.a("bottom"),
                (defaults.cx, cfgmerge.cy),
                (cfgmerge.right + 20, cfgmerge.cy),
                cfgmerge.a("right", TARGET_GAP),
            ],
            "edge-dark",
        )
    )

    # ==================================================================
    # Temporary plugin scenario registration
    # ==================================================================

    alloc = Rg(
        "alloc",
        205,
        990,
        590,
        66,
    )

    draw_node(
        out,
        alloc,
        COL["blue"],
        "Allocate plugin number and build ScenarioSpec",
        (
            "num = max(existing keys, 9) + 1; "
            "runner = _plugin_runner; supports_lv=False",
        ),
        11.6,
        8.4,
    )

    copycfg = Rg(
        "copycfg",
        205,
        1082,
        590,
        68,
    )

    draw_node(
        out,
        copycfg,
        COL["blue"],
        "Copy benchmark scenario list",
        (
            "append plugin number if absent; "
            "dataclasses.replace(config, scenarios=...)",
        ),
        11.5,
        8.3,
    )

    reg = Rg(
        "reg",
        205,
        1176,
        590,
        58,
    )

    draw_node(
        out,
        reg,
        COL["blue"],
        "SCENARIO_REGISTRY[num] = spec",
        ("temporary process-global registration",),
        11.8,
        8.5,
    )

    run = Rg(
        "run",
        205,
        1260,
        590,
        66,
    )

    draw_node(
        out,
        run,
        COL["blue"],
        "try: run_benchmark(..., config_run)",
        (
            "benchmark validates inputs, deep-copies net per scenario "
            "and dispatches plugin runner",
        ),
        11.2,
        8.1,
    )

    for a, b in [
        (cfgmerge, alloc),
        (alloc, copycfg),
        (copycfg, reg),
        (reg, run),
    ]:
        out.append(
            direct(
                a,
                "bottom",
                b,
                "top",
            )
        )

    # ==================================================================
    # finally cleanup
    #
    # Both normal and exceptional execution execute this block.
    # The two paths therefore ENTER cleanup separately.
    #
    # After cleanup:
    #   normal path       -> custom_result processing
    #   pending exception -> re-raise
    # ==================================================================

    cleanup = Rg(
        "cleanup",
        205,
        1360,
        590,
        72,
    )

    draw_node(
        out,
        cleanup,
        COL["purple"],
        "finally: cleanup always executes",
        (
            "SCENARIO_REGISTRY.pop(num)",
            "if hw_fn exists: hw_fn.close() -> Arduino END + port close",
        ),
        11.5,
        8.1,
    )

    # --------------------------------------------------------------
    # Normal run_benchmark completion -> finally.
    # --------------------------------------------------------------

    out.append(
        direct(
            run,
            "bottom",
            cleanup,
            "top",
        )
    )

    branch_label(
        out,
        run.cx + 18,
        run.bottom + 20,
        "normal",
        anchor="start",
        size=8.8,
    )

    # --------------------------------------------------------------
    # Exception from run_benchmark -> the SAME finally block.
    #
    # Enter the upper-left portion of cleanup instead of cleanup.left,
    # leaving cleanup.left free for the re-raise output.
    # --------------------------------------------------------------

    exception_rail_x = 160
    exception_entry_x = cleanup.left + 58
    exception_entry_y = cleanup.top - TARGET_GAP

    out.append(
        path(
            [
                run.a("left"),
                (exception_rail_x, run.cy),
                (exception_rail_x, 1340),
                (exception_entry_x, 1340),
                (exception_entry_x, exception_entry_y),
            ],
            "edge-hil",
        )
    )

    branch_label(
        out,
        exception_rail_x + 6,
        1334,
        "exception -> finally",
        COL["hil"],
        anchor="start",
        size=8.3,
    )

    # --------------------------------------------------------------
    # Exception resumes AFTER finally.
    # --------------------------------------------------------------

    propagate = Rg(
        "propagate",
        20,
        1362,
        160,
        68,
    )

    draw_node(
        out,
        propagate,
        COL["red"],
        "Re-raise exception",
        ("after the same finally cleanup",),
        9.5,
        7.2,
        9,
    )

    # cleanup.cy == propagate.cy, so this is a perfectly straight
    # horizontal exception-resume path.
    out.append(
        direct(
            cleanup,
            "left",
            propagate,
            "right",
            "edge-hil",
        )
    )

    branch_label(
        out,
        cleanup.left - 12,
        cleanup.cy - 10,
        "re-raise",
        COL["hil"],
        anchor="end",
        size=8.1,
    )

    # ==================================================================
    # Normal continuation after cleanup
    # ==================================================================

    dnone = Dg(
        "dnone",
        cx,
        1515,
        250,
        62,
    )

    draw_decision(
        out,
        dnone,
        "custom_result is None?",
        ts=11.8,
    )

    out.append(
        direct(
            cleanup,
            "bottom",
            dnone,
            "top",
        )
    )

    branch_label(
        out,
        cleanup.cx + 18,
        cleanup.bottom + 18,
        "normal continuation",
        anchor="start",
        size=8.5,
    )

    fail = Rg(
        "fail",
        35,
        1478,
        220,
        74,
    )

    draw_node(
        out,
        fail,
        COL["red"],
        "Raise RuntimeError",
        (
            "custom scenario failed inside benchmark",
            "include isolated runner traceback",
        ),
        10.2,
        7.7,
        10,
    )

    out.append(
        direct(
            dnone,
            "left",
            fail,
            "right",
            "edge-hil",
        )
    )

    branch_label(
        out,
        dnone.left - 50,
        dnone.cy - 10,
        "yes",
        COL["hil"],
    )

    # ==================================================================
    # Stored-summary shortcut
    # ==================================================================

    ddict = Dg(
        "ddict",
        cx,
        1625,
        270,
        62,
    )

    draw_decision(
        out,
        ddict,
        "custom_result is summary dict?",
        ts=11.2,
    )

    out.append(
        direct(
            dnone,
            "bottom",
            ddict,
            "top",
        )
    )

    branch_label(
        out,
        dnone.cx + 18,
        dnone.bottom + 18,
        "no",
        anchor="start",
    )

    recon = Rg(
        "recon",
        690,
        1589,
        250,
        72,
    )

    draw_node(
        out,
        recon,
        COL["purple"],
        "Reconstruct ScenarioResult",
        (
            "layer-1 already-complete shortcut",
            "aggregate fields restored; records list empty",
        ),
        10.2,
        7.6,
        10,
    )

    out.append(
        direct(
            ddict,
            "right",
            recon,
            "left",
        )
    )

    branch_label(
        out,
        ddict.right + 45,
        ddict.cy - 10,
        "yes",
    )

    # ==================================================================
    # Summary/result merge
    #
    # Keep a clear vertical separation before return_benchmark?.
    # ==================================================================

    rmerge = C(
        cx,
        1688,
        12,
    )

    out.append(
        circle_svg(
            rmerge,
            COL["neutral"],
        )
    )

    # Already a ScenarioResult.
    out.append(
        direct(
            ddict,
            "bottom",
            rmerge,
            "top",
        )
    )

    branch_label(
        out,
        ddict.cx + 18,
        ddict.bottom + 18,
        "no",
        anchor="start",
    )

    # Reconstructed ScenarioResult.
    out.append(
        path(
            [
                recon.a("bottom"),
                (recon.cx, rmerge.cy),
                rmerge.a("right", TARGET_GAP),
            ],
            "edge-dark",
        )
    )

    # ==================================================================
    # Public return form
    # ==================================================================

    dret = Dg(
        "dret",
        cx,
        1760,
        220,
        58,
    )

    draw_decision(
        out,
        dret,
        "return_benchmark?",
        ts=11.6,
    )

    out.append(
        direct(
            rmerge,
            "bottom",
            dret,
            "top",
        )
    )

    r1 = Rg(
        "r1",
        175,
        1804,
        245,
        48,
    )

    r2 = Rg(
        "r2",
        585,
        1804,
        310,
        48,
    )

    draw_node(
        out,
        r1,
        COL["purple"],
        "return custom_result",
        (),
        10.8,
    )

    draw_node(
        out,
        r2,
        COL["purple"],
        "return (custom_result, bench)",
        (),
        10.6,
    )

    # Simple left then down.
    out.append(
        path(
            [
                dret.a("left"),
                (r1.cx, dret.cy),
                r1.a("top", TARGET_GAP),
            ],
            "edge-dark",
        )
    )

    branch_label(
        out,
        dret.left - 35,
        dret.cy - 10,
        "no",
    )

    # Simple right then down.
    out.append(
        path(
            [
                dret.a("right"),
                (r2.cx, dret.cy),
                r2.a("top", TARGET_GAP),
            ],
            "edge-dark",
        )
    )

    branch_label(
        out,
        dret.right + 35,
        dret.cy - 10,
        "yes",
    )

    # ==================================================================
    # Geometry checks
    # ==================================================================

    for n, g in rects.items():
        audit_rect_bounds(
            n,
            g,
            W,
            H,
            8,
        )

    for n, g in diamonds.items():
        audit_diamond_bounds(
            n,
            g,
            W,
            H,
            8,
        )

    # Software outer rail must remain outside both software blocks.
    assert software_outer_x > sw_plain.right + 10
    assert software_outer_x > sw_fallback.right + 10

    # Fallback block and first software merge are deliberately aligned.
    assert abs(sw_merge.cx - sw_fallback.cx) < 1e-9

    # The final result merge must have real clearance from the
    # return_benchmark decision.
    assert dret.top - rmerge.bottom >= 25

    # Final return blocks must remain inside the canvas.
    assert r1.bottom <= H - 8
    assert r2.bottom <= H - 8

    write(
        "flow_plugin_reg_ieee_v4",
        W,
        H,
        "Custom controller plugin registration and routing - IEEE",
        "\n".join(out),
    )

def build_registration_presentation():
    W, H = 1920, 1080

    out = [
        rect_svg(
            R(0, 0, W, H),
            COL["panel"],
            0,
        )
    ]

    rects = {}
    diamonds = {}

    def Rg(n, x, y, w, h):
        g = R(x, y, w, h)
        rects[n] = g
        return g

    def Dg(n, x, y, w, h):
        g = D(x, y, w, h)
        diamonds[n] = g
        return g

    cx = 480

    # ==================================================================
    # Public entry point and plugin loading
    # ==================================================================

    start = Rg(
        "start",
        145, 18,
        670, 44,
    )

    draw_node(
        out,
        start,
        COL["blue"],
        "register_and_run(yaml, net, profiles, network_id, config, port)",
        ("public controller-plugin entry point",),
        12.4,
        8.2,
    )

    load = Rg(
        "load",
        145, 78,
        670, 52,
    )

    draw_node(
        out,
        load,
        COL["green"],
        "Load + validate YAML; import named controller function; bind kwargs",
        (
            "paths resolve relative to YAML; "
            "software function is always loaded first",
        ),
        11.7,
        8.0,
    )

    out.append(
        direct(
            start,
            "bottom",
            load,
            "top",
        )
    )

    # ==================================================================
    # Software / hardware selection
    # ==================================================================

    dhw = Dg(
        "dhw",
        cx, 176,
        230, 44,
    )

    draw_decision(
        out,
        dhw,
        "hardware:true?",
        ts=11.5,
    )

    out.append(
        direct(
            load,
            "bottom",
            dhw,
            "top",
        )
    )

    # --------------------------------------------------------------
    # hardware == False
    # --------------------------------------------------------------

    sw1 = Rg(
        "sw1",
        630, 150,
        250, 52,
    )

    draw_node(
        out,
        sw1,
        COL["dry"],
        "Python mirror",
        ("software controller_fn",),
        10.1,
        7.4,
        9,
    )

    out.append(
        direct(
            dhw,
            "right",
            sw1,
            "left",
            "edge-dry",
        )
    )

    branch_label(
        out,
        dhw.right + 34,
        dhw.cy - 7,
        "no",
        COL["dry"],
        size=8.5,
    )

    # --------------------------------------------------------------
    # hardware == True
    # --------------------------------------------------------------

    dav = Dg(
        "dav",
        cx, 244,
        250, 48,
    )

    draw_decision(
        out,
        dav,
        "dry_run=False AND port supplied?",
        ts=10.6,
    )

    out.append(
        direct(
            dhw,
            "bottom",
            dav,
            "top",
            "edge-hil",
        )
    )

    branch_label(
        out,
        dhw.cx + 14,
        dhw.bottom + 14,
        "yes",
        COL["hil"],
        "start",
        8.5,
    )

    # --------------------------------------------------------------
    # Hardware requested, but unavailable for this run
    # --------------------------------------------------------------

    sw2 = Rg(
        "sw2",
        620, 217,
        260, 54,
    )

    draw_node(
        out,
        sw2,
        COL["dry"],
        "Warn + Python mirror",
        ("dry run or missing serial port",),
        9.8,
        7.2,
        9,
    )

    out.append(
        direct(
            dav,
            "right",
            sw2,
            "left",
            "edge-dry",
        )
    )

    branch_label(
        out,
        dav.right + 34,
        dav.cy - 7,
        "no",
        COL["dry"],
        size=8.5,
    )

    # --------------------------------------------------------------
    # Hardware available
    # --------------------------------------------------------------

    hw = Rg(
        "hw",
        350, 304,
        260, 64,
    )

    draw_node(
        out,
        hw,
        COL["red"],
        "HardwareControllerFn(port)",
        (
            "first call opens + configures Arduino",
            "per call V:/Q: exchange; close later sends END",
        ),
        9.8,
        7.2,
        9,
    )

    out.append(
        direct(
            dav,
            "bottom",
            hw,
            "top",
            "edge-hil",
        )
    )

    branch_label(
        out,
        dav.cx + 14,
        dav.bottom + 14,
        "yes",
        COL["hil"],
        "start",
        8.5,
    )

    # ==================================================================
    # Software merge
    #
    # IMPORTANT:
    # sw1.cx lies inside sw2, so sw1 must NOT leave vertically from its
    # bottom centre. Route it around the right side instead.
    # ==================================================================

    sw_merge = C(
        sw2.cx,   # 750
        300,
        10,
    )

    out.append(
        circle_svg(
            sw_merge,
            COL["neutral"],
        )
    )

    # Outer software rail stays outside both green blocks.
    software_outer_x = 910

    # Plain software path:
    #
    # sw1.right -> outside rail -> down -> sw_merge.right
    out.append(
        path(
            [
                sw1.a("right"),
                (software_outer_x, sw1.cy),
                (software_outer_x, sw_merge.cy),
                sw_merge.a("right", TARGET_GAP),
            ],
            "edge-dry",
        )
    )

    # Fallback software block is centred on sw_merge,
    # therefore it can enter vertically.
    out.append(
        direct(
            sw2,
            "bottom",
            sw_merge,
            "top",
            "edge-dry",
        )
    )

    # ==================================================================
    # Final controller-selection merge
    #
    # Hardware enters from top.
    # Unified software path enters from right.
    # ==================================================================

    m = C(
        cx,
        406,
        10,
    )

    out.append(
        circle_svg(
            m,
            COL["neutral"],
        )
    )

    # Hardware path.
    out.append(
        direct(
            hw,
            "bottom",
            m,
            "top",
            "edge-hil",
        )
    )

    # Unified software path.
    #
    # Stay to the right of HardwareControllerFn until below it,
    # then enter m from the right.
    out.append(
        path(
            [
                sw_merge.a("bottom"),
                (sw_merge.cx, m.cy),
                m.a("right", TARGET_GAP),
            ],
            "edge-dry",
        )
    )

    # ==================================================================
    # BenchmarkConfig selection
    # ==================================================================

    dcfg = Dg(
        "dcfg",
        cx, 458,
        220, 42,
    )

    draw_decision(
        out,
        dcfg,
        "benchmark_config is None?",
        ts=10.6,
    )

    out.append(
        direct(
            m,
            "bottom",
            dcfg,
            "top",
        )
    )

    defaults = Rg(
        "defaults",
        680, 436,
        235, 44,
    )

    draw_node(
        out,
        defaults,
        COL["blue"],
        "Create BenchmarkConfig()",
        ("otherwise keep caller config",),
        9.3,
        7.0,
        9,
    )

    out.append(
        direct(
            dcfg,
            "right",
            defaults,
            "left",
        )
    )

    branch_label(
        out,
        dcfg.right + 35,
        dcfg.cy - 7,
        "yes",
        size=8.5,
    )

    cfgmerge = C(
        cx,
        510,
        9,
    )

    out.append(
        circle_svg(
            cfgmerge,
            COL["neutral"],
        )
    )

    # Existing caller config.
    out.append(
        direct(
            dcfg,
            "bottom",
            cfgmerge,
            "top",
        )
    )

    branch_label(
        out,
        dcfg.cx + 12,
        dcfg.bottom + 13,
        "no",
        anchor="start",
        size=8.5,
    )

    # Default config rejoins from the right.
    out.append(
        path(
            [
                defaults.a("bottom"),
                (defaults.cx, cfgmerge.cy),
                cfgmerge.a("right", TARGET_GAP),
            ],
            "edge-dark",
        )
    )

    # ==================================================================
    # Temporary plugin registration
    # ==================================================================

    alloc = Rg(
        "alloc",
        120, 540,
        720, 52,
    )

    draw_node(
        out,
        alloc,
        COL["blue"],
        "Allocate num >= 10; build ScenarioSpec; "
        "copy scenario list and append plugin num if absent",
        (
            "temporary runner captures controller_fn, scenario_id, "
            "label, gate and clamp settings",
        ),
        10.9,
        7.7,
    )

    reg = Rg(
        "reg",
        120, 610,
        720, 46,
    )

    draw_node(
        out,
        reg,
        COL["blue"],
        "Temporarily register ScenarioSpec in SCENARIO_REGISTRY",
        ("benchmark sees the plugin as one additional scenario",),
        10.8,
        7.6,
    )

    run = Rg(
        "run",
        120, 674,
        720, 52,
    )

    draw_node(
        out,
        run,
        COL["blue"],
        "try: run_benchmark(net, profiles, config_run)",
        (
            "benchmark validates inputs, deep-copies net per scenario "
            "and dispatches _plugin_runner",
        ),
        10.8,
        7.6,
    )

    cleanup = Rg(
        "cleanup",
        120, 744,
        720, 54,
    )

    draw_node(
        out,
        cleanup,
        COL["purple"],
        "finally: remove temporary registry entry; close hw_fn if created",
        (
            "hardware close sends END before releasing the serial port",
        ),
        10.5,
        7.5,
    )

    for a, b in [
        (cfgmerge, alloc),
        (alloc, reg),
        (reg, run),
    ]:
        out.append(
            direct(
                a,
                "bottom",
                b,
                "top",
            )
        )

    # ==================================================================
    # run_benchmark -> finally
    #
    # Normal execution enters finally from the top.
    # Exception execution enters a distinct top-left position.
    # ==================================================================

    out.append(
        direct(
            run,
            "bottom",
            cleanup,
            "top",
        )
    )

    branch_label(
        out,
        run.cx + 14,
        run.bottom + 16,
        "normal",
        anchor="start",
        size=7.8,
    )

    # Exception path into finally.
    exception_rail_x = 82
    exception_entry_x = cleanup.left + 46
    exception_entry_y = cleanup.top - TARGET_GAP

    out.append(
        path(
            [
                run.a("left"),
                (exception_rail_x, run.cy),
                (exception_rail_x, 734),
                (exception_entry_x, 734),
                (exception_entry_x, exception_entry_y),
            ],
            "edge-hil",
        )
    )

    branch_label(
        out,
        exception_rail_x + 5,
        730,
        "exception -> finally",
        COL["hil"],
        anchor="start",
        size=7.2,
    )

    # ==================================================================
    # Pending exception after finally
    #
    # Keep this block ABOVE RuntimeError so the red blocks do not overlap.
    # ==================================================================

    prop = Rg(
        "prop",
        8, 748,
        90, 46,
    )

    draw_node(
        out,
        prop,
        COL["red"],
        "Re-raise",
        ("after finally",),
        7.8,
        6.1,
        8,
    )

    # cleanup.cy == prop.cy, giving a straight horizontal exit.
    out.append(
        direct(
            cleanup,
            "left",
            prop,
            "right",
            "edge-hil",
        )
    )

    branch_label(
        out,
        cleanup.left - 10,
        cleanup.cy - 8,
        "re-raise",
        COL["hil"],
        anchor="end",
        size=7.1,
    )

    # ==================================================================
    # Normal continuation after finally
    # ==================================================================

    dnone = Dg(
        "dnone",
        cx, 846,
        220, 42,
    )

    draw_decision(
        out,
        dnone,
        "custom_result is None?",
        ts=10.4,
    )

    out.append(
        direct(
            cleanup,
            "bottom",
            dnone,
            "top",
        )
    )

    branch_label(
        out,
        cleanup.cx + 12,
        cleanup.bottom + 14,
        "normal continuation",
        anchor="start",
        size=7.5,
    )

    # ==================================================================
    # Plugin failed inside benchmark isolation
    # ==================================================================

    runtime = Rg(
        "runtime",
        25, 822,
        230, 48,
    )

    draw_node(
        out,
        runtime,
        COL["red"],
        "Raise RuntimeError",
        ("custom plugin failed inside benchmark",),
        9.5,
        7.1,
        9,
    )

    out.append(
        direct(
            dnone,
            "left",
            runtime,
            "right",
            "edge-hil",
        )
    )

    branch_label(
        out,
        dnone.left - 38,
        dnone.cy - 7,
        "yes",
        COL["hil"],
        size=8.5,
    )

    # ==================================================================
    # Stored-summary shortcut
    # ==================================================================

    ddict = Dg(
        "ddict",
        cx, 912,
        230, 42,
    )

    draw_decision(
        out,
        ddict,
        "summary dict shortcut?",
        ts=10.5,
    )

    out.append(
        direct(
            dnone,
            "bottom",
            ddict,
            "top",
        )
    )

    branch_label(
        out,
        dnone.cx + 12,
        dnone.bottom + 13,
        "no",
        anchor="start",
        size=8.5,
    )

    recon = Rg(
        "recon",
        680, 888,
        250, 48,
    )

    draw_node(
        out,
        recon,
        COL["purple"],
        "Reconstruct ScenarioResult",
        ("aggregate fields from stored summary",),
        9.4,
        7.1,
        9,
    )

    out.append(
        direct(
            ddict,
            "right",
            recon,
            "left",
        )
    )

    branch_label(
        out,
        ddict.right + 35,
        ddict.cy - 7,
        "yes",
        size=8.5,
    )

    # ==================================================================
    # ScenarioResult merge
    # ==================================================================

    mr = C(
        cx,
        964,
        9,
    )

    out.append(
        circle_svg(
            mr,
            COL["neutral"],
        )
    )

    # Already a ScenarioResult.
    out.append(
        direct(
            ddict,
            "bottom",
            mr,
            "top",
        )
    )

    branch_label(
        out,
        ddict.cx + 12,
        ddict.bottom + 13,
        "no",
        anchor="start",
        size=8.5,
    )

    # Reconstructed ScenarioResult.
    out.append(
        path(
            [
                recon.a("bottom"),
                (recon.cx, mr.cy),
                mr.a("right", TARGET_GAP),
            ],
            "edge-dark",
        )
    )

    # ==================================================================
    # Public return mode
    # ==================================================================

    dret = Dg(
        "dret",
        cx, 1018,
        190, 38,
    )

    draw_decision(
        out,
        dret,
        "return_benchmark?",
        ts=10.2,
    )

    out.append(
        direct(
            mr,
            "bottom",
            dret,
            "top",
        )
    )

    r1 = Rg(
        "r1",
        180, 1042,
        235, 32,
    )

    r2 = Rg(
        "r2",
        555, 1042,
        310, 32,
    )

    draw_node(
        out,
        r1,
        COL["purple"],
        "return ScenarioResult",
        (),
        9.5,
    )

    draw_node(
        out,
        r2,
        COL["purple"],
        "return ScenarioResult + BenchmarkResult",
        (),
        9.1,
    )

    # Straight left, then down.
    out.append(
        path(
            [
                dret.a("left"),
                (r1.cx, dret.cy),
                r1.a("top", TARGET_GAP),
            ],
            "edge-dark",
        )
    )

    branch_label(
        out,
        dret.left - 28,
        dret.cy - 7,
        "no",
        size=8.3,
    )

    # Straight right, then down.
    out.append(
        path(
            [
                dret.a("right"),
                (r2.cx, dret.cy),
                r2.a("top", TARGET_GAP),
            ],
            "edge-dark",
        )
    )

    branch_label(
        out,
        dret.right + 28,
        dret.cy - 7,
        "yes",
        size=8.3,
    )

    # ==================================================================
    # Presentation explanation panels
    # ==================================================================

    px, pw = 1060, 820

    panels = [
        R(px, 24,  pw, 255),
        R(px, 301, pw, 320),
        R(px, 643, pw, 395),
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
            "Plugin configuration and import",
            COL["text"],
            17,
            700,
            "start",
        )
    )

    lines = [
        "YAML defines a unique scenario name and label, Python module path, function, hardware flag and optional kwargs.",
        "load_plugin validates the schema and resolves module/firmware paths relative to the YAML directory.",
        "The controller module is imported directly from its file path and the configured function must be callable.",
        "Configured kwargs are bound before execution, so the scenario runner always sees controller_fn(vm_pu, p_mw).",
        "gate_clean_timesteps and clamp_to_net_limits are carried into the custom scenario unchanged.",
        "For hardware:true, the YAML also identifies the firmware sketch used for this run's traceability.",
    ]

    for i, t in enumerate(lines):
        out.append(
            label(
                px + 24,
                92 + 30 * i,
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
            334,
            "Software / hardware controller routing",
            COL["text"],
            17,
            700,
            "start",
        )
    )

    lines = [
        "The Python controller is always available as the software execution path and dry-run mirror.",
        "hardware:true selects Arduino execution only when the benchmark is not dry-run and a serial port is supplied.",
        "Otherwise execution remains on the Python controller and the benchmark continues without opening serial hardware.",
        "HardwareControllerFn owns the Arduino interface: first call opens the port and performs INIT/CFG/P configuration.",
        "Each controller call sends DER voltages and receives Q setpoints through the standard V:/Q: exchange.",
        "If installed-P metadata changes, the hardware wrapper reconfigures before the next exchange.",
        "The wrapper is closed in the outer finally block, so END is sent on both successful and failed benchmark runs.",
    ]

    for i, t in enumerate(lines):
        out.append(
            label(
                px + 24,
                369 + 32 * i,
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
            676,
            "Benchmark registration and public return",
            COL["text"],
            17,
            700,
            "start",
        )
    )

    lines = [
        "A plugin receives a dynamic scenario number >= 10 and a temporary ScenarioSpec pointing to _plugin_runner.",
        "The caller's BenchmarkConfig is copied; the plugin number is added without mutating the caller's scenario list.",
        "run_benchmark executes the custom controller alongside the selected built-in scenarios on isolated network copies.",
        "The registry entry is removed after the run so later registrations start from a clean framework state.",
        "If the plugin scenario failed inside benchmark isolation, register_and_run raises with the stored traceback.",
        "A stored already-complete summary is converted back to a ScenarioResult at this public API boundary.",
        "return_benchmark=False returns the custom ScenarioResult only; True additionally returns the full BenchmarkResult.",
    ]

    for i, t in enumerate(lines):
        out.append(
            label(
                px + 24,
                712 + 40 * i,
                t,
                COL["text"],
                11.1,
                600,
                "start",
            )
        )

    # ==================================================================
    # Geometry checks
    # ==================================================================

    for n, g in rects.items():
        audit_rect_bounds(
            n,
            g,
            W,
            H,
            0,
        )

    for n, g in diamonds.items():
        audit_diamond_bounds(
            n,
            g,
            W,
            H,
            0,
        )

    # sw1 cannot drop vertically through sw2.
    assert sw1.cx >= sw2.left
    assert sw1.cx <= sw2.right

    # Outer software rail must remain clear of both software blocks.
    assert software_outer_x > sw1.right + 20
    assert software_outer_x > sw2.right + 20

    # Fallback block and software merge deliberately share a centreline.
    assert abs(sw_merge.cx - sw2.cx) < 1e-9

    # Re-raise and RuntimeError must not overlap vertically.
    assert prop.bottom < runtime.top

    # Final merge requires real clearance before return decision.
    assert dret.top - mr.bottom >= 20

    # Returns stay inside slide.
    assert r1.bottom <= H
    assert r2.bottom <= H

    write(
        "flow_plugin_reg_presentation_v4",
        W,
        H,
        "Custom controller plugin registration - presentation",
        "\n".join(out),
    )

def build_loop_ieee():
    W,H=1040,1900; out=[rect_svg(R(14,14,W-28,H-28),COL["panel"],16,COL["panel_border"],1.5)]
    rects={}; diamonds={}
    def Rg(n,x,y,w,h): g=R(x,y,w,h); rects[n]=g; return g
    def Dg(n,x,y,w,h): g=D(x,y,w,h); diamonds[n]=g; return g
    cx=570; fail_x=55; loop_x=30
    out.append(label(W/2,38,"Custom Controller Plugin - Detailed Timestep Execution",COL["text"],17,700))
    init=Rg("init",280,62,580,64); draw_node(out,init,COL["blue"],"adapt_profiles(net, profiles)",("derive load P/Q, DER P, time axis and dt",),12.5,8.9)
    dder=Dg("dder",cx,180,240,62); draw_decision(out,dder,"ap.der_p empty?",ts=12.2); out.append(direct(init,"bottom",dder,"top"))
    noders=Rg("noders",55,146,220,68); draw_node(out,noders,COL["red"],"Raise ValueError",("custom controller requires profiled DERs",),10.2,7.8,10); out.append(direct(dder,"left",noders,"right","edge-hil")); branch_label(out,dder.left-48,dder.cy-9,"yes",COL["hil"])
    prep=Rg("prep",280,230,580,78); draw_node(out,prep,COL["green"],"Initialize custom scenario",("optional on_scenario_start; construct dry-run VoltVarController for DER metadata/Q clamp", "clear net.controller + pp.reset_results; runpp uses voltage_depend_loads=False"),11.2,8.1)
    out.append(direct(dder,"bottom",prep,"top")); branch_label(out,dder.cx+18,dder.bottom+18,"no",anchor="start")
    ckpt=Rg("ckpt",280,334,580,66); draw_node(out,ckpt,COL["purple"],"Optional checkpoint recovery",("get_resume_records when publisher + checkpointing + method available; compute start_t",),11.2,8.1)
    out.append(direct(prep,"bottom",ckpt,"top"))
    dcomplete=Dg("dcomplete",cx,465,270,64); draw_decision(out,dcomplete,"checkpoint covers all T?",ts=11.5); out.append(direct(ckpt,"bottom",dcomplete,"top"))
    early=Rg("early",775,429,235,72); draw_node(out,early,COL["purple"],"Return resumed result",("ScenarioResult.from_records", "optional on_scenario_end"),10.1,7.7,10); out.append(direct(dcomplete,"right",early,"left","edge-gate")); branch_label(out,dcomplete.right+44,dcomplete.cy-9,"yes",COL["gate"])
    loop=Dg("loop",cx,560,210,54); draw_decision(out,loop,"for t = start_t ... T-1",ts=11.1,fill=COL["neutral"]); out.append(direct(dcomplete,"bottom",loop,"top")); branch_label(out,dcomplete.cx+18,dcomplete.bottom+18,"no",anchor="start")

    state=Rg("state",250,610,640,82); draw_node(out,state,COL["blue"],"[A,B,0,1] Write timestep inputs and pre-PF invariant",("p_target in controlled sgen order; load P/Q; derive profile totals", "write DER P=p_target and reset controlled DER Q=0"),11.3,8.2)
    pre=Rg("pre",250,718,640,62); draw_node(out,pre,COL["blue"],"[2] Pre-control power flow",("try pp.runpp(); detect_violations(v_min, v_max)",),11.5,8.4)
    out += [direct(loop,"bottom",state,"top"),direct(state,"bottom",pre,"top")]
    dpre=Dg("dpre",cx,850,240,62); draw_decision(out,dpre,"pre-PF converged?",ts=11.7); out.append(direct(pre,"bottom",dpre,"top"))
    prefail=Rg("prefail",65,814,245,72); draw_node(out,prefail,COL["red"],"Pre-PF failed record",("converged=False; populate timing/P/load", "on_timestep if publisher; append; continue"),10.0,7.6,10); out.append(direct(dpre,"left",prefail,"right","edge-hil")); branch_label(out,dpre.left-48,dpre.cy-9,"no",COL["hil"])

    dgate=Dg("dgate",cx,960,285,72); draw_decision(out,dgate,"gate_clean_timesteps AND",("no pre-PF violations?",),11.4,8.6); out.append(direct(dpre,"bottom",dgate,"top")); branch_label(out,dpre.cx+18,dpre.bottom+18,"yes",anchor="start")
    clean=Rg("clean",65,922,300,76); draw_node(out,clean,COL["neutral"],"Clean-gate settled record",("hold Q=0 and reuse pre-PF state", "q_applied=0; p_applied=p_target; losses/import"),10.2,7.7,10)
    out.append(direct(dgate,"left",clean,"right","edge-gate")); branch_label(out,dgate.left-50,dgate.cy-10,"yes",COL["gate"])
    call=Rg("call",390,1044,360,78); draw_node(out,call,COL["green"],"[3] controller_fn(vm_pre, p_installed)",("read pre-PF voltage at each DER bus", "validate returned Q: 1-D, correct length, finite"),10.9,8.0)
    out.append(direct(dgate,"bottom",call,"top")); branch_label(out,dgate.cx+18,dgate.bottom+18,"no",anchor="start")
    cerr=Rg("cerr",790,1047,220,72); draw_node(out,cerr,COL["red"],"Controller exception",("plugin/validation/serial error propagates", "benchmark isolates scenario as failed"),9.7,7.4,9)
    out.append(direct(call,"right",cerr,"left","edge-hil")); branch_label(out,call.right+52,call.cy-9,"exception",COL["hil"],size=8.4)

    dclamp=Dg("dclamp",cx,1175,240,60); draw_decision(out,dclamp,"clamp_to_net_limits?",ts=11.2); out.append(path([call.a("bottom"),(call.cx,1140),(dclamp.cx,1140),dclamp.a("top",TARGET_GAP)],"edge-dark"))
    clamp=Rg("clamp",780,1144,230,62); draw_node(out,clamp,COL["green"],"Clamp Q setpoints",("explicit min/max Q plus inverter S-circle",),9.8,7.4,9)
    out.append(direct(dclamp,"right",clamp,"left")); branch_label(out,dclamp.right+42,dclamp.cy-9,"yes")
    qm=C(cx,1248,12); out.append(circle_svg(qm,COL["neutral"])); out.append(direct(dclamp,"bottom",qm,"top")); branch_label(out,dclamp.cx+18,dclamp.bottom+17,"no",anchor="start"); out.append(path([clamp.a("bottom"),(clamp.cx,qm.cy),(qm.right+18,qm.cy),qm.a("right",TARGET_GAP)],"edge-dark"))
    applyq=Rg("applyq",390,1280,360,58); draw_node(out,applyq,COL["blue"],"[4] Apply Q to controlled DERs",("P remains p_target; no dynamics or ramp stage",),10.7,7.8); out.append(direct(qm,"bottom",applyq,"top"))
    post=Rg("post",390,1362,360,62); draw_node(out,post,COL["blue"],"[5] Post-control power flow",("try pp.runpp(); detect_violations; converged = PF ok AND report.converged",),10.4,7.5); out.append(direct(applyq,"bottom",post,"top"))
    dpost=Dg("dpost",cx,1492,240,62); draw_decision(out,dpost,"post-PF converged?",ts=11.4); out.append(direct(post,"bottom",dpost,"top"))

        # ------------------------------------------------------------------
    # Post-PF outcome
    # ------------------------------------------------------------------

    # Narrower failure block leaves a dedicated vertical channel between
    # its right edge and the main execution column.
    bad=Rg("bad",70,1456,205,72)
    draw_node(
        out,bad,COL["red"],
        "Post-PF failed record",
        (
            "make_record_from_report(converged=False)",
            "Q/P settled fields are not attached",
        ),
        9.6,7.3,10
    )

    good=Rg("good",770,1454,240,76)
    draw_node(
        out,good,COL["purple"],
        "Settled controlled record",
        (
            "q_applied=q_arr; p_applied=p_target",
            "losses/import; curtailment_needed=False",
        ),
        9.7,7.3,10
    )

    out.append(
        direct(dpost,"left",bad,"right","edge-hil")
    )
    branch_label(
        out,dpost.left-48,dpost.cy-9,
        "no",COL["hil"]
    )

    out.append(
        direct(dpost,"right",good,"left")
    )
    branch_label(
        out,dpost.right+48,dpost.cy-9,
        "yes"
    )

    # ------------------------------------------------------------------
    # Record-path merging
    #
    # Do NOT force clean-gate + post-PF-failure + successful-control
    # onto the same side of one circle.
    #
    # Stage 1:
    #   clean gate --------\
    #                       > left_recm
    #   post-PF failure ---/
    #
    # Stage 2:
    #   left_recm --------\
    #                      > recm -> [F]
    #   settled control ---/
    # ------------------------------------------------------------------

    left_recm = C(343, 1560, 11)
    recm=C(cx,1560,13)

    out.append(circle_svg(left_recm,COL["neutral"]))
    out.append(circle_svg(recm,COL["neutral"]))

    # Clean-gate route.
    #
    # IMPORTANT:
    # do not use clean.a("bottom"), because clean.cx = 215 and that rail
    # would run through the post-PF failed block.
    #
    # Exit near the bottom-right of the clean block instead.
    clean_exit_x = clean.right - 22        # 343 px

    out.append(
        path(
            [
                (clean_exit_x, clean.bottom),
                left_recm.a("top", TARGET_GAP),
            ],
            "edge-gate"
        )
    )

    # Post-PF failure enters the LEFT side of the first merge.
    # Its horizontal segment lies below the failed block and does not
    # intersect the clean-gate vertical rail.
    out.append(
        path(
            [
                bad.a("bottom"),
                (bad.cx, left_recm.cy),
                left_recm.a("left", TARGET_GAP),
            ],
            "edge-hil"
        )
    )

    # The two left-side outcomes are now one logical stream.
    out.append(
        direct(
            left_recm,"right",
            recm,"left",
            "edge-dark"
        )
    )

    # Successful controlled result gets its own independent entry
    # from the RIGHT side.
    out.append(
        path(
            [
                good.a("bottom"),
                (good.cx, recm.cy),
                recm.a("right",TARGET_GAP),
            ],
            "edge-dark"
        )
    )

    # ------------------------------------------------------------------
    # Common record finalisation
    # ------------------------------------------------------------------

    finish=Rg("finish",300,1590,540,68)
    draw_node(
        out,finish,COL["purple"],
        "[F] Finish normal/clean record",
        (
            "populate t_total_ms, p_target_mw, der_gen_mw and load_mw",
            "optional on_timestep(rec); append record",
        ),
        10.9,8.0
    )

    out.append(
        direct(recm,"bottom",finish,"top")
    )

    # ------------------------------------------------------------------
    # Periodic progress
    # ------------------------------------------------------------------

    d96=Dg("d96",cx,1698,220,56)
    draw_decision(
        out,d96,
        "t % 96 == 0?",
        ts=11.1
    )
    out.append(
        direct(finish,"bottom",d96,"top")
    )

    prog=Rg("prog",765,1667,245,62)
    draw_node(
        out,prog,COL["purple"],
        "Periodic progress",
        (
            "optional partial live CSV rewrite",
            "_periodic_log daily progress",
        ),
        9.6,7.3,9
    )

    out.append(
        direct(d96,"right",prog,"left","edge-gate")
    )
    branch_label(
        out,d96.right+42,d96.cy-8,
        "yes",COL["gate"]
    )

    pm=C(cx,1773,11)
    out.append(circle_svg(pm,COL["neutral"]))

    out.append(
        direct(d96,"bottom",pm,"top")
    )
    branch_label(
        out,d96.cx+16,d96.bottom+16,
        "no",
        anchor="start"
    )

    out.append(
        path(
            [
                prog.a("bottom"),
                (prog.cx,pm.cy),
                (pm.right+18,pm.cy),
                pm.a("right",TARGET_GAP),
            ],
            "edge-gate"
        )
    )

    # Pre-PF failure bypass remains completely independent of the
    # record merge above.
    out.append(
        path(
            [
                prefail.a("left"),
                (fail_x,prefail.cy),
                (fail_x,pm.cy),
                (pm.left-18,pm.cy),
                pm.a("left",TARGET_GAP),
            ],
            "edge-hil"
        )
    )

    # ------------------------------------------------------------------
    # Loop control
    # ------------------------------------------------------------------

    dmore=Dg("dmore",cx,1840,210,54)
    draw_decision(
        out,dmore,
        "more timesteps?",
        ts=11.0
    )

    out.append(
        direct(pm,"bottom",dmore,"top")
    )

    done=Rg("done",770,1809,240,62)
    draw_node(
        out,done,COL["purple"],
        "Loop complete",
        (
            "ScenarioResult.from_records",
            "optional on_scenario_end; return",
        ),
        9.7,7.4,9
    )

    out.append(
        direct(dmore,"right",done,"left")
    )
    branch_label(
        out,dmore.right+42,dmore.cy-8,
        "no"
    )

    # Direct left-side loopback. No unnecessary excursion to the
    # bottom of the SVG.
    out.append(
        path(
            [
                dmore.a("left"),
                (loop_x,dmore.cy),
                (loop_x,loop.cy),
                loop.a("left",TARGET_GAP),
            ],
            "edge-loop"
        )
    )

    branch_label(
        out,
        dmore.left-34,
        dmore.cy+14,
        "yes - next t",
        COL["loop"],
        "end",
        8.8
    )
    
    for n,g in rects.items(): audit_rect_bounds(n,g,W,H,8)
    for n,g in diamonds.items(): audit_diamond_bounds(n,g,W,H,8)
    assert loop_x < fail_x < prefail.left
    write("flow_plugin_loop_ieee_v4",W,H,"Custom controller plugin detailed timestep execution - IEEE","\n".join(out))


def build_loop_presentation():
    W,H=1920,1080; out=[rect_svg(R(0,0,W,H),COL["panel"],0)]; rects={}; diamonds={}
    def Rg(n,x,y,w,h): g=R(x,y,w,h); rects[n]=g; return g
    def Dg(n,x,y,w,h): g=D(x,y,w,h); diamonds[n]=g; return g
    cx=520; fail_x=20; loop_x=6
    init=Rg("init",210,16,620,44); draw_node(out,init,COL["blue"],"adapt_profiles; require profiled DERs",("start publisher if present; initialize metadata/Q-clamp controller",),11.3,7.7)
    setup=Rg("setup",210,76,620,46); draw_node(out,setup,COL["green"],"Clear stale controllers/results; recover checkpoint when enabled",("compute start_t from recovered records",),10.9,7.5); out.append(direct(init,"bottom",setup,"top"))
    dcomplete=Dg("dcomplete",cx,164,230,42); draw_decision(out,dcomplete,"checkpoint covers all T?",ts=10.5); out.append(direct(setup,"bottom",dcomplete,"top"))
    early=Rg("early",760,141,245,46); draw_node(out,early,COL["purple"],"Return resumed ScenarioResult",("optional on_scenario_end",),9.4,7.0,9); out.append(direct(dcomplete,"right",early,"left","edge-gate")); branch_label(out,dcomplete.right+35,dcomplete.cy-7,"yes",COL["gate"],size=8.2)
    loop=Dg("loop",cx,222,190,38); draw_decision(out,loop,"for each t from start_t",ts=10.1,fill=COL["neutral"]); out.append(direct(dcomplete,"bottom",loop,"top")); branch_label(out,dcomplete.cx+12,dcomplete.bottom+12,"no",anchor="start",size=8.2)
    state=Rg("state",170,260,700,50); draw_node(out,state,COL["blue"],"[A,B,0,1] write p_target + load P/Q; derive totals; reset controlled DER Q=0",("pre-PF state is always uncontrolled reactive power at the controlled DERs",),10.2,7.1)
    pre=Rg("pre",170,326,700,46); draw_node(out,pre,COL["blue"],"[2] Pre-PF -> detect_violations(v_min, v_max)",("runpp exceptions become a non-converged pre-state",),10.4,7.1); out += [direct(loop,"bottom",state,"top"),direct(state,"bottom",pre,"top")]
    dpre=Dg("dpre",cx,414,210,42); draw_decision(out,dpre,"pre-PF converged?",ts=10.2); out.append(direct(pre,"bottom",dpre,"top"))
    pref=Rg("pref",35,391,230,46); draw_node(out,pref,COL["red"],"Pre-PF failure",("failed record -> publish -> append -> continue",),9.0,6.8,9); out.append(direct(dpre,"left",pref,"right","edge-hil")); branch_label(out,dpre.left-35,dpre.cy-7,"no",COL["hil"],size=8.2)
    dgate=Dg("dgate",cx,476,250,44); draw_decision(out,dgate,"gate enabled AND pre-state clean?",ts=9.8); out.append(direct(dpre,"bottom",dgate,"top")); branch_label(out,dpre.cx+12,dpre.bottom+13,"yes",anchor="start",size=8.2)
    clean=Rg("clean",35,451,280,50); draw_node(out,clean,COL["neutral"],"Clean gate",("reuse pre-PF; Q=0; build settled record",),9.0,6.8,9); out.append(direct(dgate,"left",clean,"right","edge-gate")); branch_label(out,dgate.left-35,dgate.cy-7,"yes",COL["gate"],size=8.2)
    call=Rg("call",360,526,320,54); draw_node(out,call,COL["green"],"[3] controller_fn(vm_pre, p_installed)",("software plugin or hardware wrapper; validate finite Q vector",),9.8,7.0); out.append(direct(dgate,"bottom",call,"top")); branch_label(out,dgate.cx+12,dgate.bottom+13,"no",anchor="start",size=8.2)
    cerr=Rg("cerr",740,527,265,52); draw_node(out,cerr,COL["red"],"Controller / serial exception",("propagates; benchmark marks this scenario failed",),8.9,6.7,9); out.append(direct(call,"right",cerr,"left","edge-hil")); branch_label(out,call.right+38,call.cy-7,"exception",COL["hil"],size=7.8)
    dclamp=Dg("dclamp",cx,626,205,40); draw_decision(out,dclamp,"clamp Q to net limits?",ts=9.7); out.append(direct(call,"bottom",dclamp,"top"))
    clamp=Rg("clamp",760,604,245,44); draw_node(out,clamp,COL["green"],"Apply Q safety clamp",("explicit Q limits + inverter S-circle",),8.8,6.6,9); out.append(direct(dclamp,"right",clamp,"left")); branch_label(out,dclamp.right+33,dclamp.cy-7,"yes",size=8.0)
    qm=C(cx,676,9); out.append(circle_svg(qm,COL["neutral"])); out.append(direct(dclamp,"bottom",qm,"top")); branch_label(out,dclamp.cx+12,dclamp.bottom+12,"no",anchor="start",size=8.0); out.append(path([clamp.a("bottom"),(clamp.cx,qm.cy),(qm.right+18,qm.cy),qm.a("right",TARGET_GAP)],"edge-dark"))
    applyq=Rg("applyq",360,700,320,44); draw_node(out,applyq,COL["blue"],"[4] write Q; P remains p_target",("no coordination, PT1/ramp or curtailment layer",),9.4,6.8); out.append(direct(qm,"bottom",applyq,"top"))
    post=Rg("post",360,760,320,44); draw_node(out,post,COL["blue"],"[5] Post-PF -> detect_violations",("post-PF exception is recorded, not propagated",),9.3,6.8); out.append(direct(applyq,"bottom",post,"top"))
    dpost=Dg("dpost",cx,846,205,40); draw_decision(out,dpost,"post-PF converged?",ts=9.7); out.append(direct(post,"bottom",dpost,"top"))

        # ------------------------------------------------------------------
    # Post-PF result branches
    # ------------------------------------------------------------------

    # Narrow enough to leave a dedicated clean-gate drop channel.
    bad=Rg("bad",95,824,180,44)
    good=Rg("good",740,821,265,50)

    draw_node(
        out,bad,COL["red"],
        "Post-PF failed record",
        ("converged=False",),
        8.8,6.5,9
    )

    draw_node(
        out,good,COL["purple"],
        "Settled controlled record",
        ("Q/P + losses/import; no curtailment",),
        8.8,6.5,9
    )

    out.append(
        direct(dpost,"left",bad,"right","edge-hil")
    )
    branch_label(
        out,dpost.left-34,dpost.cy-7,
        "no",COL["hil"],size=8.0
    )

    out.append(
        direct(dpost,"right",good,"left")
    )
    branch_label(
        out,dpost.right+34,dpost.cy-7,
        "yes",size=8.0
    )

    # ------------------------------------------------------------------
    # Staged record merge
    # ------------------------------------------------------------------

    clean_exit_x = clean.right - 10       # 305

    left_rm = C(clean_exit_x,884,9)
    rm      = C(cx,884,9)

    out.append(circle_svg(left_rm,COL["neutral"]))
    out.append(circle_svg(rm,COL["neutral"]))

    # Clean gate: perfectly straight downward.
    out.append(
        path(
            [
                (clean_exit_x,clean.bottom),
                left_rm.a("top",TARGET_GAP),
            ],
            "edge-gate"
        )
    )

    # Post-PF failure: left side of first merge.
    out.append(
        path(
            [
                bad.a("bottom"),
                (bad.cx,left_rm.cy),
                left_rm.a("left",TARGET_GAP),
            ],
            "edge-hil"
        )
    )

    # First merge -> final merge.
    out.append(
        direct(
            left_rm,"right",
            rm,"left",
            "edge-dark"
        )
    )

    # Successful controlled result -> right side of final merge.
    out.append(
        path(
            [
                good.a("bottom"),
                (good.cx,rm.cy),
                rm.a("right",TARGET_GAP),
            ],
            "edge-dark"
        )
    )

    # ------------------------------------------------------------------
    # Common record finalisation
    # ------------------------------------------------------------------

    finish=Rg("finish",220,906,600,44)
    draw_node(
        out,finish,COL["purple"],
        "[F] populate record metrics -> optional on_timestep -> append",
        (
            "timing, p_target, DER generation and load fields "
            "are added before publishing",
        ),
        9.6,6.9
    )

    out.append(
        direct(rm,"bottom",finish,"top")
    )

    # ------------------------------------------------------------------
    # Periodic progress
    # ------------------------------------------------------------------

    d96=Dg("d96",cx,981,190,36)
    draw_decision(
        out,d96,
        "t % 96 == 0?",
        ts=9.3
    )

    out.append(
        direct(finish,"bottom",d96,"top")
    )

    prog=Rg("prog",760,962,245,38)
    draw_node(
        out,prog,COL["purple"],
        "Periodic progress",
        ("optional partial CSV + daily log",),
        8.4,6.3,9
    )

    out.append(
        direct(d96,"right",prog,"left","edge-gate")
    )
    branch_label(
        out,d96.right+30,d96.cy-6,
        "yes",COL["gate"],size=7.8
    )

    pm=C(cx,1026,8)
    out.append(circle_svg(pm,COL["neutral"]))

    out.append(
        direct(d96,"bottom",pm,"top")
    )
    branch_label(
        out,d96.cx+11,d96.bottom+11,
        "no",anchor="start",size=7.8
    )

    out.append(
        path(
            [
                prog.a("bottom"),
                (prog.cx,pm.cy),
                (pm.right+18,pm.cy),
                pm.a("right",TARGET_GAP),
            ],
            "edge-gate"
        )
    )

    # Pre-PF failure bypass.
    out.append(
        path(
            [
                pref.a("left"),
                (fail_x,pref.cy),
                (fail_x,pm.cy),
                (pm.left-18,pm.cy),
                pm.a("left",TARGET_GAP),
            ],
            "edge-hil"
        )
    )

    # ------------------------------------------------------------------
    # Next timestep / finish
    # ------------------------------------------------------------------

    dmore=Dg("dmore",520,1058,180,26)
    draw_decision(
        out,dmore,
        "more timesteps?",
        ts=9.0
    )

    out.append(
        direct(pm,"bottom",dmore,"top")
    )

    done=Rg("done",790,1044,215,28)
    draw_node(
        out,done,COL["purple"],
        "Build ScenarioResult + return",
        ("optional on_scenario_end",),
        8.3,6.1,8
    )

    out.append(
        direct(dmore,"right",done,"left")
    )
    branch_label(
        out,dmore.right+28,dmore.cy-5,
        "no",size=7.6
    )

    out.append(
        path(
            [
                dmore.a("left"),
                (loop_x,dmore.cy),
                (loop_x,loop.cy),
                loop.a("left",TARGET_GAP),
            ],
            "edge-loop"
        )
    )

    branch_label(
        out,
        dmore.left-20,
        dmore.cy+11,
        "yes",
        COL["loop"],
        "end",
        7.2
    )
    
    px,pw=1080,800; panels=[R(px,24,pw,260),R(px,306,pw,342),R(px,670,pw,368)]
    for p in panels: out.append(rect_svg(p,COL["white"],16,COL["detail_border"],1.2))
    out.append(label(px+24,57,"Scenario initialization and resume",COL["text"],17,700,"start"))
    lines=[
        "The runner adapts profile tables and requires at least one profiled DER before entering the custom-control path.",
        "A dry-run VoltVarController is created only to reproduce built-in DER ordering, bus mapping, installed-P resolution and Q limits.",
        "No serial interface is opened by that metadata controller. Hardware serial ownership remains in the controller_fn wrapper.",
        "Existing pandapower controllers/results are cleared so this scenario starts from an isolated simulation state.",
        "When checkpointing is available, prior TimestepRecords are recovered and start_t advances to the first missing timestep.",
        "A complete checkpoint is converted directly into ScenarioResult and returned without repeating any power-flow timestep.",
    ]
    for i,t in enumerate(lines): out.append(label(px+24,91+31*i,t,COL["text"],11.1,600,"start"))
    out.append(label(px+24,339,"Per-timestep control sequence",COL["text"],17,700,"start"))
    lines=[
        "Each timestep writes the current active-power/load profiles and enforces Q=0 before the uncontrolled pre-power-flow snapshot.",
        "If the pre-PF fails, a non-converged record is published/appended immediately and the controller is not evaluated.",
        "With gate_clean_timesteps enabled, a clean pre-state is already the settled state, so Q remains zero and no second PF is run.",
        "Otherwise controller_fn receives pre-control DER-bus voltages plus installed/rated active power in the same DER order.",
        "The return vector is validated before optional clamping to explicit Q limits and the inverter apparent-power capability circle.",
        "The accepted Q vector is written directly. Active power remains p_target because this custom path has no dynamics/ramp layer.",
        "A second PF evaluates the controlled state. Its non-convergence is recorded locally so later timesteps can still run.",
        "Residual voltage/thermal violations remain in the record; the custom path does not perform active-power curtailment.",
    ]
    for i,t in enumerate(lines): out.append(label(px+24,374+32*i,t,COL["text"],11.0,600,"start"))
    out.append(label(px+24,703,"Publishing, progress and hardware execution",COL["text"],17,700,"start"))
    lines=[
        "Normal, clean-gate and post-PF-failure records receive timing/profile metrics before optional on_timestep publishing and append.",
        "Every 96th normal/clean timestep may rewrite the partial benchmark CSV and emits the daily custom-controller progress log.",
        "The pre-PF-failure continue bypasses that periodic branch and goes directly to the next timestep.",
        "When controller_fn is software, the imported Python algorithm computes Q locally from vm_pu and installed P.",
        "When controller_fn is HardwareControllerFn, its first call opens/configures Arduino and each call exchanges V:/Q: setpoints.",
        "Controller-function errors, output-validation errors and serial errors propagate out of this scenario; benchmark_runner isolates the failure.",
        "After the final timestep, all records are aggregated into ScenarioResult and on_scenario_end is emitted when a publisher is attached.",
    ]
    for i,t in enumerate(lines): out.append(label(px+24,739+39*i,t,COL["text"],11.0,600,"start"))
    for n,g in rects.items(): audit_rect_bounds(n,g,W,H,0)
    for n,g in diamonds.items(): audit_diamond_bounds(n,g,W,H,0)
    write("flow_plugin_loop_presentation_v4",W,H,"Custom controller plugin detailed loop - presentation","\n".join(out))


def main():
    build_registration_ieee(); build_registration_presentation(); build_loop_ieee(); build_loop_presentation()
    print(f"Wrote plugin flowcharts to {OUT}")

if __name__ == "__main__": main()
