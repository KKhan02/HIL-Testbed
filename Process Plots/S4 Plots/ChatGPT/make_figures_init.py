from pathlib import Path
import cairosvg, html

OUT = Path(f'D:\\My Files\\Personal Projects\\HIL-Testbed\\Process Plots\\S4 Plots\\ChatGPT\\s4_top_level_reverted')
COL = {
    'blue':'#0C447C','blue_lane':'#3D8FD9','green':'#0F6E56','green_lane':'#1F9D78',
    'purple':'#3C3489','red':'#993C1D','red_lane':'#D95B32','dry':'#3B6D11',
    'decision':'#854F0B','neutral':'#5F5E5A','white':'#FFFFFF','text':'#111111',
    'edge':'#263238','hil_edge':'#D85A30','dry_edge':'#6B9E26'
}

def esc(s): return html.escape(s)
def rect(x,y,w,h,fill,rx=15,stroke='none',sw=0):
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>'
def diamond(cx,cy,w,h,fill):
    pts=f'{cx},{cy-h/2} {cx+w/2},{cy} {cx},{cy+h/2} {cx-w/2},{cy}'
    return f'<polygon points="{pts}" fill="{fill}"/>'
def label(x,y,t,fill,size=14,weight=700,anchor='middle'):
    return f'<text x="{x}" y="{y}" text-anchor="{anchor}" fill="{fill}" font-size="{size}" font-weight="{weight}">{esc(t)}</text>'
def text_lines(x,y,title,subs=(),title_size=18,sub_size=14,line_gap=21,fill='white'):
    out=[f'<text x="{x}" y="{y}" text-anchor="middle" class="node-title" font-size="{title_size}" fill="{fill}">{esc(title)}</text>']
    yy=y+line_gap
    for s in subs:
        out.append(f'<text x="{x}" y="{yy}" text-anchor="middle" class="node-sub" font-size="{sub_size}" fill="{fill}" opacity="0.86">{esc(s)}</text>')
        yy += line_gap-2
    return '\n'.join(out)

def path(d,cls='edge',marker=True):
    marker_map={'edge':'arrowWhite','edge-dark':'arrowDark','edge-hil':'arrowHil','edge-dry':'arrowDry'}
    mark=f' marker-end="url(#{marker_map.get(cls,"arrowDark")})"' if marker else ''
    return f'<path d="{d}" class="{cls}"{mark}/>'

def header(w,h,title):
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">
<title>{esc(title)}</title>
<defs>
  <marker id="arrowWhite" markerWidth="6" markerHeight="6" refX="8" refY="4.5" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L9,4.5 L0,9 z" fill="#FFFFFF"/></marker>
  <marker id="arrowDark" markerWidth="6" markerHeight="6" refX="8" refY="4.5" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L9,4.5 L0,9 z" fill="#263238"/></marker>
  <marker id="arrowHil" markerWidth="6" markerHeight="6" refX="8" refY="4.5" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L9,4.5 L0,9 z" fill="#D85A30"/></marker>
  <marker id="arrowDry" markerWidth="6" markerHeight="6" refX="8" refY="4.5" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L9,4.5 L0,9 z" fill="#6B9E26"/></marker>
</defs>
<style>
text {{ font-family: Helvetica, Arial, sans-serif; }}
.node-title {{ font-weight:700; }} .node-sub {{ font-weight:400; }}
.edge {{ fill:none; stroke:#FFFFFF; stroke-width:3; stroke-linejoin:round; stroke-linecap:round; }}
.edge-dark {{ fill:none; stroke:#263238; stroke-width:3; stroke-linejoin:round; stroke-linecap:round; }}
.edge-hil {{ fill:none; stroke:#D85A30; stroke-width:3.2; stroke-linejoin:round; stroke-linecap:round; }}
.edge-dry {{ fill:none; stroke:#6B9E26; stroke-width:3.2; stroke-linejoin:round; stroke-linecap:round; }}
</style>'''

def write(name,w,h,title,body):
    svg=header(w,h,title)+'\n'+body+'\n</svg>'
    (OUT/f'{name}.svg').write_text(svg,encoding='utf-8')
    cairosvg.svg2pdf(bytestring=svg.encode(),write_to=str(OUT/f'{name}.pdf'))
    cairosvg.svg2png(bytestring=svg.encode(),write_to=str(OUT/f'{name}.png'),output_width=w*2,output_height=h*2)

# IEEE compact single-column
w,h=780,1040
b=[]
b.append(rect(42,32,696,974,'#F8FAFC',20,'#D9E1E8',2))
b.append(label(
    390,70,
    'Scenario 4 - Top-Level Volt-Var HIL Flow',
    COL['text'],22,700
))

# Move everything below the title 20 px upward
b.append('<g transform="translate(0,-20)">')

b.append(rect(180,124,420,74,COL['blue'],14))
b.append(text_lines(
    390,154,
    'run_scenario_4()',
    ('network, profiles, mode',)
))

b.append(path('M390 198 L390 224','edge-dark'))

b.append(rect(180,226,420,88,COL['blue'],14))
b.append(text_lines(
    390,258,
    'Prepare simulation',
    ('adapt profiles  |  publisher start',
     'reset stale controllers / results')
))

b.append(path('M390 314 L390 338','edge-dark'))

b.append(rect(180,340,420,88,COL['green'],14))
b.append(text_lines(
    390,372,
    'Build control stack',
    ('Volt-Var controller  |  coordinator',
     'DER dynamics')
))

b.append(path('M390 428 L390 450','edge-dark'))

b.append(diamond(390,494,150,88,COL['decision']))
b.append(text_lines(
    390,494,
    'dry_run?',
    (),
    title_size=17
))

# branches
b.append(path(
    'M315 494 L225 494 L225 565',
    'edge-dry'
))
b.append(label(
    255,482,
    'yes',
    COL['dry_edge'],13,700
))

b.append(rect(110,567,230,80,COL['dry'],14))
b.append(text_lines(
    225,598,
    'Dry-run path',
    ('configure() no-op',
     'local Q(V)'),
    title_size=16,
    sub_size=13
))

b.append(path(
    'M465 494 L555 494 L555 545',
    'edge-hil'
))
b.append(label(
    525,482,
    'no - HIL',
    COL['hil_edge'],13,700
))

b.append(rect(440,547,230,80,COL['red'],14))
b.append(text_lines(
    555,578,
    'HIL path',
    ('open serial link',
     'configure handshake'),
    title_size=16,
    sub_size=13
))

# merge
b.append(diamond(390,665,42,32,COL['neutral']))

b.append(path(
    'M225 647 L225 665 L369 665',
    'edge-dry'
))

b.append(path(
    'M555 627 L555 665 L411 665',
    'edge-hil'
))

b.append(path(
    'M390 681 L390 703',
    'edge-dark'
))

b.append(rect(180,705,420,72,COL['blue'],14))
b.append(text_lines(
    390,735,
    'Initialize state',
    ('DERDynamics.reset()',)
))

b.append(path(
    'M390 777 L390 801',
    'edge-dark'
))

b.append(rect(180,803,420,88,COL['neutral'],14))
b.append(text_lines(
    390,835,
    'Per-timestep loop',
    ('control + power flow + curtailment',
     'record / publish   |   see Diagram 2')
))

b.append(path(
    'M390 891 L390 915',
    'edge-dark'
))

b.append(rect(180,917,420,76,COL['purple'],14))
b.append(text_lines(
    390,947,
    'Finalize and return',
    ('HIL: close() -> TX END   |   dry-run: no close',
     'from_records() -> on_scenario_end -> ScenarioResult'),
    title_size=17,
    sub_size=13
))

b.append('</g>')

write(
    'flow_s4_top_ieee_reverted',
    w,h,
    'Scenario 4 top-level flow - IEEE',
    '\n'.join(b)
)

# Presentation, three lanes
# Logic checked against scenario_4_volt_var.py and the HIL handshake PUML.
# The IEEE compact single-column section above is intentionally unchanged.
w,h=1280,1270
b=[]
b.append(rect(32,64,420,1200,COL['blue_lane'],24)); b.append(rect(464,64,382,1200,COL['green_lane'],24)); b.append(rect(858,64,390,1200,COL['red_lane'],24))
b.append(label(242,42,'Master - scenario_4',COL['text'],20,700)); b.append(label(655,42,'Sub-modules + publisher',COL['text'],20,700)); b.append(label(1053,42,'Hardware (HIL)',COL['text'],20,700))

# top flow
b.append(rect(92,96,300,68,COL['blue'],14)); b.append(text_lines(242,125,'run_scenario_4()',('network, profiles, mode',)))
b.append(path('M242 164 L242 188','edge'))
b.append(rect(92,190,300,72,COL['blue'],14)); b.append(text_lines(242,220,'adapt_profiles(net)',('DER P, load P/Q, times',)))
b.append(path('M392 226 L510 226','edge'))
b.append(rect(510,192,290,68,COL['purple'],14)); b.append(text_lines(655,221,'on_scenario_start',('publisher / checkpoint state',),title_size=17,sub_size=13))
b.append(path('M655 260 L655 282 L242 282 L242 304','edge'))
b.append(rect(92,306,300,66,COL['blue'],14)); b.append(text_lines(242,334,'Reset network state',('drop controllers / reset results',),title_size=17,sub_size=13))

# hardware dispatch occurs before _run_loop()
b.append(path('M242 372 L242 400','edge'))
b.append(diamond(241,440,132,76,COL['decision'])); b.append(text_lines(241,445,'dry_run?',(),title_size=16))

# dry-run enters _run_loop directly
b.append(path('M242 478 L242 520','edge-dry')); b.append(label(184,490,'yes - dry-run','#E8F6DA',12,700))

# HIL opens the serial interface first, then enters _run_loop
b.append(path('M308 440 L900 440','edge')); b.append(label(620,427,'no - HIL','#FFFFFF',12,700))
b.append(rect(900,410,306,66,COL['red'],14)); b.append(text_lines(1053,438,'open() @115200',('reset / flush serial buffer',),title_size=16,sub_size=13))
b.append(path('M1053 476 L1053 536 L258 536','edge'))

# both modes now enter the same _run_loop() body
b.append(f'<circle cx="242" cy="536" r="15" fill="{COL["neutral"]}"/>')
b.append(path('M242 552 L242 578','edge'))
b.append(rect(92,580,300,72,COL['blue'],14)); b.append(text_lines(242,609,'VoltVarController(...)',('resolve DER idx / buses / MW',),title_size=17,sub_size=13))

# configure() is called in both modes:
# dry-run -> no-op/local Q(V); HIL -> INIT / CFG / P handshake
b.append(path('M242 652 L242 703','edge-dry')); b.append(label(194,681,'dry-run','#E8F6DA',12,700))
b.append(rect(92,700,300,68,COL['dry'],14)); b.append(text_lines(242,729,'configure() no-op',('local Q(V)',),title_size=16,sub_size=13))

b.append(path('M392 616 L1053 616 L1053 698','edge')); b.append(label(714,603,'HIL','#FFFFFF',12,700))
b.append(rect(900,700,306,72,COL['red'],14)); b.append(text_lines(1053,729,'configure() handshake',('INIT -> CFG -> P   |   ACK each',),title_size=16,sub_size=13))

# merge after configure()
b.append(f'<circle cx="242" cy="809" r="15" fill="{COL["neutral"]}"/>')
b.append(path('M242 768 L242 793','edge-dry'))
b.append(path('M1053 772 L1053 809 L258 809','edge'))

# construct remaining control objects only after configure()
b.append(path('M242 824 L242 866 L510 866','edge'))
b.append(rect(510,830,290,72,COL['green'],14)); b.append(text_lines(655,859,'Coordinator + DER dynamics',('SensitivityCoordinator / DERDynamics',),title_size=16,sub_size=12))

# initialize shared state
b.append(path('M655 902 L655 943 L392 943','edge'))
b.append(rect(92,910,300,66,COL['blue'],14)); b.append(text_lines(242,938,'dynamics.reset()',('seed q_prev / p_prev',),title_size=17,sub_size=13))
b.append(path('M242 976 L242 1002','edge'))

# shared per-timestep loop
b.append(rect(92,1004,300,72,COL['neutral'],14))
b.append(text_lines(
    242,1030,
    'Per-timestep loop',
    ('control / PF / curtailment',
     'see Diagram 2'),
    title_size=17,
    sub_size=13
))


# ============================================================
# HIL serial exchange inside each timestep
# ============================================================

b.append(rect(900,1000,306,72,COL['red'],14))
b.append(text_lines(
    1053,1016,
    'Serial exchange',
    ('V:<vm> ->   |   <- Q:<q>',
     'each HIL timestep'),
    title_size=16,
    sub_size=12
))

b.append(path(
    'M392 1028 L900 1028',
    'edge'
))
b.append(label(
    646,1017,
    'V:<vm>',
    '#FFFFFF',12,700
))

b.append(path(
    'M900 1054 L392 1054',
    'edge'
))
b.append(label(
    646,1070,
    'Q:<q>',
    '#FFFFFF',12,700
))


# ============================================================
# HIL teardown
# Serial exchange -> close()
# ============================================================

b.append(rect(900,1099,306,64,COL['red'],14))
b.append(text_lines(
    1053,1120,
    'close() -> TX END',
    ('firmware clears configured / port closes',),
    title_size=16,
    sub_size=12
))

# Connect bottom-center of Serial exchange to top-center of close()
# Serial exchange bottom = 1000 + 72 = 1072
# close() top = 1099
b.append(path(
    'M1053 1072 L1053 1097',
    'edge'
))


# ============================================================
# Scenario result
# Moved 5 px UP: 1094 -> 1089
# ============================================================

b.append(rect(510,1089,290,70,COL['green'],14))
b.append(text_lines(
    655,1114,
    'ScenarioResult.from_records()',
    ('reduce timestep records',),
    title_size=15,
    sub_size=12
))


# Dry-run:
# Per-timestep loop -> left-middle of ScenarioResult.from_records()
#
# ScenarioResult center:
# y = 1089 + 70/2 = 1124
b.append(path(
    'M242 1076 L242 1124 L510 1124',
    'edge'
))
b.append(label(
    350,1114,
    'dry-run',
    '#FFFFFF',12,700
))


# HIL:
# close() -> right-middle of ScenarioResult.from_records()
#
# close() center y = 1099 + 64/2 = 1131
# ScenarioResult center y = 1124
#
# Small vertical bend keeps both block connections clean.
b.append(path(
    'M900 1131 L800 1131',
    'edge'
))


# ============================================================
# Publisher end
# Moved 5 px DOWN: 1178 -> 1183
# ============================================================

b.append(rect(510,1183,290,62,COL['purple'],14))
b.append(text_lines(
    655,1204,
    'on_scenario_end',
    ('flush summary / final frame',),
    title_size=17,
    sub_size=13
))


# ScenarioResult -> on_scenario_end
#
# ScenarioResult bottom = 1089 + 70 = 1159
# on_scenario_end top = 1183
#
# End at 1181 so the arrowhead remains visible before the block.
b.append(path(
    'M655 1159 L655 1181',
    'edge'
))


# ============================================================
# Return block
# Moved down with on_scenario_end to preserve alignment
# ============================================================

b.append(rect(92,1183,300,62,COL['blue'],14))
b.append(text_lines(
    242,1220,
    'return ScenarioResult',
    (),
    title_size=17
))


# on_scenario_end -> return ScenarioResult
#
# Both rectangles have:
# y = 1183
# height = 62
# center y = 1214
b.append(path(
    'M510 1214 L392 1214',
    'edge'
))


write(
    'flow_s4_top_presentation_reverted',
    w,h,
    'Scenario 4 top-level flow - presentation',
    '\n'.join(b)
)

(OUT/'README.txt').write_text(
"Scenario 4 top-level figures, reverted to the initial design direction.\n\n"
"The IEEE figure is compact and single-column oriented.\n"
"The presentation figure uses three high-contrast conceptual lanes.\n\n"
"Logic: dry-run and HIL split during initialization, merge before dynamics.reset(), share the timestep loop, then split after the loop so only HIL executes close() -> TX END. Both paths then converge at ScenarioResult.from_records(), followed by on_scenario_end and return ScenarioResult.\n",
encoding='utf-8')