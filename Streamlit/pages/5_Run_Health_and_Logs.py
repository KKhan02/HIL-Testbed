"""
Run health & logs
=================
Two views of "did anything go wrong":

  1. Run health — structured, from comparison.json (status, convergence,
     violations, failed-scenario tracebacks) plus hc.json errors. Reliable
     and producer-agnostic; no log file needed.
  2. Console — tails the run's session log (runs/<id>/session.log for the
     CLI, outputs/logs/session_*.log for the script), with level filter,
     search, error/warning highlighting, and optional auto-refresh.

Streamlit's *own* page errors show in the browser (red traceback) and in the
terminal you launched it from. To see them here too, redirect Streamlit's
output to a file and pick it below, e.g.:
    streamlit run Home.py --server.headless true > streamlit.log 2>&1
"""
import re
import sys
import time
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import _data

st.set_page_config(page_title="Run Health & Logs", layout="wide")
run_dir = _data.sidebar_run_selector()
st.title("🩺 Run Health & Logs")

if run_dir is None:
    st.info("Pick a run in the sidebar.")
    st.stop()

project_root = Path(st.session_state.get("project_root", "."))

# ===========================================================================
# 1. RUN HEALTH (structured)
# ===========================================================================
st.subheader("Run health")

records = _data.load_comparison(run_dir)
hc = _data.load_hc(run_dir)

if not records:
    # Fall back to per-scenario summaries while the run is still in progress.
    rows = []
    for sid in _data.list_scenarios(run_dir):
        sc = _data.load_scenario(run_dir, sid)
        summ = (sc or {}).get("summary", {}) or {}
        rows.append({
            "scenario_id": sid,
            "status": summ.get("status", "ok"),
            "n_converged": summ.get("n_converged"),
            "n_timesteps": summ.get("n_timesteps"),
            "n_violation_steps": summ.get("n_violation_steps"),
        })
    records = rows or None

if not records:
    st.info("No finished scenarios yet — health appears as scenarios complete.")
else:
    df = pd.DataFrame(records)

    def _int(x):
        try:
            return int(x)
        except (TypeError, ValueError):
            return None

    failed, diverged = [], []
    for r in records:
        status = str(r.get("status", "ok"))
        nc, nt = _int(r.get("n_converged")), _int(r.get("n_timesteps"))
        if status == "failed":
            failed.append(r)
        elif nc is not None and nt is not None and nc < nt:
            diverged.append((r, nt - nc))

    total_viol = sum(_int(r.get("n_violation_steps")) or 0 for r in records)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Scenarios", len(records))
    m2.metric("Failed", len(failed), delta=None)
    m3.metric("With divergence", len(diverged))
    m4.metric("Total violation steps", f"{total_viol:,}")

    for r in failed:
        msg = r.get("error_message") or "(no message)"
        st.error(f"**{_data.scenario_label(r.get('scenario_id',''))}** failed — {msg}")
    for r, d in diverged:
        st.warning(
            f"**{_data.scenario_label(r.get('scenario_id',''))}** — "
            f"{d:,} of {r.get('n_timesteps')} steps did not converge."
        )
    if hc and isinstance(hc, dict) and "error" in hc:
        st.error(f"Hosting-capacity analysis failed — {hc['error']}")

    if not failed and not diverged and not (hc and "error" in (hc or {})):
        st.success("No failed scenarios, no divergence, no HC errors.")

    cols = [c for c in ["scenario_id", "status", "n_converged", "n_timesteps",
                        "n_violation_steps", "max_vm_pu", "min_vm_pu"] if c in df.columns]
    with st.expander("Per-scenario detail"):
        st.dataframe(df[cols] if cols else df, use_container_width=True)

# ===========================================================================
# 2. CONSOLE (log tail)
# ===========================================================================
st.subheader("Console")

logs = _data.discover_logs(project_root, run_dir)
options = [str(p) for p in logs]
picked = st.selectbox(
    "Log file", options + ["Other…"] if options else ["Other…"],
    help="CLI writes runs/<id>/session.log; the script writes outputs/logs/session_*.log.",
)
log_path = Path(st.text_input("Path", value="")) if picked == "Other…" else Path(picked)

if not log_path or not log_path.exists():
    st.info("No log file selected/found yet. It appears once a run starts writing.")
    st.stop()

c1, c2, c3, c4 = st.columns([1.2, 1.2, 1, 1])
with c1:
    level = st.radio("Show", ["All", "Warnings & errors", "Errors only"], index=0)
with c2:
    query = st.text_input("Search", value="")
with c3:
    n_tail = st.slider("Lines", 50, 1000, 200, step=50)
with c4:
    follow = st.toggle("Auto-refresh", value=False)
    refresh_s = st.slider("s", 1.0, 5.0, 2.0, step=0.5, disabled=not follow)

lines, size = _data.tail_lines(log_path, n=n_tail)

_ERR = re.compile(r"(ERROR|CRITICAL|FATAL|Traceback \(most recent call last\))")
_WARN = re.compile(r"(WARNING|\bWARN\b)")

def level_of(line: str) -> str:
    if _ERR.search(line):
        return "error"
    if _WARN.search(line):
        return "warning"
    return "info"

classed = [(ln, level_of(ln)) for ln in lines]
n_err = sum(1 for _, lv in classed if lv == "error")
n_warn = sum(1 for _, lv in classed if lv == "warning")

mc1, mc2, mc3 = st.columns(3)
mc1.metric("Errors (in view)", n_err)
mc2.metric("Warnings (in view)", n_warn)
mc3.metric("File size", f"{size/1024:.0f} KB")

# Always surface the latest problems, highlighted, regardless of filter.
problems = [(ln, lv) for ln, lv in classed if lv in ("error", "warning")]
if problems:
    with st.expander(f"⚠ Latest errors & warnings ({len(problems)})", expanded=bool(n_err)):
        for ln, lv in problems[-30:]:
            (st.error if lv == "error" else st.warning)(ln)

# Filtered raw tail.
if level == "Errors only":
    shown = [ln for ln, lv in classed if lv == "error"]
elif level == "Warnings & errors":
    shown = [ln for ln, lv in classed if lv in ("error", "warning")]
else:
    shown = lines
if query.strip():
    q = query.lower()
    shown = [ln for ln in shown if q in ln.lower()]

st.caption(f"`{log_path}` — showing {len(shown)} of last {len(lines)} lines")
st.code("\n".join(shown) if shown else "(no matching lines)", language="log")

if follow:
    time.sleep(refresh_s)
    st.rerun()
