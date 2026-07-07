import streamlit as st
import redis
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pickle
import shap
import math
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════════════════
# NEW LIGHT-THEME UI — built from the reviewed UI/UX mockup
# (MuleDetection_UIUX_Mockup.html). This file replaces dashboard.py's
# single-page layout with a multipage shell using session_state-based
# navigation (not st.navigation/st.Page — requirements.txt does not pin
# a Streamlit version, and st.navigation needs 1.36+; session_state
# navigation works on any version, so this avoids an unverified
# assumption).
#
# STATUS: Dashboard page is fully wired to real Redis/model data,
# identical underlying logic to the existing dashboard.py. The other
# six pages (Live Monitoring, Flagged Accounts, Risk Analysis, Hop Map,
# Reports, Settings) are placeholder shells matching the approved
# mockup's visual language, clearly labeled as not-yet-implemented --
# built one at a time per instruction, not all at once.
# ═══════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Mule Account Detection — CyberPeace Foundation",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Design tokens (from the approved mockup) ──────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:wght@500;600;700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

    :root {
        --paper:#F5F6F4; --paper-raised:#FFFFFF; --ink:#14202B; --ink-soft:#4B5A66;
        --line:#D8DEE2; --line-strong:#B9C2C9;
        --navy:#1B3A63; --navy-deep:#0F2540;
        --seal:#9C6B1D; --danger:#8C2F2F; --warn:#9C6B1D; --safe:#3F6B4A;
    }
    html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
    html, body, .stApp, [data-testid="stAppViewContainer"],
    [data-testid="stMain"], .main, .block-container {
        background-color: var(--paper) !important;
        color: var(--ink) !important;
    }
    [data-testid="stAppViewContainer"] * , .block-container * {
        color: var(--ink);
    }
    .main { background-color: var(--paper); }
    .block-container { padding: 1rem 1.5rem 2rem 1.5rem; max-width: 100%; }

    h1, h2, h3, .page-title { font-family: 'Source Serif 4', serif !important; color: var(--ink); }

    /* ── Sidebar (ledger-navy, matches mockup) ── */
    section[data-testid="stSidebar"] { background: var(--navy-deep); }
    section[data-testid="stSidebar"] * { color: #DCE4EA !important; }
    section[data-testid="stSidebar"] .stButton button {
        width: 100%; text-align: left; background: transparent; border: none;
        color: #B9C6D0 !important; font-size: 0.85rem; padding: 0.5rem 0.7rem;
        border-radius: 2px; font-family: 'IBM Plex Sans', sans-serif;
    }
    section[data-testid="stSidebar"] .stButton button:hover { background: rgba(255,255,255,0.06); color: #fff !important; }
    section[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.12) !important; }

    /* ── Top status pill ── */
    .status-pill {
        display:inline-flex; align-items:center; gap:7px; font-size:0.78rem;
        padding:5px 11px; border:1px solid var(--line-strong); border-radius:2px; background:#F0FBF3;
    }
    .status-dot { width:7px; height:7px; border-radius:50%; }
    .status-dot.live { background: var(--safe); }
    .status-dot.warn { background: var(--warn); }
    .status-dot.stale { background: var(--danger); }

    /* ── KPI cards (ledger style, not floating) ── */
    .kpi { background:var(--paper-raised); border:1px solid var(--line); border-top:3px solid var(--navy);
           padding:0.9rem 1rem; border-radius:2px; text-align:left; }
    .kpi.danger { border-top-color: var(--danger); }
    .kpi.warn   { border-top-color: var(--warn); }
    .kpi.safe   { border-top-color: var(--safe); }
    .kpi .val   { font-family:'IBM Plex Mono', monospace; font-size:1.7rem; font-weight:600; color:var(--ink); }
    .kpi .lbl   { font-size:0.65rem; text-transform:uppercase; letter-spacing:0.08em; color:var(--ink-soft); margin-top:4px; }

    /* ── Panels ── */
    .panel-head { font-family:'Source Serif 4', serif; font-size:0.95rem; font-weight:600;
                  color:var(--ink); border-bottom:1px solid var(--line); padding-bottom:0.4rem; margin-bottom:0.7rem; }

    /* ── Ledger table ── */
    .ledger-table { width:100%; border-collapse:collapse; font-size:0.78rem; }
    .ledger-table th {
        text-align:left; font-size:0.62rem; text-transform:uppercase; letter-spacing:0.06em;
        color:var(--ink-soft); padding:0.5rem 0.7rem; border-bottom:2px solid var(--ink); font-weight:600;
    }
    .ledger-table td { padding:0.55rem 0.7rem; border-bottom:1px solid var(--line); font-family:'IBM Plex Mono', monospace; color:var(--ink); }
    .ledger-table tr:hover td { background:#FAFAF8; }

    /* ── Stamp badge (signature element, from mockup) ── */
    .stamp {
        display:inline-flex; align-items:center; justify-content:center; width:44px; height:44px;
        border:2px dashed currentColor; border-radius:50%; transform:rotate(-8deg);
        font-family:'IBM Plex Mono', monospace; font-size:0.5rem; font-weight:700;
        letter-spacing:0.03em; text-align:center; line-height:1.1;
    }
    .stamp.l3 { color: var(--danger); }
    .stamp.l2 { color: var(--warn); }
    .stamp.l1 { color: var(--safe); }
    .badge-pattern { background:#E7E9EC; color:var(--navy-deep); border:1px solid var(--line-strong);
                      font-size:0.62rem; padding:2px 6px; border-radius:2px; font-family:'IBM Plex Mono', monospace; }

    .placeholder-panel {
        border:1px dashed var(--line-strong); border-radius:2px; padding:3rem 2rem; text-align:center;
        color:var(--ink-soft); background:repeating-linear-gradient(45deg, #FAFAF8, #FAFAF8 10px, #F3F4F2 10px, #F3F4F2 20px);
    }
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    /* Header intentionally left untouched -- previous attempts to hide
       parts of it via guessed data-testid selectors were unconfirmed
       and may have been wrong for this Streamlit version. Top
       navigation buttons above no longer depend on the sidebar's
       native collapse/reopen control, so there's no reason to keep
       risking that control by touching the header at all. */
</style>
""", unsafe_allow_html=True)

# ── Cached resources (unchanged from dashboard.py) ────────────────────────────
@st.cache_resource
def get_redis():
    return redis.Redis(host='localhost', port=6379, decode_responses=True)

@st.cache_resource
def load_models():
    with open('xgb_hybrid_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler_hybrid.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('hybrid_model_config.json') as f:
        config = json.load(f)
    explainer = shap.TreeExplainer(model)
    return model, scaler, config, explainer

r = get_redis()
model, scaler, config, explainer = load_models()

L3_THRESHOLD, L2_THRESHOLD, L1_THRESHOLD = 0.70, 0.50, 0.30

# ── Data functions (identical logic to dashboard.py) ─────────────────────────
def get_alerts(n=1000):
    raw = r.lrange('alerts', 0, n - 1)
    alerts = []
    for item in raw:
        try:
            alerts.append(json.loads(item))
        except Exception:
            pass
    return alerts

def get_total_flagged():
    count = r.get("alerts_total_count")
    return int(count) if count else None

def get_stats(alerts):
    real_total = get_total_flagged()
    total = real_total if real_total else len(alerts)
    l3 = sum(1 for a in alerts if 'L3' in a.get('tier', ''))
    l2 = sum(1 for a in alerts if 'L2' in a.get('tier', '') and 'L3' not in a.get('tier', ''))
    l1 = sum(1 for a in alerts if 'L1' in a.get('tier', '') and 'L2' not in a.get('tier', '') and 'L3' not in a.get('tier', ''))
    # NOTE: r.keys() is O(N) and blocks Redis -- same known tradeoff as
    # dashboard.py. Not re-solved here; see Progress Report II Section 5.
    confirmed = len(r.keys("launder_status:*"))
    gnn = sum(1 for a in alerts if a.get('has_gnn_embed'))
    gnn_pct = int(gnn / len(alerts) * 100) if len(alerts) > 0 else 0
    return total, l3, l2, l1, confirmed, gnn_pct

def deduplicate_alerts(alerts):
    seen, counts = {}, {}
    for a in alerts:
        aid = a.get('account_id', '')
        counts[aid] = counts.get(aid, 0) + 1
        if aid not in seen or a.get('probability', 0) > seen[aid].get('probability', 0):
            seen[aid] = a
    return seen, counts

def get_cep_badge_html(aid):
    try:
        raw = r.lindex(f"cep_alerts:{aid}", 0)
        if not raw:
            return "—"
        pattern = json.loads(raw).get('pattern_type', '')
        return f'<span class="badge-pattern">{pattern}</span>' if pattern else "—"
    except Exception:
        return "—"

def get_system_status(alerts):
    if not alerts:
        return "NO DATA", "stale"
    try:
        latest_dt = datetime.fromisoformat(alerts[0].get('timestamp', '').replace('Z', ''))
        age_seconds = (datetime.now() - latest_dt).total_seconds()
    except Exception:
        return "UNKNOWN", "stale"
    if age_seconds < 30:
        return "LIVE", "live"
    elif age_seconds < 120:
        return "SLOWING", "warn"
    return "STALE", "stale"

def stamp_html(tier):
    if 'L3' in tier:
        return '<span class="stamp l3">L3<br>FREEZE</span>'
    if 'L2' in tier:
        return '<span class="stamp l2">L2<br>HOLD</span>'
    if 'L1' in tier:
        return '<span class="stamp l1">L1<br>REVIEW</span>'
    return '—'

# ── Sidebar navigation (session_state based -- version-safe) ─────────────────
PAGES = ["Dashboard", "Live Monitoring", "Flagged Accounts", "Risk Analysis",
         "Hop Map", "Reports", "Settings"]
NOT_YET_BUILT = set(PAGES) - {"Dashboard"}

if "page" not in st.session_state:
    st.session_state.page = "Dashboard"

with st.sidebar:
    st.markdown("### Mule Account Detection")
    st.caption("CyberPeace Research Wing")
    st.markdown("---")
    for pg in PAGES:
        if st.button(pg, key=f"nav_{pg}", use_container_width=True):
            st.session_state.page = pg
    st.markdown("---")
    st.caption("v0.2 — UI redesign in progress\nPages beyond Dashboard are placeholders")

page = st.session_state.page

# ── Top navigation (redundant with sidebar, always visible) ──────────────────
# This exists specifically so closing/losing the sidebar can never trap
# navigation again -- it does not depend on Streamlit's collapse/reopen
# control working correctly.
nav_cols = st.columns(len(PAGES))
for i, pg in enumerate(PAGES):
    with nav_cols[i]:
        if st.button(pg, key=f"topnav_{pg}", use_container_width=True,
                     type="primary" if page == pg else "secondary"):
            st.session_state.page = pg
            st.rerun()
st.markdown("<hr style='margin:0.5rem 0 1rem 0; border-color:var(--line);'>", unsafe_allow_html=True)

# ── Top bar ────────────────────────────────────────────────────────────────
alerts = get_alerts(1000)
status_label, status_cls = get_system_status(alerts)
top_l, top_r = st.columns([4, 1])
with top_l:
    st.markdown(f'<div class="page-title" style="font-size:1.4rem; font-weight:600;">{page}</div>', unsafe_allow_html=True)
with top_r:
    st.markdown(f'<div class="status-pill"><span class="status-dot {status_cls}"></span>{status_label}</div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# DASHBOARD PAGE — fully wired, real data
# ═══════════════════════════════════════════════════════════════════════════
if page == "Dashboard":
    total, l3, l2, l1, confirmed, gnn_pct = get_stats(alerts)
    deduped, counts = deduplicate_alerts(alerts)

    c1, c2, c3, c4, c5 = st.columns(5)
    def kpi(col, value, label, style="neutral"):
        col.markdown(f'<div class="kpi {style}"><div class="val">{value}</div><div class="lbl">{label}</div></div>', unsafe_allow_html=True)
    kpi(c1, total, "Total Flagged (All-Time)")
    kpi(c2, l3, "L3 Auto-Freeze", "danger")
    kpi(c3, l2, "L2 Soft Hold", "warn")
    kpi(c4, l1, "L1 Review", "safe")
    kpi(c5, confirmed, "Confirmed Laundering", "danger")

    st.markdown(
        "<div style='font-size:0.7rem; color:var(--ink-soft); margin-top:0.4rem;'>"
        f"L1/L2/L3 reflect the current alert queue (last {len(alerts)} entries); "
        "Total Flagged is an all-time counter — the two are not expected to sum.</div>",
        unsafe_allow_html=True
    )
    st.markdown("<br>", unsafe_allow_html=True)

    left, right = st.columns([3, 2])
    with left:
        st.markdown('<div class="panel-head">Alert Queue — Ranked by Score</div>', unsafe_allow_html=True)
        if not deduped:
            st.markdown('<div class="placeholder-panel">No alerts detected. Start the producer and consumer to begin real-time detection.</div>', unsafe_allow_html=True)
        else:
            sorted_accounts = sorted(deduped.items(), key=lambda x: x[1].get('probability', 0), reverse=True)
            rows = ""
            for aid, a in sorted_accounts[:25]:
                tier = a.get('tier', 'CLEAR')
                prob = a.get('probability', 0)
                cep = get_cep_badge_html(aid)
                conf = a.get('account_laundering', False) or a.get('is_laundering', 0) == 1
                conf_cell = '<span style="color:var(--danger); font-weight:700;">YES</span>' if conf else '—'
                rows += (f"<tr><td>{aid[:14]}</td><td style='text-align:right'>{prob:.4f}</td>"
                         f"<td>{stamp_html(tier)}</td><td>{cep}</td><td>{conf_cell}</td></tr>")
            st.markdown(
                f"""<table class="ledger-table">
                <tr><th>Account</th><th style="text-align:right">Score</th><th>Tier</th><th>Pattern</th><th>Confirmed</th></tr>
                {rows}</table>""", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="panel-head">Score Distribution</div>', unsafe_allow_html=True)
        if alerts:
            probs = [a.get('probability', 0) for a in alerts]
            fig, ax = plt.subplots(figsize=(5, 2.4))
            ax.hist(probs, bins=25, color='#1B3A63', edgecolor='#F5F6F4', linewidth=0.4)
            ax.axvline(x=L1_THRESHOLD, color='#3F6B4A', linestyle='--', linewidth=0.9, label=f'L1 ({L1_THRESHOLD})')
            ax.axvline(x=L2_THRESHOLD, color='#9C6B1D', linestyle='--', linewidth=0.9, label=f'L2 ({L2_THRESHOLD})')
            ax.axvline(x=L3_THRESHOLD, color='#8C2F2F', linestyle='--', linewidth=0.9, label=f'L3 ({L3_THRESHOLD})')
            ax.set_facecolor('#FAFAF8')
            fig.patch.set_facecolor('white')
            for spine in ax.spines.values():
                spine.set_color('#D8DEE2')
            ax.tick_params(labelsize=6, colors='#4B5A66')
            ax.legend(fontsize=6, framealpha=0.9)
            plt.tight_layout(pad=0.5)
            st.pyplot(fig)
            plt.close()
        else:
            st.markdown('<div class="placeholder-panel">No data yet.</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# PLACEHOLDER PAGES — visual shell only, matches approved mockup.
# Built one at a time, per instruction. Say which one is next.
# ═══════════════════════════════════════════════════════════════════════════
elif page in NOT_YET_BUILT:
    st.markdown(
        f"""<div class="placeholder-panel">
        <div style="font-family:'Source Serif 4', serif; font-size:1.1rem;">{page} — not yet implemented</div>
        <div style="font-size:0.8rem; margin-top:0.5rem;">This page exists in the reviewed UI mockup. Real data wiring
        is built one page at a time — tell me when to build this one.</div>
        </div>""", unsafe_allow_html=True
    )

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown(
    f"<div style='text-align:right; font-size:0.65rem; color:var(--ink-soft); margin-top:2rem; "
    f"padding-top:0.5rem; border-top:1px solid var(--line);'>"
    f"Last updated: {datetime.now().strftime('%H:%M:%S')} &nbsp;|&nbsp; Mule Account Detection System v0.2 (UI redesign)"
    f"</div>", unsafe_allow_html=True
)