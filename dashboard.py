import streamlit as st
import redis
import json
import pandas as pd
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

# ── Page configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Mule Account Detection — CyberPeace Foundation",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main {
        background-color: #F0F2F5;
    }

    .block-container {
        padding: 0 1.5rem 2rem 1.5rem;
        max-width: 100%;
    }

    /* ── Header ── */
    .sys-header {
        background: #0F1923;
        color: white;
        padding: 0.9rem 1.8rem;
        margin: 0 -1.5rem 1.5rem -1.5rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-bottom: 2px solid #C8A032;
    }
    .sys-header-left h1 {
        font-size: 1.05rem;
        font-weight: 600;
        margin: 0;
        letter-spacing: 0.03em;
        color: white;
    }
    .sys-header-left p {
        font-size: 0.68rem;
        margin: 0.15rem 0 0 0;
        color: #8A9BB0;
        letter-spacing: 0.04em;
    }
    .sys-header-right {
        font-size: 0.68rem;
        color: #8A9BB0;
        text-align: right;
    }
    /* Status dot now driven by actual data freshness, not hardcoded */
    .status-dot {
        display: inline-block;
        width: 7px;
        height: 7px;
        border-radius: 50%;
        margin-right: 5px;
    }
    .status-dot.live  { background: #2ECC71; animation: pulse 2s infinite; }
    .status-dot.warn  { background: #B7791F; animation: pulse 1.2s infinite; }
    .status-dot.stale { background: #C0392B; animation: none; }
    @keyframes pulse {
        0%   { opacity: 1; }
        50%  { opacity: 0.4; }
        100% { opacity: 1; }
    }

    /* ── Metric cards ── */
    .metric-row {
        display: grid;
        grid-template-columns: repeat(6, 1fr);
        gap: 0.75rem;
        margin-bottom: 1.25rem;
    }
    .metric-card {
        background: white;
        border: 1px solid #DDE3EC;
        border-radius: 3px;
        padding: 0.9rem 1rem;
        text-align: center;
    }
    .metric-card.danger  { border-top: 3px solid #C0392B; }
    .metric-card.warning { border-top: 3px solid #B7791F; }
    .metric-card.safe    { border-top: 3px solid #1E8449; }
    .metric-card.info    { border-top: 3px solid #1A5276; }
    .metric-card.neutral { border-top: 3px solid #5D6D7E; }

    .metric-value {
        font-size: 1.85rem;
        font-weight: 700;
        line-height: 1;
        color: #0F1923;
    }
    .metric-value.danger  { color: #C0392B; }
    .metric-value.warning { color: #B7791F; }
    .metric-value.safe    { color: #1E8449; }
    .metric-label {
        font-size: 0.65rem;
        font-weight: 500;
        color: #7F8C9A;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-top: 0.35rem;
    }
    .metric-caption {
        font-size: 0.65rem;
        color: #9AA7B5;
        margin: -0.75rem 0 1rem 0;
        font-style: italic;
    }

    /* ── Section labels ── */
    .section-label {
        font-size: 0.65rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        color: #5D6D7E;
        padding-bottom: 0.4rem;
        border-bottom: 1px solid #DDE3EC;
        margin-bottom: 0.85rem;
    }

    /* ── Tier badges ── */
    .badge {
        display: inline-block;
        font-size: 0.62rem;
        font-weight: 600;
        letter-spacing: 0.06em;
        padding: 2px 7px;
        border-radius: 2px;
        font-family: 'JetBrains Mono', monospace;
    }
    .badge-l3    { background: #C0392B; color: white; }
    .badge-l2    { background: #B7791F; color: white; }
    .badge-l1    { background: #1E8449; color: white; }
    .badge-clear { background: #5D6D7E; color: white; }
    .badge-cep   { background: #5B4B8A; color: white; white-space: nowrap; }

    /* ── Alert table ── */
    .alert-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.78rem;
    }
    .alert-table th {
        background: #0F1923;
        color: #8A9BB0;
        font-weight: 500;
        font-size: 0.62rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        padding: 0.5rem 0.75rem;
        text-align: left;
        border-bottom: 2px solid #C8A032;
    }
    .alert-table td {
        padding: 0.5rem 0.75rem;
        border-bottom: 1px solid #EDF0F5;
        color: #2C3E50;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
    }
    .alert-table tr:hover td { background: #F7F9FC; }
    .alert-table tr.row-l3 td { border-left: 3px solid #C0392B; }
    .alert-table tr.row-l2 td { border-left: 3px solid #B7791F; }
    .alert-table tr.row-l1 td { border-left: 3px solid #1E8449; }
    .alert-table .confirmed-yes {
        color: #C0392B;
        font-weight: 600;
    }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background: #0F1923;
    }
    section[data-testid="stSidebar"] * {
        color: #C8D6E5 !important;
    }
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] strong {
        color: white !important;
    }
    section[data-testid="stSidebar"] hr {
        border-color: #1E2D3D !important;
    }

    /* ── Hide Streamlit chrome ── */
    #MainMenu { visibility: hidden; }
    footer     { visibility: hidden; }
    header     { visibility: hidden; }

    /* ── Chart containers ── */
    .chart-box {
        background: white;
        border: 1px solid #DDE3EC;
        border-radius: 3px;
        padding: 0.75rem;
        margin-bottom: 0.75rem;
    }

    /* ── Account summary panel ── */
    .acct-field {
        display: flex;
        justify-content: space-between;
        padding: 0.3rem 0;
        border-bottom: 1px solid #F0F2F5;
        font-size: 0.78rem;
    }
    .acct-field-label { color: #7F8C9A; }
    .acct-field-value {
        font-weight: 500;
        color: #0F1923;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
    }
</style>
""", unsafe_allow_html=True)

# ── Cached resources ──────────────────────────────────────────────────────────
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

BEHAVIORAL_LABELS = [
    'Velocity Ratio', 'Balance Retention', 'Fan-Out Degree',
    'Cross-Bank Degree', 'Round Amount Ratio', 'Structuring Score',
    'Amount Variance', 'Session Burst', 'Dormancy Ratio',
    'Total Transactions', 'Avg Amount', 'Max Amount'
]

L3_THRESHOLD = 0.70
L2_THRESHOLD = 0.50
L1_THRESHOLD = 0.30

# ── Data functions ────────────────────────────────────────────────────────────
def get_alerts(n=1000):
    raw    = r.lrange('alerts', 0, n - 1)
    alerts = []
    for item in raw:
        try:
            alerts.append(json.loads(item))
        except:
            pass
    return alerts

def get_total_flagged():
    count = r.get("alerts_total_count")
    return int(count) if count else None

def get_stats(alerts):
    real_total = get_total_flagged()
    total     = real_total if real_total else len(alerts)
    l3        = sum(1 for a in alerts if 'L3' in a.get('tier', ''))
    l2        = sum(1 for a in alerts if 'L2' in a.get('tier', '') and 'L3' not in a.get('tier', ''))
    l1        = sum(1 for a in alerts if 'L1' in a.get('tier', '') and 'L2' not in a.get('tier', '') and 'L3' not in a.get('tier', ''))
    # NOTE: r.keys() is O(N) and blocks Redis. Left as-is deliberately --
    # a partial SCAN would silently under-report this number in a live demo,
    # which is worse than an accurate-but-slower one. Real fix: maintain an
    # INCR counter in consumer.py when launder_status:{id} is first set.
    confirmed = len(r.keys("launder_status:*"))
    gnn       = sum(1 for a in alerts if a.get('has_gnn_embed'))
    gnn_pct   = int(gnn / len(alerts) * 100) if len(alerts) > 0 else 0
    return total, l3, l2, l1, confirmed, gnn_pct

def deduplicate_alerts(alerts):
    seen   = {}
    counts = {}
    for a in alerts:
        aid = a.get('account_id', '')
        counts[aid] = counts.get(aid, 0) + 1
        if aid not in seen or a.get('probability', 0) > seen[aid].get('probability', 0):
            seen[aid] = a
    return seen, counts

def get_cep_badge_html(aid):
    """Cheap O(1) lookup (lindex, not lrange) -- one Redis round-trip per row,
    fetching only the newest CEP hit rather than the whole history."""
    try:
        raw = r.lindex(f"cep_alerts:{aid}", 0)
        if not raw:
            return "—"
        pattern = json.loads(raw).get('pattern_type', '')
        return f'<span class="badge badge-cep">{pattern}</span>' if pattern else "—"
    except Exception:
        return "—"

def get_system_status(alerts):
    """Derives live/warn/stale from the actual timestamp of the newest alert,
    instead of a hardcoded green dot that lies if the pipeline stalls."""
    if not alerts:
        return "NO DATA", "stale"
    try:
        latest_ts = alerts[0].get('timestamp', '')
        latest_dt = datetime.fromisoformat(latest_ts.replace('Z', ''))
        age_seconds = (datetime.now() - latest_dt).total_seconds()
    except Exception:
        return "UNKNOWN", "stale"
    if age_seconds < 30:
        return "LIVE", "live"
    elif age_seconds < 120:
        return "SLOWING", "warn"
    else:
        return "STALE", "stale"

# ── Fetch data (moved above header so the status dot can use it) ──────────────
alerts                = get_alerts(1000)
total, l3, l2, l1, confirmed, gnn_pct = get_stats(alerts)
deduped, counts       = deduplicate_alerts(alerts)
status_label, status_cls = get_system_status(alerts)

# ── Header ────────────────────────────────────────────────────────────────────
now = datetime.now().strftime('%Y-%m-%d  %H:%M:%S')
st.markdown(f"""
<div class="sys-header">
    <div class="sys-header-left">
        <h1>Mule Account Detection System</h1>
        <p>CyberPeace Foundation &nbsp;/&nbsp; Research Wing &nbsp;/&nbsp;
        Hybrid XGBoost + GNN &nbsp;/&nbsp; IBM AML &nbsp;/&nbsp;
        PMLA 2002 &nbsp;/&nbsp; RBI FREE-AI 2025</p>
    </div>
    <div class="sys-header-right">
        <span class="status-dot {status_cls}"></span>{status_label} &nbsp;&nbsp; {now}
    </div>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Controls")
    auto_refresh = st.toggle("Live refresh", value=True)
    refresh_rate = st.slider("Interval (seconds)", 3, 30, 5)
    st.markdown("---")
    st.markdown("### Detection Thresholds")
    st.markdown(f"Optimal score: **{config['optimal_threshold']:.4f}**")
    st.markdown(f"L3 Auto-Freeze: **{L3_THRESHOLD}+**")
    st.markdown(f"L2 Soft Hold: **{L2_THRESHOLD} – {L3_THRESHOLD}**")
    st.markdown(f"L1 Review: **{L1_THRESHOLD} – {L2_THRESHOLD}**")
    st.markdown("---")
    st.markdown("### Model")
    st.markdown(f"Hybrid XGBoost v1")
    st.markdown(f"Features: **{config['n_total_features']}**")
    st.markdown(f"Behavioral: **12**")
    st.markdown(f"GNN embeddings: **64**")
    st.markdown(f"AUC-ROC: **{config['auc_roc']:.4f}**")
    st.markdown(f"Avg Precision: **{config['avg_precision']:.4f}**")
    st.markdown("---")
    st.markdown("### Regulatory Alignment")
    for reg in ["PMLA 2002", "RBI Master Directions",
                "RBI Directive Jul 2024", "DPDP Act 2023",
                "RBI FREE-AI Aug 2025"]:
        st.markdown(f"- {reg}")

# ── Metric cards ──────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5, c6 = st.columns(6)

def metric_card(col, value, label, style="neutral"):
    col.markdown(f"""
    <div class="metric-card {style}">
        <div class="metric-value {style if style in ('danger','warning','safe') else ''}">{value}</div>
        <div class="metric-label">{label}</div>
    </div>""", unsafe_allow_html=True)

metric_card(c1, total,     "Total Flagged (All-Time)", "info")
metric_card(c2, l3,        "L3 Auto-Freeze",         "danger")
metric_card(c3, l2,        "L2 Soft Hold",           "warning")
metric_card(c4, l1,        "L1 Review",              "safe")
metric_card(c5, confirmed, "Confirmed Laundering",   "danger")
metric_card(c6, f"{gnn_pct}%", "GNN-Assisted",       "neutral")

st.markdown(
    "<div class='metric-caption'>L1/L2/L3 reflect the current alert queue "
    f"(last {len(alerts)} entries); Total Flagged is an all-time counter — "
    "the two are not expected to sum.</div>",
    unsafe_allow_html=True
)

# ── Main layout ───────────────────────────────────────────────────────────────
left, right = st.columns([3, 2])

# ── LEFT: Alert table ─────────────────────────────────────────────────────────
with left:
    st.markdown('<div class="section-label">Alert Queue — Unique Accounts, Ranked by Score</div>',
                unsafe_allow_html=True)

    if not deduped:
        st.markdown("""
        <div style="background:white;border:1px solid #DDE3EC;border-radius:3px;
        padding:2rem;text-align:center;color:#7F8C9A;font-size:0.85rem;">
            No alerts detected. Start the producer and consumer to begin real-time detection.
        </div>""", unsafe_allow_html=True)
    else:
        sorted_accounts = sorted(
            deduped.items(),
            key=lambda x: x[1].get('probability', 0),
            reverse=True
        )

        rows_html = ""
        for aid, a in sorted_accounts[:40]:
            tier  = a.get('tier', 'CLEAR')
            prob  = a.get('probability', 0)
            ts    = a.get('timestamp', '')[:16].replace('T', ' ')
            cnt   = counts.get(aid, 1)
            fan   = a.get('fan_out', '-')
            xbank = a.get('cross_bank', '-')
            txn   = a.get('total_txn', '-')
            gnn   = 'Yes' if a.get('has_gnn_embed') else 'No'
            conf  = a.get('account_laundering', False) or a.get('is_laundering', 0) == 1
            cep_badge = get_cep_badge_html(aid)

            if 'L3' in tier:
                badge   = '<span class="badge badge-l3">L3</span>'
                row_cls = 'row-l3'
            elif 'L2' in tier:
                badge   = '<span class="badge badge-l2">L2</span>'
                row_cls = 'row-l2'
            elif 'L1' in tier:
                badge   = '<span class="badge badge-l1">L1</span>'
                row_cls = 'row-l1'
            else:
                badge   = '<span class="badge badge-clear">CLR</span>'
                row_cls = ''

            confirmed_cell = (
                '<td class="confirmed-yes">YES</td>'
                if conf else '<td>-</td>'
            )

            rows_html += f"""
            <tr class="{row_cls}">
                <td>{ts}</td>
                <td>{aid[:14]}</td>
                <td style="text-align:right">{a.get('amount',0):,.0f}</td>
                <td style="text-align:right"><strong>{prob:.4f}</strong></td>
                <td>{badge}</td>
                <td>{cep_badge}</td>
                <td style="text-align:right">{cnt}</td>
                <td style="text-align:right">{txn}</td>
                <td style="text-align:right">{fan}</td>
                <td style="text-align:right">{xbank}</td>
                <td style="text-align:center">{gnn}</td>
                {confirmed_cell}
            </tr>"""

        st.markdown(f"""
        <table class="alert-table">
            <thead>
                <tr>
                    <th>Time</th>
                    <th>Account</th>
                    <th style="text-align:right">Amount (Rs.)</th>
                    <th style="text-align:right">Score</th>
                    <th>Tier</th>
                    <th>Pattern</th>
                    <th style="text-align:right">Alerts</th>
                    <th style="text-align:right">Txn</th>
                    <th style="text-align:right">Fan-Out</th>
                    <th style="text-align:right">X-Bank</th>
                    <th style="text-align:center">GNN</th>
                    <th>Confirmed</th>
                </tr>
            </thead>
            <tbody>{rows_html}</tbody>
        </table>
        """, unsafe_allow_html=True)

# ── RIGHT: Charts ─────────────────────────────────────────────────────────────
with right:

    # Score distribution
    st.markdown('<div class="section-label">Score Distribution</div>',
                unsafe_allow_html=True)
    if alerts:
        probs = [a.get('probability', 0) for a in alerts]
        fig, ax = plt.subplots(figsize=(5, 2.2))
        ax.hist(probs, bins=25, color='#1A5276', edgecolor='#F0F2F5', linewidth=0.4)
        ax.axvline(x=L1_THRESHOLD, color='#1E8449', linestyle='--', linewidth=0.9,
                   label=f'L1 ({L1_THRESHOLD})')
        ax.axvline(x=L2_THRESHOLD, color='#B7791F', linestyle='--', linewidth=0.9,
                   label=f'L2 ({L2_THRESHOLD})')
        ax.axvline(x=L3_THRESHOLD, color='#C0392B', linestyle='--', linewidth=0.9,
                   label=f'L3 ({L3_THRESHOLD})')
        ax.set_xlabel('Fraud Probability', fontsize=7, color='#5D6D7E')
        ax.set_ylabel('Count', fontsize=7, color='#5D6D7E')
        ax.tick_params(labelsize=6, colors='#5D6D7E')
        ax.legend(fontsize=6, framealpha=0.9)
        ax.set_facecolor('#F8FAFC')
        fig.patch.set_facecolor('white')
        for spine in ax.spines.values():
            spine.set_color('#DDE3EC')
        plt.tight_layout(pad=0.5)
        st.pyplot(fig)
        plt.close()
    else:
        st.markdown('<div style="color:#7F8C9A;font-size:0.78rem;padding:1rem 0">No data yet.</div>',
                    unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Tier breakdown
    st.markdown('<div class="section-label">Alert Tier Breakdown</div>',
                unsafe_allow_html=True)
    if total > 0 and (l3 + l2 + l1) > 0:
        fig2, ax2 = plt.subplots(figsize=(4, 2.2))
        data   = [(l3,'L3 Auto-Freeze','#C0392B'),
                  (l2,'L2 Soft Hold',  '#B7791F'),
                  (l1,'L1 Review',     '#1E8449')]
        data   = [(s, l, c) for s, l, c in data if s > 0]
        sizes, labels, colors = zip(*data)
        wedges, texts, autotexts = ax2.pie(
            sizes, labels=labels, colors=colors,
            autopct='%1.0f%%',
            textprops={'fontsize': 6.5},
            startangle=90,
            wedgeprops={'linewidth': 0.5, 'edgecolor': 'white'}
        )
        for at in autotexts:
            at.set_fontsize(6)
        fig2.patch.set_facecolor('white')
        plt.tight_layout(pad=0.3)
        st.pyplot(fig2)
        plt.close()

    st.markdown("<br>", unsafe_allow_html=True)

    # Top accounts
    st.markdown('<div class="section-label">Highest Risk Accounts</div>',
                unsafe_allow_html=True)
    if deduped:
        top5 = sorted(deduped.items(),
                      key=lambda x: x[1].get('probability', 0),
                      reverse=True)[:5]
        for aid, a in top5:
            prob  = a.get('probability', 0)
            cnt   = counts.get(aid, 1)
            tier  = a.get('tier', '')
            color = '#C0392B' if 'L3' in tier else '#B7791F' if 'L2' in tier else '#1E8449'
            st.markdown(
                f"<div style='display:flex;justify-content:space-between;"
                f"padding:0.3rem 0;border-bottom:1px solid #F0F2F5;"
                f"font-size:0.75rem'>"
                f"<span style='font-family:JetBrains Mono,monospace;color:#0F1923'>{aid[:14]}</span>"
                f"<span style='color:{color};font-weight:600'>{prob:.4f}</span>"
                f"<span style='color:#7F8C9A'>{cnt} alerts</span>"
                f"</div>",
                unsafe_allow_html=True
            )

# ── SHAP Explanation ──────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<div class="section-label">Decision Explanation — SHAP Feature Attribution</div>',
            unsafe_allow_html=True)

if deduped:
    account_list = [aid for aid, _ in sorted(
        deduped.items(),
        key=lambda x: x[1].get('probability', 0),
        reverse=True
    )]
    selected = st.selectbox("Select account to investigate",
                            account_list, label_visibility="collapsed")

    if selected:
        best_alert = deduped[selected]
        alert_count = counts.get(selected, 1)

        exp_left, exp_right = st.columns([1, 2])

        with exp_left:
            st.markdown('<div class="section-label">Account Summary</div>',
                        unsafe_allow_html=True)

            # Get CEP patterns for this account
            cep_raw = r.lrange(f"cep_alerts:{selected}", 0, -1)
            cep_patterns = list({json.loads(a)['pattern_type'] for a in cep_raw}) if cep_raw else []
            cep_str = ", ".join(cep_patterns) if cep_patterns else "None"
            is_confirmed = (best_alert.get('account_laundering') or
                           best_alert.get('is_laundering') == 1)

            # Use native Streamlit components -- custom CSS classes are
            # blocked by Streamlit's shadow DOM in newer versions
            fields = [
                ("Account ID",        selected),
                ("Total Alerts",      str(alert_count)),
                ("Max Score",         f"{best_alert.get('probability',0):.4f}"),
                ("Risk Tier",         best_alert.get('tier','—')),
                ("CEP Patterns",      cep_str),
                ("Transactions",      str(best_alert.get('total_txn','—'))),
                ("Fan-Out Degree",    str(best_alert.get('fan_out','—'))),
                ("Cross-Bank",        str(best_alert.get('cross_bank','—'))),
                ("Max Amount",        f"Rs. {best_alert.get('max_amount',0):,.0f}"),
                ("GNN Embedding",     "Present" if best_alert.get('has_gnn_embed') else "Not found"),
                ("Confirmed Launder", "YES" if is_confirmed else "No"),
            ]

            for label, value in fields:
                col_a, col_b = st.columns([1, 1])
                with col_a:
                    st.markdown(f"<span style='color:#7F8C9A;font-size:0.78rem'>{label}</span>",
                                unsafe_allow_html=True)
                with col_b:
                    color = "#C0392B" if label == "Confirmed Launder" and value == "YES" else "#0F1923"
                    st.markdown(f"<span style='color:{color};font-size:0.78rem;font-weight:500'>{value}</span>",
                                unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # STR generation — use session_state to persist download button
            # across Streamlit re-renders (clicking st.button triggers re-render
            # which destroys any widget created inside the if block)
            if 'str_text' not in st.session_state:
                st.session_state.str_text = None
            if 'str_account' not in st.session_state:
                st.session_state.str_account = None

            # Reset STR if account changed
            if st.session_state.str_account != selected:
                st.session_state.str_text = None
                st.session_state.str_account = selected

            if st.button("Generate STR Report", type="primary"):
                tier       = best_alert.get('tier','—')
                prob       = best_alert.get('probability',0)
                rec_action = (
                    "IMMEDIATE ACCOUNT FREEZE AND STR FILING WITH FIU-IND"
                    if prob >= L3_THRESHOLD
                    else "Analyst review and enhanced due diligence required"
                )
                st.session_state.str_text = f"""SUSPICIOUS TRANSACTION REPORT
FIU-IND FORMAT - PMLA 2002 SECTION 12

Generated:      {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Report Type:    Automated Detection - Mule Account Behavioral Analysis

SUBJECT ACCOUNT
Account ID:           {selected}
Risk Tier:            {tier}
Fraud Probability:    {prob:.4f}
Total System Alerts:  {alert_count}
CEP Patterns:         {cep_str}

BEHAVIORAL INDICATORS
Total Transactions:    {best_alert.get('total_txn','—')}
Fan-Out Degree:        {best_alert.get('fan_out','—')} unique destination banks
Cross-Bank Degree:     {best_alert.get('cross_bank','—')} unique receiving banks
Peak Transaction:      Rs. {best_alert.get('max_amount',0):,.2f}
Detection Amount:      Rs. {best_alert.get('amount',0):,.2f}
Payment Format:        {best_alert.get('payment_format','—')}
GNN Graph Context:     {'Present - account found in IBM AML transaction graph' if best_alert.get('has_gnn_embed') else 'Not in graph - new account'}
Confirmed Laundering:  {'YES - account appears in confirmed IBM AML laundering records' if is_confirmed else 'Not confirmed in current dataset'}

DETECTION METHODOLOGY
Phase 1:   CEP Engine - 6 IBM AML-aligned behavioral patterns
Phase 2:   XGBoost Classifier - 12 behavioral feature vectors
Phase 3:   MuleGNN (GCN + EdgeProp) - 64-dimensional graph embeddings
Hybrid:    76-feature XGBoost trained on behavioral + GNN features
Score:     {prob:.4f} / threshold: {config['optimal_threshold']:.4f}
AUC-ROC:   {config['auc_roc']:.4f}

REGULATORY BASIS
Prevention of Money Laundering Act 2002
RBI Master Direction on KYC (Updated 2024)
RBI Governor Directive on Mule Accounts - July 2024
Digital Personal Data Protection Act 2023
RBI Framework for Responsible and Ethical Enablement of AI (FREE-AI) - August 2025

RECOMMENDED ACTION
{rec_action}

---
Generated by: Mule Account Detection System v1.0
Organisation: CyberPeace Foundation Research Wing
Intern:        Anish Banerjee - B.Tech CSE, DBS Global University
"""

            if st.session_state.str_text:
                st.success("STR generated. Click below to download.")
                st.download_button(
                    label="Download STR (.txt)",
                    data=st.session_state.str_text,
                    file_name=f"STR_{selected}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )

        with exp_right:
            st.markdown('<div class="section-label">SHAP Feature Attribution (Behavioral Features)</div>',
                        unsafe_allow_html=True)
            st.caption("Red bars increase fraud risk. Green bars decrease it.")

            acct_state = r.hgetall(f"acct:{selected}")
            if acct_state:
                total_txn  = int(acct_state.get('total_txn', 1))
                sum_amount = float(acct_state.get('sum_amount', 0))
                max_amount = float(acct_state.get('max_amount', 0))
                sum_sq     = float(acct_state.get('sum_sq_amount', 0))
                ud         = set(acct_state.get('unique_dest','').split(','))
                ub         = set(acct_state.get('unique_banks','').split(','))
                ud.discard('')
                ub.discard('')
                round_cnt  = int(acct_state.get('round_count', 0))
                avg_amount = sum_amount / total_txn if total_txn > 0 else 0
                amt        = best_alert.get('amount', avg_amount)

                vel   = min(amt / (max_amount + 1e-9), 1.0) if total_txn > 1 else 0.95
                bret  = max(0.0, 1.0 - vel)
                rrat  = round_cnt / total_txn if total_txn > 0 else 0.0
                if total_txn > 1:
                    msq  = sum_sq / total_txn
                    var  = msq - avg_amount**2
                    std  = math.sqrt(max(var, 0))
                    avar = 1.0 / (1.0 + std / (avg_amount + 1e-9))
                else:
                    avar = 1.0

                behavioral = [
                    vel, bret,
                    float(len(ud)), float(len(ub)),
                    rrat, rrat * 0.5, avar,
                    min(amt / (avg_amount * 5 + 1e-9), 1.0),
                    max(0.0, 1.0 - min(total_txn / 10.0, 1.0)),
                    float(total_txn), avg_amount, max_amount,
                ]

                emb_json = r.get(f"embed:{selected}")
                gnn_emb  = json.loads(emb_json) if emb_json else [0.0] * 64
                features = behavioral + gnn_emb

                X         = np.array(features).reshape(1, -1)
                X_scaled  = scaler.transform(X)
                # Computed once and sliced -- previously called twice
                # (once for the behavioral bars, once for the GNN %),
                # doubling SHAP compute cost per account selection for no reason.
                shap_row  = explainer.shap_values(X_scaled)[0]
                shap_vals = shap_row[:12]

                sorted_idx = np.argsort(np.abs(shap_vals))[::-1][:8]
                vals       = shap_vals[sorted_idx]
                lbls       = [BEHAVIORAL_LABELS[i] for i in sorted_idx]
                bar_colors = ['#C0392B' if v > 0 else '#1E8449' for v in vals]

                fig3, ax3 = plt.subplots(figsize=(6, 3.2))
                bars = ax3.barh(lbls, vals, color=bar_colors,
                                edgecolor='white', linewidth=0.4, height=0.6)
                ax3.axvline(x=0, color='#5D6D7E', linewidth=0.7)
                ax3.set_xlabel('SHAP value', fontsize=7, color='#5D6D7E')
                ax3.tick_params(labelsize=6.5, colors='#2C3E50')
                ax3.set_facecolor('#F8FAFC')
                fig3.patch.set_facecolor('white')
                for spine in ax3.spines.values():
                    spine.set_color('#DDE3EC')
                red_p   = mpatches.Patch(color='#C0392B', label='Increases fraud risk')
                green_p = mpatches.Patch(color='#1E8449', label='Decreases fraud risk')
                ax3.legend(handles=[red_p, green_p], fontsize=6.5,
                           framealpha=0.95, loc='lower right')
                plt.tight_layout(pad=0.5)
                st.pyplot(fig3)
                plt.close()

                # GNN contribution summary (reuses shap_row, no second call)
                gnn_shap_sum = float(np.sum(np.abs(shap_row[12:])))
                beh_shap_sum = float(np.sum(np.abs(shap_vals)))
                total_shap   = gnn_shap_sum + beh_shap_sum
                if total_shap > 0:
                    gnn_pct_contrib = gnn_shap_sum / total_shap * 100
                    beh_pct_contrib = beh_shap_sum / total_shap * 100
                    st.markdown(
                        f"<div style='font-size:0.72rem;color:#5D6D7E;margin-top:0.5rem'>"
                        f"Feature contribution — Behavioral: <strong>{beh_pct_contrib:.0f}%</strong> &nbsp;|&nbsp; "
                        f"GNN graph embeddings: <strong>{gnn_pct_contrib:.0f}%</strong>"
                        f"</div>",
                        unsafe_allow_html=True
                    )
            else:
                st.markdown(
                    "<div style='color:#7F8C9A;font-size:0.78rem;padding:1rem 0'>"
                    "Account state not found in Redis. "
                    "Process more transactions to build account history."
                    "</div>",
                    unsafe_allow_html=True
                )

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown(
    f"<div style='text-align:right;font-size:0.65rem;color:#A0AEC0;"
    f"margin-top:2rem;padding-top:0.5rem;border-top:1px solid #DDE3EC'>"
    f"Last updated: {datetime.now().strftime('%H:%M:%S')} &nbsp;|&nbsp; "
    f"Refreshing every {refresh_rate if auto_refresh else '—'}s &nbsp;|&nbsp; "
    f"Mule Account Detection System v1.0 &nbsp;|&nbsp; "
    f"CyberPeace Foundation Research Wing</div>",
    unsafe_allow_html=True
)

# ── Auto refresh ──────────────────────────────────────────────────────────────
if auto_refresh:
    time.sleep(refresh_rate)
    st.rerun()