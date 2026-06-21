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
    .status-dot {
        display: inline-block;
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background: #2ECC71;
        margin-right: 5px;
        animation: pulse 2s infinite;
    }
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
    .metric-card.cep     { border-top: 3px solid #6C3FA8; }

    .metric-value {
        font-size: 1.85rem;
        font-weight: 700;
        line-height: 1;
        color: #0F1923;
    }
    .metric-value.danger  { color: #C0392B; }
    .metric-value.warning { color: #B7791F; }
    .metric-value.safe    { color: #1E8449; }
    .metric-value.cep     { color: #6C3FA8; }
    .metric-label {
        font-size: 0.65rem;
        font-weight: 500;
        color: #7F8C9A;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-top: 0.35rem;
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
    .badge-cep   { background: #6C3FA8; color: white; }

    /* ── CEP pattern chip ── */
    .cep-chip {
        display: inline-block;
        font-size: 0.58rem;
        font-weight: 600;
        letter-spacing: 0.04em;
        padding: 1px 6px;
        margin: 1px 2px 1px 0;
        border-radius: 8px;
        background: #EFE6FB;
        color: #6C3FA8;
        font-family: 'JetBrains Mono', monospace;
        border: 1px solid #D9C5F0;
    }

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
    .alert-table tr.row-cep td { border-left: 3px solid #6C3FA8; background: #FBF9FE; }
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
def get_alerts(n=200):
    raw    = r.lrange('alerts', 0, n - 1)
    alerts = []
    for item in raw:
        try:
            alerts.append(json.loads(item))
        except:
            pass
    return alerts

def is_cep_only(alert):
    """True if this alert exists ONLY because CEP caught a pattern —
    the model's own tier did not independently flag it (tier was set
    to the CEP marker in consumer.py rather than L1/L2/L3)."""
    tier = alert.get('tier', '')
    return tier == "CEP — PATTERN MATCH"

def has_cep_pattern(alert):
    """True if this alert has ANY CEP pattern attached, regardless of
    whether the model also flagged it independently."""
    return bool(alert.get('cep_patterns'))

def get_stats(alerts):
    total     = len(alerts)
    l3        = sum(1 for a in alerts if 'L3' in a.get('tier', ''))
    l2        = sum(1 for a in alerts if 'L2' in a.get('tier', '') and 'L3' not in a.get('tier', ''))
    l1        = sum(1 for a in alerts if 'L1' in a.get('tier', '') and 'L2' not in a.get('tier', '') and 'L3' not in a.get('tier', ''))
    cep_only  = sum(1 for a in alerts if is_cep_only(a))
    cep_total = sum(1 for a in alerts if has_cep_pattern(a))
    confirmed = sum(1 for a in alerts if a.get('account_laundering') is True or a.get('is_laundering') == 1)
    gnn       = sum(1 for a in alerts if a.get('has_gnn_embed'))
    gnn_pct   = int(gnn / total * 100) if total > 0 else 0
    return total, l3, l2, l1, cep_only, cep_total, confirmed, gnn_pct

def deduplicate_alerts(alerts):
    """Keep the highest-priority alert per account. Priority order:
    1) CEP-only catches always win (model missed it, CEP didn't —
       this is the account we most want visible)
    2) otherwise, highest probability wins, as before
    """
    seen   = {}
    counts = {}
    for a in alerts:
        aid = a.get('account_id', '')
        counts[aid] = counts.get(aid, 0) + 1
        if aid not in seen:
            seen[aid] = a
            continue
        existing = seen[aid]
        existing_cep_only = is_cep_only(existing)
        new_cep_only       = is_cep_only(a)
        if new_cep_only and not existing_cep_only:
            seen[aid] = a
        elif new_cep_only == existing_cep_only:
            if a.get('probability', 0) > existing.get('probability', 0):
                seen[aid] = a
    return seen, counts

def sort_for_display(deduped_items):
    """Sort so CEP-only catches always surface at the very top of the
    table, regardless of their (typically low) model score — since the
    entire point of CEP is to catch what the model's score would have
    missed. Within each group, sort by probability as before."""
    cep_group    = [item for item in deduped_items if is_cep_only(item[1])]
    normal_group = [item for item in deduped_items if not is_cep_only(item[1])]
    cep_group.sort(key=lambda x: x[1].get('probability', 0), reverse=True)
    normal_group.sort(key=lambda x: x[1].get('probability', 0), reverse=True)
    return cep_group + normal_group

# ── Header ────────────────────────────────────────────────────────────────────
now = datetime.now().strftime('%Y-%m-%d  %H:%M:%S')
st.markdown(f"""
<div class="sys-header">
    <div class="sys-header-left">
        <h1>Mule Account Detection System</h1>
        <p>CyberPeace Foundation &nbsp;/&nbsp; Research Wing &nbsp;/&nbsp;
        Hybrid XGBoost + GNN + CEP &nbsp;/&nbsp; IBM AML &nbsp;/&nbsp;
        PMLA 2002 &nbsp;/&nbsp; RBI FREE-AI 2025</p>
    </div>
    <div class="sys-header-right">
        <span class="status-dot"></span>Live &nbsp;&nbsp; {now}
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
    st.markdown(f"CEP — Pattern Match: **independent of score**")
    st.markdown("---")
    st.markdown("### Model")
    st.markdown(f"Hybrid XGBoost v1")
    st.markdown(f"Features: **{config['n_total_features']}**")
    st.markdown(f"Behavioral: **12**")
    st.markdown(f"GNN embeddings: **64**")
    st.markdown(f"AUC-ROC: **{config['auc_roc']:.4f}**")
    st.markdown(f"Avg Precision: **{config['avg_precision']:.4f}**")
    st.markdown("---")
    st.markdown("### CEP Patterns")
    for p in ["RAPID_FORWARD", "CONCENTRATED_DESTINATION",
              "BURST_ACTIVATION", "FAN_OUT_SPIKE"]:
        st.markdown(f"- {p}")
    st.markdown("---")
    st.markdown("### Regulatory Alignment")
    for reg in ["PMLA 2002", "RBI Master Directions",
                "RBI Directive Jul 2024", "DPDP Act 2023",
                "RBI FREE-AI Aug 2025"]:
        st.markdown(f"- {reg}")

# ── Fetch data ────────────────────────────────────────────────────────────────
alerts = get_alerts(200)
total, l3, l2, l1, cep_only, cep_total, confirmed, gnn_pct = get_stats(alerts)
deduped, counts = deduplicate_alerts(alerts)

# ── Metric cards ──────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5, c6, c7 = st.columns(7)

def metric_card(col, value, label, style="neutral"):
    col.markdown(f"""
    <div class="metric-card {style}">
        <div class="metric-value {style if style in ('danger','warning','safe','cep') else ''}">{value}</div>
        <div class="metric-label">{label}</div>
    </div>""", unsafe_allow_html=True)

metric_card(c1, total,     "Total Alerts",          "info")
metric_card(c2, l3,        "L3 Auto-Freeze",         "danger")
metric_card(c3, l2,        "L2 Soft Hold",           "warning")
metric_card(c4, l1,        "L1 Review",              "safe")
metric_card(c5, cep_only,  "CEP-Only Catches",       "cep")
metric_card(c6, confirmed, "Confirmed Laundering",   "danger")
metric_card(c7, f"{gnn_pct}%", "GNN-Assisted",       "neutral")

if cep_only > 0:
    st.markdown(
        f"<div style='background:#FBF9FE;border:1px solid #D9C5F0;border-radius:3px;"
        f"padding:0.6rem 1rem;margin-bottom:1rem;font-size:0.8rem;color:#4A2A78;'>"
        f"<strong>{cep_only} account(s)</strong> flagged by behavioral pattern detection "
        f"(CEP) alone — the model's probability score did not independently cross "
        f"threshold for these. Shown pinned at the top of the alert queue below."
        f"</div>", unsafe_allow_html=True
    )

st.markdown("<br>", unsafe_allow_html=True)

# ── Main layout ───────────────────────────────────────────────────────────────
left, right = st.columns([3, 2])

# ── LEFT: Alert table ─────────────────────────────────────────────────────────
with left:
    st.markdown('<div class="section-label">Alert Queue — Unique Accounts (CEP-Only Catches Pinned to Top)</div>',
                unsafe_allow_html=True)

    if not deduped:
        st.markdown("""
        <div style="background:white;border:1px solid #DDE3EC;border-radius:3px;
        padding:2rem;text-align:center;color:#7F8C9A;font-size:0.85rem;">
            No alerts detected. Start the producer and consumer to begin real-time detection.
        </div>""", unsafe_allow_html=True)
    else:
        sorted_accounts = sort_for_display(list(deduped.items()))

        rows_html = ""
        for aid, a in sorted_accounts[:40]:
            tier     = a.get('tier', 'CLEAR')
            prob     = a.get('probability', 0)
            ts       = a.get('timestamp', '')[:16].replace('T', ' ')
            cnt      = counts.get(aid, 1)
            fan      = a.get('fan_out', '-')
            xbank    = a.get('cross_bank', '-')
            txn      = a.get('total_txn', '-')
            gnn      = 'Yes' if a.get('has_gnn_embed') else 'No'
            conf     = a.get('account_laundering', False) or a.get('is_laundering', 0) == 1
            patterns = a.get('cep_patterns', []) or []
            cep_flag = is_cep_only(a)

            if cep_flag:
                badge   = '<span class="badge badge-cep">CEP</span>'
                row_cls = 'row-cep'
            elif 'L3' in tier:
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

            pattern_chips = ''.join(
                f'<span class="cep-chip">{p}</span>' for p in patterns
            ) if patterns else '<span style="color:#B8C2CC">—</span>'

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
                <td>{pattern_chips}</td>
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
                    <th>CEP Pattern</th>
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
    if total > 0 and (l3 + l2 + l1 + cep_only) > 0:
        fig2, ax2 = plt.subplots(figsize=(4, 2.2))
        data   = [(l3,'L3 Auto-Freeze','#C0392B'),
                  (l2,'L2 Soft Hold',  '#B7791F'),
                  (l1,'L1 Review',     '#1E8449'),
                  (cep_only,'CEP-Only','#6C3FA8')]
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
    st.markdown('<div class="section-label">Highest Priority Accounts</div>',
                unsafe_allow_html=True)
    if deduped:
        top5 = sort_for_display(list(deduped.items()))[:5]
        for aid, a in top5:
            prob     = a.get('probability', 0)
            cnt      = counts.get(aid, 1)
            tier     = a.get('tier', '')
            cep_flag = is_cep_only(a)
            if cep_flag:
                color = '#6C3FA8'
                tag   = 'CEP'
            elif 'L3' in tier:
                color = '#C0392B'
                tag   = 'L3'
            elif 'L2' in tier:
                color = '#B7791F'
                tag   = 'L2'
            else:
                color = '#1E8449'
                tag   = 'L1'
            st.markdown(
                f"<div style='display:flex;justify-content:space-between;"
                f"padding:0.3rem 0;border-bottom:1px solid #F0F2F5;"
                f"font-size:0.75rem'>"
                f"<span style='font-family:JetBrains Mono,monospace;color:#0F1923'>{aid[:14]}</span>"
                f"<span style='color:{color};font-weight:600'>{tag} · {prob:.4f}</span>"
                f"<span style='color:#7F8C9A'>{cnt} alerts</span>"
                f"</div>",
                unsafe_allow_html=True
            )

# ── SHAP Explanation ──────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<div class="section-label">Decision Explanation — SHAP Feature Attribution</div>',
            unsafe_allow_html=True)

if deduped:
    account_list = [aid for aid, _ in sort_for_display(list(deduped.items()))]
    selected = st.selectbox("Select account to investigate",
                            account_list, label_visibility="collapsed")

    if selected:
        best_alert = deduped[selected]
        alert_count = counts.get(selected, 1)
        cep_patterns_selected = best_alert.get('cep_patterns', []) or []

        exp_left, exp_right = st.columns([1, 2])

        with exp_left:
            st.markdown('<div class="section-label">Account Summary</div>',
                        unsafe_allow_html=True)

            fields = [
                ("Account ID",       selected),
                ("Total alerts",     str(alert_count)),
                ("Max score",        f"{best_alert.get('probability',0):.4f}"),
                ("Risk tier",        best_alert.get('tier','—')),
                ("CEP patterns",     ", ".join(cep_patterns_selected) if cep_patterns_selected else "None"),
                ("Transactions",     str(best_alert.get('total_txn','—'))),
                ("Fan-out degree",   str(best_alert.get('fan_out','—'))),
                ("Cross-bank",       str(best_alert.get('cross_bank','—'))),
                ("Max amount",       f"Rs. {best_alert.get('max_amount',0):,.0f}"),
                ("GNN embedding",    "Present" if best_alert.get('has_gnn_embed') else "Not found"),
                ("Confirmed launder",
                 "YES" if (best_alert.get('account_laundering') or
                           best_alert.get('is_laundering')==1) else "No"),
            ]

            html_fields = ""
            for label, value in fields:
                color = "#C0392B" if label == "Confirmed launder" and value == "YES" else "#0F1923"
                if label == "CEP patterns" and cep_patterns_selected:
                    color = "#6C3FA8"
                html_fields += (
                    f"<div class='acct-field'>"
                    f"<span class='acct-field-label'>{label}</span>"
                    f"<span class='acct-field-value' style='color:{color}'>{value}</span>"
                    f"</div>"
                )
            st.markdown(html_fields, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # STR generation
            if st.button("Generate STR Report", type="primary"):
                tier      = best_alert.get('tier','—')
                prob      = best_alert.get('probability',0)
                rec_action = (
                    "IMMEDIATE ACCOUNT FREEZE AND STR FILING WITH FIU-IND"
                    if prob >= L3_THRESHOLD
                    else "Analyst review and enhanced due diligence required"
                )
                cep_section = (
                    f"\nCEP BEHAVIORAL PATTERNS DETECTED\n"
                    f"{', '.join(cep_patterns_selected)}\n"
                    f"Note: These patterns were detected via stateful sequence analysis "
                    f"independent of the XGBoost probability score.\n"
                    if cep_patterns_selected else ""
                )
                str_text = f"""SUSPICIOUS TRANSACTION REPORT
FIU-IND FORMAT — PMLA 2002 SECTION 12

Generated:      {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Report Type:    Automated Detection — Mule Account Behavioral Analysis

SUBJECT ACCOUNT
Account ID:           {selected}
Risk Tier:            {tier}
Fraud Probability:    {prob:.4f}
Total System Alerts:  {alert_count}
{cep_section}
BEHAVIORAL INDICATORS
Total Transactions:    {best_alert.get('total_txn','—')}
Fan-Out Degree:        {best_alert.get('fan_out','—')} unique destination banks
Cross-Bank Degree:     {best_alert.get('cross_bank','—')} unique receiving banks
Peak Transaction:      Rs. {best_alert.get('max_amount',0):,.2f}
Detection Amount:      Rs. {best_alert.get('amount',0):,.2f}
Payment Format:        {best_alert.get('payment_format','—')}
GNN Graph Context:     {'Present — account found in IBM AML transaction graph' if best_alert.get('has_gnn_embed') else 'Not in graph — new account'}
Confirmed Laundering:  {'YES — account appears in confirmed IBM AML laundering records' if (best_alert.get('account_laundering') or best_alert.get('is_laundering')==1) else 'Not confirmed in current dataset'}

DETECTION METHODOLOGY
Phase 1:   Rule Engine — PMLA-aligned threshold rules (R1-R5)
Phase 2:   XGBoost Classifier — 12 behavioral feature vectors
Phase 4:   MuleGNN (GCN + EdgeProp) — 64-dimensional graph embeddings
Hybrid:    76-feature XGBoost trained on behavioral + GNN features
CEP:       Stateful multi-transaction pattern detection (4 pattern types)
Score:     {prob:.4f} / threshold: {config['optimal_threshold']:.4f}
AUC-ROC:   {config['auc_roc']:.4f}

REGULATORY BASIS
Prevention of Money Laundering Act 2002
RBI Master Direction on KYC (Updated 2024)
RBI Governor Directive on Mule Accounts — July 2024
Digital Personal Data Protection Act 2023
RBI Framework for Responsible and Ethical Enablement of AI (FREE-AI) — August 2025

RECOMMENDED ACTION
{rec_action}

---
Generated by: Mule Account Detection System v1.0
Organisation: CyberPeace Foundation Research Wing
Intern:        Anish Banerjee — B.Tech CSE, DBS Global University
"""
                st.download_button(
                    label="Download STR (.txt)",
                    data=str_text,
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
                shap_vals = explainer.shap_values(X_scaled)[0][:12]

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

                # GNN contribution summary
                gnn_shap_sum = float(np.sum(np.abs(explainer.shap_values(X_scaled)[0][12:])))
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

                if cep_patterns_selected:
                    st.markdown(
                        f"<div style='font-size:0.72rem;color:#6C3FA8;margin-top:0.3rem'>"
                        f"This account was also flagged by CEP behavioral pattern detection: "
                        f"<strong>{', '.join(cep_patterns_selected)}</strong> — a signal the "
                        f"SHAP attribution above does not capture, since it depends on "
                        f"transaction sequence and timing rather than single-snapshot features."
                        f"</div>", unsafe_allow_html=True
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