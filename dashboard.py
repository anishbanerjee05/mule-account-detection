import streamlit as st
import redis
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pickle
import shap
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

# ── Custom CSS — professional bank analyst interface ──────────────────────────
st.markdown("""
<style>
    /* Base */
    .main { background-color: #F8F9FA; }
    .block-container { padding: 1.5rem 2rem; }

    /* Header */
    .dashboard-header {
        background: #1A2B4A;
        color: white;
        padding: 1.2rem 2rem;
        margin: -1.5rem -2rem 1.5rem -2rem;
        border-bottom: 3px solid #C8A951;
    }
    .dashboard-header h1 {
        font-size: 1.4rem;
        font-weight: 600;
        margin: 0;
        letter-spacing: 0.02em;
    }
    .dashboard-header p {
        font-size: 0.8rem;
        margin: 0.2rem 0 0 0;
        opacity: 0.7;
    }

    /* Metric cards */
    .metric-card {
        background: white;
        border: 1px solid #E2E8F0;
        border-radius: 4px;
        padding: 1rem 1.2rem;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #1A2B4A;
        line-height: 1;
    }
    .metric-label {
        font-size: 0.72rem;
        color: #718096;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-top: 0.3rem;
    }
    .metric-value.red  { color: #C53030; }
    .metric-value.amber{ color: #B7791F; }
    .metric-value.green{ color: #276749; }

    /* Alert table rows */
    .alert-l3 { background: #FFF5F5; border-left: 4px solid #C53030; }
    .alert-l2 { background: #FFFBEB; border-left: 4px solid #B7791F; }
    .alert-l1 { background: #F0FFF4; border-left: 4px solid #276749; }

    /* Section headers */
    .section-title {
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #4A5568;
        border-bottom: 1px solid #E2E8F0;
        padding-bottom: 0.4rem;
        margin-bottom: 1rem;
    }

    /* Tier badges */
    .badge-l3 { background:#C53030; color:white; padding:2px 8px; border-radius:3px; font-size:0.7rem; font-weight:600; }
    .badge-l2 { background:#B7791F; color:white; padding:2px 8px; border-radius:3px; font-size:0.7rem; font-weight:600; }
    .badge-l1 { background:#276749; color:white; padding:2px 8px; border-radius:3px; font-size:0.7rem; font-weight:600; }
    .badge-clear { background:#718096; color:white; padding:2px 8px; border-radius:3px; font-size:0.7rem; }

    /* Hide Streamlit default elements */
    #MainMenu { visibility: hidden; }
    footer    { visibility: hidden; }
    header    { visibility: hidden; }

    /* Sidebar */
    .css-1d391kg { background: #1A2B4A; }
    section[data-testid="stSidebar"] { background: #1A2B4A; }
    section[data-testid="stSidebar"] * { color: white !important; }
</style>
""", unsafe_allow_html=True)

# ── Redis connection ───────────────────────────────────────────────────────────
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

BEHAVIORAL_FEATURES = config['feature_cols_behavioral']

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="dashboard-header">
    <h1>Mule Account Detection System</h1>
    <p>CyberPeace Foundation &nbsp;|&nbsp; Research Wing &nbsp;|&nbsp;
    Hybrid XGBoost + GNN &nbsp;|&nbsp; IBM AML Dataset &nbsp;|&nbsp;
    PMLA / RBI FREE-AI Aligned</p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### System Controls")
    auto_refresh = st.toggle("Live refresh", value=True)
    refresh_rate = st.slider("Refresh interval (seconds)", 2, 30, 5)
    st.markdown("---")
    st.markdown("### Detection Thresholds")
    st.markdown(f"**Optimal:** {config['optimal_threshold']:.4f}")
    st.markdown("**L3 Auto-Freeze:** 0.85+")
    st.markdown("**L2 Soft Hold:** 0.50 - 0.85")
    st.markdown("**L1 Review:** 0.30 - 0.50")
    st.markdown("---")
    st.markdown("### Model Info")
    st.markdown(f"**Model:** Hybrid XGBoost v1")
    st.markdown(f"**Features:** {config['n_total_features']} (12 behavioral + 64 GNN)")
    st.markdown(f"**AUC-ROC:** {config['auc_roc']:.4f}")
    st.markdown(f"**Avg Precision:** {config['avg_precision']:.4f}")
    st.markdown("---")
    st.markdown("### Regulatory")
    st.markdown("PMLA 2002")
    st.markdown("RBI Master Directions")
    st.markdown("DPDP Act 2023")
    st.markdown("RBI FREE-AI Framework 2025")

# ── Fetch alerts from Redis ───────────────────────────────────────────────────
def get_alerts(n=100):
    raw = r.lrange('alerts', 0, n-1)
    alerts = []
    for item in raw:
        try:
            alerts.append(json.loads(item))
        except:
            pass
    return alerts

def get_system_stats(alerts):
    total    = len(alerts)
    l3_count = sum(1 for a in alerts if '3' in a.get('tier',''))
    l2_count = sum(1 for a in alerts if '2' in a.get('tier','') and '3' not in a.get('tier',''))
    l1_count = sum(1 for a in alerts if '1' in a.get('tier',''))
    confirmed_launder = sum(1 for a in alerts if a.get('is_laundering') == 1)
    gnn_assisted = sum(1 for a in alerts if a.get('has_gnn_embed', False))
    return total, l3_count, l2_count, l1_count, confirmed_launder, gnn_assisted

# ── Main content ──────────────────────────────────────────────────────────────
alerts = get_alerts(200)
total, l3, l2, l1, confirmed, gnn_assisted = get_system_stats(alerts)

# ── Metric row ────────────────────────────────────────────────────────────────
col1, col2, col3, col4, col5, col6 = st.columns(6)

with col1:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-value">{total}</div>
        <div class="metric-label">Total Alerts</div>
    </div>""", unsafe_allow_html=True)

with col2:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-value red">{l3}</div>
        <div class="metric-label">L3 Auto-Freeze</div>
    </div>""", unsafe_allow_html=True)

with col3:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-value amber">{l2}</div>
        <div class="metric-label">L2 Soft Hold</div>
    </div>""", unsafe_allow_html=True)

with col4:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-value green">{l1}</div>
        <div class="metric-label">L1 Review</div>
    </div>""", unsafe_allow_html=True)

with col5:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-value red">{confirmed}</div>
        <div class="metric-label">Confirmed Laundering</div>
    </div>""", unsafe_allow_html=True)

with col6:
    pct = int(gnn_assisted/total*100) if total > 0 else 0
    st.markdown(f"""<div class="metric-card">
        <div class="metric-value">{pct}%</div>
        <div class="metric-label">GNN-Assisted</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Two-column layout ─────────────────────────────────────────────────────────
left_col, right_col = st.columns([3, 2])

# ── LEFT: Alert queue ─────────────────────────────────────────────────────────
with left_col:
    st.markdown('<div class="section-title">Alert Queue — Most Recent First</div>', unsafe_allow_html=True)

    if not alerts:
        st.info("No alerts yet. Start the producer and consumer to begin detection.")
    else:
        # Build display dataframe
        rows = []
        for a in alerts[:50]:
            tier = a.get('tier', 'CLEAR')
            if 'L3' in tier or '3' in tier:
                badge = '<span class="badge-l3">L3</span>'
            elif 'L2' in tier or '2' in tier:
                badge = '<span class="badge-l2">L2</span>'
            elif 'L1' in tier or '1' in tier:
                badge = '<span class="badge-l1">L1</span>'
            else:
                badge = '<span class="badge-clear">CLR</span>'

            ts = a.get('timestamp','')[:19].replace('T',' ')
            rows.append({
                'Time':        ts,
                'Account':     a.get('account_id','')[:16],
                'Amount':      f"Rs. {a.get('amount',0):>14,.2f}",
                'Score':       f"{a.get('probability',0):.4f}",
                'Tier':        badge,
                'Txn Count':   a.get('total_txn', '-'),
                'Fan-Out':     a.get('fan_out', '-'),
                'Cross-Bank':  a.get('cross_bank', '-'),
                'GNN':         'Yes' if a.get('has_gnn_embed') else 'No',
                'Confirmed':   'YES' if a.get('is_laundering') == 1 else '-'
            })

        df_display = pd.DataFrame(rows)
        st.write(
            df_display.to_html(escape=False, index=False),
            unsafe_allow_html=True
        )

# ── RIGHT: Charts and SHAP ────────────────────────────────────────────────────
with right_col:

    # Score distribution chart
    st.markdown('<div class="section-title">Score Distribution</div>', unsafe_allow_html=True)
    if alerts:
        probs = [a.get('probability', 0) for a in alerts]
        fig, ax = plt.subplots(figsize=(5, 2.5))
        ax.hist(probs, bins=20, color='#1A2B4A', edgecolor='white', linewidth=0.5)
        ax.axvline(x=0.30, color='#276749', linestyle='--', linewidth=1, label='L1 (0.30)')
        ax.axvline(x=0.50, color='#B7791F', linestyle='--', linewidth=1, label='L2 (0.50)')
        ax.axvline(x=0.85, color='#C53030', linestyle='--', linewidth=1, label='L3 (0.85)')
        ax.set_xlabel('Fraud Probability', fontsize=8)
        ax.set_ylabel('Count', fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=7)
        ax.set_facecolor('#F8F9FA')
        fig.patch.set_facecolor('#F8F9FA')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    else:
        st.write("No data yet.")

    st.markdown("<br>", unsafe_allow_html=True)

    # Tier breakdown pie chart
    st.markdown('<div class="section-title">Alert Tier Breakdown</div>', unsafe_allow_html=True)
    if total > 0:
        fig2, ax2 = plt.subplots(figsize=(4, 2.5))
        sizes  = [l3, l2, l1]
        labels = ['L3 Auto-Freeze', 'L2 Soft Hold', 'L1 Review']
        colors = ['#C53030', '#B7791F', '#276749']
        non_zero = [(s,l,c) for s,l,c in zip(sizes,labels,colors) if s > 0]
        if non_zero:
            s,l,c = zip(*non_zero)
            ax2.pie(s, labels=l, colors=c, autopct='%1.0f%%',
                    textprops={'fontsize':7}, startangle=90)
        ax2.set_facecolor('#F8F9FA')
        fig2.patch.set_facecolor('#F8F9FA')
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()

    st.markdown("<br>", unsafe_allow_html=True)

    # Top flagged accounts
    st.markdown('<div class="section-title">Most Flagged Accounts</div>', unsafe_allow_html=True)
    if alerts:
        account_counts = {}
        account_scores = {}
        for a in alerts:
            aid = a.get('account_id','')
            account_counts[aid] = account_counts.get(aid, 0) + 1
            account_scores[aid] = max(account_scores.get(aid, 0), a.get('probability', 0))

        top_accounts = sorted(account_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        for acc, count in top_accounts:
            score = account_scores[acc]
            st.markdown(
                f"**{acc[:16]}** &nbsp; {count} alerts &nbsp; "
                f"max score: `{score:.4f}`"
            )

# ── SHAP Explanation section ──────────────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="section-title">SHAP Decision Explanation — Select Account to Investigate</div>',
            unsafe_allow_html=True)

if alerts:
    account_ids = list(dict.fromkeys([a['account_id'] for a in alerts]))
    selected    = st.selectbox("Account ID", account_ids, label_visibility="collapsed")

    selected_alerts = [a for a in alerts if a['account_id'] == selected]
    if selected_alerts:
        best_alert = max(selected_alerts, key=lambda x: x.get('probability', 0))

        exp_col1, exp_col2 = st.columns([1, 2])

        with exp_col1:
            st.markdown("**Account Summary**")
            st.markdown(f"Account ID: `{selected}`")
            st.markdown(f"Alert count: {len(selected_alerts)}")
            st.markdown(f"Max probability: `{best_alert.get('probability',0):.4f}`")
            st.markdown(f"Risk tier: **{best_alert.get('tier','—')}**")
            st.markdown(f"Total transactions: {best_alert.get('total_txn','—')}")
            st.markdown(f"Fan-out degree: {best_alert.get('fan_out','—')}")
            st.markdown(f"Cross-bank degree: {best_alert.get('cross_bank','—')}")
            st.markdown(f"GNN embedding: {'Present' if best_alert.get('has_gnn_embed') else 'Not found'}")
            st.markdown(f"Confirmed laundering: {'YES' if best_alert.get('is_laundering')==1 else 'No'}")

            # STR button
            if st.button("Generate STR Report", type="primary"):
                str_content = f"""SUSPICIOUS TRANSACTION REPORT
FIU-IND FORMAT — PMLA 2002

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Report Type: Automated Detection — Mule Account

ACCOUNT DETAILS
Account ID:        {selected}
Risk Tier:         {best_alert.get('tier','—')}
Fraud Probability: {best_alert.get('probability',0):.4f}
Total Alerts:      {len(selected_alerts)}

BEHAVIORAL INDICATORS
Total Transactions: {best_alert.get('total_txn','—')}
Fan-Out Degree:     {best_alert.get('fan_out','—')} unique destinations
Cross-Bank Degree:  {best_alert.get('cross_bank','—')} unique banks
Max Amount:         Rs. {best_alert.get('amount',0):,.2f}
GNN Graph Context:  {'Present — account found in transaction graph' if best_alert.get('has_gnn_embed') else 'Not in graph'}

DETECTION METHOD
Phase 1: Rule Engine — PMLA-aligned thresholds
Phase 2: XGBoost — 12 behavioral features
Phase 4: GNN EdgeProp — 64-dimensional graph embedding
Hybrid Score: {best_alert.get('probability',0):.4f} (threshold: {config['optimal_threshold']:.4f})

REGULATORY BASIS
Prevention of Money Laundering Act 2002
RBI Master Direction on KYC
RBI Governor Directive July 2024
RBI FREE-AI Framework August 2025

RECOMMENDED ACTION
{'Immediate account freeze and STR filing with FIU-IND' if best_alert.get('probability',0) >= 0.85 else 'Analyst review and enhanced due diligence'}
"""
                st.download_button(
                    label="Download STR",
                    data=str_content,
                    file_name=f"STR_{selected}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )

        with exp_col2:
            st.markdown("**SHAP Feature Attribution**")
            st.caption("Shows which features drove the fraud score for this account")

            # Build feature vector for SHAP
            acct_state = r.hgetall(f"acct:{selected}")
            if acct_state:
                total_txn    = int(acct_state.get('total_txn', 1))
                sum_amount   = float(acct_state.get('sum_amount', 0))
                max_amount   = float(acct_state.get('max_amount', 0))
                sum_sq       = float(acct_state.get('sum_sq_amount', 0))
                unique_dest  = set(acct_state.get('unique_dest','').split(','))
                unique_banks = set(acct_state.get('unique_banks','').split(','))
                unique_dest.discard('')
                unique_banks.discard('')
                round_count  = int(acct_state.get('round_count', 0))
                avg_amount   = sum_amount / total_txn if total_txn > 0 else 0

                import math
                velocity_ratio    = min(best_alert.get('amount',0) / (max_amount + 1e-9), 1.0)
                balance_retention = max(0.0, 1.0 - velocity_ratio)
                round_ratio       = round_count / total_txn if total_txn > 0 else 0.0

                if total_txn > 1:
                    mean_sq = sum_sq / total_txn
                    var     = mean_sq - avg_amount**2
                    std     = math.sqrt(max(var, 0))
                    cv      = std / (avg_amount + 1e-9)
                    amt_var = 1.0 / (1.0 + cv)
                else:
                    amt_var = 1.0

                behavioral = [
                    velocity_ratio,
                    balance_retention,
                    float(len(unique_dest)),
                    float(len(unique_banks)),
                    round_ratio,
                    round_ratio * 0.5,
                    amt_var,
                    min(best_alert.get('amount',0) / (avg_amount * 5 + 1e-9), 1.0),
                    max(0.0, 1.0 - min(total_txn/10.0, 1.0)),
                    float(total_txn),
                    avg_amount,
                    max_amount,
                ]

                emb_json = r.get(f"embed:{selected}")
                gnn_emb  = json.loads(emb_json) if emb_json else [0.0]*64
                features = behavioral + gnn_emb

                X        = np.array(features).reshape(1,-1)
                X_scaled = scaler.transform(X)
                shap_vals = explainer.shap_values(X_scaled)[0]

                # Show top behavioral features only (first 12)
                behavioral_labels = [
                    'Velocity Ratio','Balance Retention','Fan-Out Degree',
                    'Cross-Bank Degree','Round Amount Ratio','Structuring Score',
                    'Amount Variance','Session Burst','Dormancy Ratio',
                    'Total Transactions','Avg Amount','Max Amount'
                ]
                shap_behavioral = shap_vals[:12]
                sorted_idx = np.argsort(np.abs(shap_behavioral))[::-1][:8]

                fig3, ax3 = plt.subplots(figsize=(6, 3.5))
                colors_bar = ['#C53030' if v > 0 else '#276749' for v in shap_behavioral[sorted_idx]]
                bars = ax3.barh(
                    [behavioral_labels[i] for i in sorted_idx],
                    shap_behavioral[sorted_idx],
                    color=colors_bar, edgecolor='white', linewidth=0.5
                )
                ax3.axvline(x=0, color='#4A5568', linewidth=0.8)
                ax3.set_xlabel('SHAP Value (positive = increases fraud risk)', fontsize=7)
                ax3.tick_params(labelsize=7)
                ax3.set_facecolor('#F8F9FA')
                fig3.patch.set_facecolor('#F8F9FA')
                red_patch   = mpatches.Patch(color='#C53030', label='Increases fraud risk')
                green_patch = mpatches.Patch(color='#276749', label='Decreases fraud risk')
                ax3.legend(handles=[red_patch, green_patch], fontsize=7, loc='lower right')
                plt.tight_layout()
                st.pyplot(fig3)
                plt.close()
            else:
                st.info("Account state not found in Redis. Process more transactions first.")

# ── Auto refresh ──────────────────────────────────────────────────────────────
if auto_refresh:
    st.markdown(
        f"<div style='text-align:right;font-size:0.7rem;color:#718096;"
        f"margin-top:1rem'>Last updated: {datetime.now().strftime('%H:%M:%S')} "
        f"— refreshing every {refresh_rate}s</div>",
        unsafe_allow_html=True
    )
    time.sleep(refresh_rate)
    st.rerun()