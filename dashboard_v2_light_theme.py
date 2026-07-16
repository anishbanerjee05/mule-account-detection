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
import requests
import warnings
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════════════════
# NEW LIGHT-THEME UI — built from the reviewed UI/UX mockup
# (MuleDetection_UIUX_Mockup.html).
# ═══════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Mule Account Detection — CyberPeace Foundation",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

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
    [data-testid="stAppViewContainer"] * , .block-container * { color: var(--ink); }
    .main { background-color: var(--paper); }
    .block-container { padding: 1rem 1.5rem 2rem 1.5rem; max-width: 100%; }
    h1, h2, h3, .page-title { font-family: 'Source Serif 4', serif !important; color: var(--ink); }
    section[data-testid="stSidebar"] { background: var(--navy-deep); }
    section[data-testid="stSidebar"] * { color: #DCE4EA !important; }
    section[data-testid="stSidebar"] .stButton button {
        width: 100%; text-align: left; background: transparent; border: none;
        color: #B9C6D0 !important; font-size: 0.85rem; padding: 0.5rem 0.7rem;
        border-radius: 2px; font-family: 'IBM Plex Sans', sans-serif;
    }
    section[data-testid="stSidebar"] .stButton button:hover { background: rgba(255,255,255,0.06); color: #fff !important; }
    section[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.12) !important; }
    .status-pill {
        display:inline-flex; align-items:center; gap:7px; font-size:0.78rem;
        padding:5px 11px; border:1px solid var(--line-strong); border-radius:2px; background:#F0FBF3;
    }
    .status-dot { width:7px; height:7px; border-radius:50%; }
    .status-dot.live { background: var(--safe); }
    .status-dot.warn { background: var(--warn); }
    .status-dot.stale { background: var(--danger); }
    .kpi { background:var(--paper-raised); border:1px solid var(--line); border-top:3px solid var(--navy);
           padding:0.9rem 1rem; border-radius:2px; text-align:left; }
    .kpi.danger { border-top-color: var(--danger); }
    .kpi.warn   { border-top-color: var(--warn); }
    .kpi.safe   { border-top-color: var(--safe); }
    .kpi .val   { font-family:'IBM Plex Mono', monospace; font-size:1.7rem; font-weight:600; color:var(--ink); }
    .kpi .lbl   { font-size:0.65rem; text-transform:uppercase; letter-spacing:0.08em; color:var(--ink-soft); margin-top:4px; }
    .panel-head { font-family:'Source Serif 4', serif; font-size:0.95rem; font-weight:600;
                  color:var(--ink); border-bottom:1px solid var(--line); padding-bottom:0.4rem; margin-bottom:0.7rem; }
    .ledger-table { width:100%; border-collapse:collapse; font-size:0.78rem; }
    .ledger-table th {
        text-align:left; font-size:0.62rem; text-transform:uppercase; letter-spacing:0.06em;
        color:var(--ink-soft); padding:0.5rem 0.7rem; border-bottom:2px solid var(--ink); font-weight:600;
    }
    .ledger-table td { padding:0.55rem 0.7rem; border-bottom:1px solid var(--line); font-family:'IBM Plex Mono', monospace; color:var(--ink); }
    .ledger-table tr:hover td { background:#FAFAF8; }
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
</style>
""", unsafe_allow_html=True)

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

PAGES = ["Dashboard", "Live Monitoring", "Flagged Accounts", "Risk Analysis",
         "Hop Map", "Reports", "Settings"]
NOT_YET_BUILT = set(PAGES) - {"Dashboard", "Hop Map", "Live Monitoring"}

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
    st.caption("v0.3 — UI redesign in progress\nPages beyond Dashboard/Hop Map/Live Monitoring are placeholders")

page = st.session_state.page

nav_cols = st.columns(len(PAGES))
for i, pg in enumerate(PAGES):
    with nav_cols[i]:
        if st.button(pg, key=f"topnav_{pg}", use_container_width=True,
                     type="primary" if page == pg else "secondary"):
            st.session_state.page = pg
            st.rerun()
st.markdown("<hr style='margin:0.5rem 0 1rem 0; border-color:var(--line);'>", unsafe_allow_html=True)

alerts = get_alerts(1000)
status_label, status_cls = get_system_status(alerts)
top_l, top_r = st.columns([4, 1])
with top_l:
    st.markdown(f'<div class="page-title" style="font-size:1.4rem; font-weight:600;">{page}</div>', unsafe_allow_html=True)
with top_r:
    st.markdown(f'<div class="status-pill"><span class="status-dot {status_cls}"></span>{status_label}</div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

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

elif page == "Live Monitoring":
    left, right = st.columns([3, 2])

    with left:
        st.markdown('<div class="panel-head">Live Transaction Feed</div>', unsafe_allow_html=True)
        if not alerts:
            st.markdown('<div class="placeholder-panel">No alerts yet. Start the producer and consumer to see live traffic.</div>', unsafe_allow_html=True)
        else:
            rows = ""
            for a in alerts[:40]:
                aid = a.get('account_id', '')
                tier = a.get('tier', 'CLEAR')
                prob = a.get('probability', 0)
                ts = a.get('timestamp', '')[:19].replace('T', ' ')
                gnn = 'Yes' if a.get('has_gnn_embed') else 'No'
                cep = get_cep_badge_html(aid)
                rows += (f"<tr><td>{ts}</td><td>{aid[:14]}</td>"
                         f"<td style='text-align:right'>{a.get('amount', 0):,.0f}</td>"
                         f"<td style='text-align:right'>{prob:.4f}</td>"
                         f"<td>{stamp_html(tier)}</td><td>{cep}</td><td>{gnn}</td></tr>")
            st.markdown(
                f"""<table class="ledger-table">
                <tr><th>Time</th><th>Account</th><th style="text-align:right">Amount</th>
                <th style="text-align:right">Score</th><th>Tier</th><th>Pattern</th><th>GNN</th></tr>
                {rows}</table>""", unsafe_allow_html=True)
            st.markdown(f"<div style='font-size:0.68rem; color:var(--ink-soft); margin-top:0.5rem;'>"
                        f"Showing the {min(40, len(alerts))} most recent entries from a {len(alerts)}-entry buffer.</div>",
                        unsafe_allow_html=True)

    with right:
        st.markdown('<div class="panel-head">Pipeline Health</div>', unsafe_allow_html=True)

        import subprocess
        lag_display = "Unavailable"
        partition_display = "Unavailable"
        try:
            result = subprocess.run(
                ["docker", "exec", "kafka", "kafka-consumer-groups",
                 "--bootstrap-server", "localhost:9092",
                 "--describe", "--group", "mule-detector-prod"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                lines = [l for l in result.stdout.splitlines() if l.strip() and not l.startswith("GROUP")]
                active_rows = [l for l in lines if "rdkafka-" in l]
                if active_rows:
                    total_lag = sum(int(l.split()[4]) for l in active_rows if l.split()[4].lstrip('-').isdigit())
                    partition_display = f"{len(active_rows)} / 4 assigned"
                    lag_display = f"{total_lag:,}"
        except Exception:
            pass

        st.markdown(f'<div class="kpi"><div class="val">{lag_display}</div><div class="lbl">Consumer Lag (Active Topic)</div></div>',
                    unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f'<div class="kpi"><div class="val">{partition_display}</div><div class="lbl">Kafka Partitions Assigned</div></div>',
                    unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        gnn_status, gnn_cls = "Unknown", "stale"
        try:
            gnn_resp = requests.get("http://localhost:8002/stats", timeout=2)
            if gnn_resp.status_code == 200:
                gnn_data = gnn_resp.json()
                if gnn_data.get("persistent_store_loaded"):
                    gnn_status, gnn_cls = "Healthy", "live"
                else:
                    gnn_status, gnn_cls = "Loaded, store not ready", "warn"
        except requests.exceptions.RequestException:
            gnn_status, gnn_cls = "Unreachable", "stale"
        st.markdown(f'<div class="kpi {"safe" if gnn_cls=="live" else ("warn" if gnn_cls=="warn" else "danger")}">'
                    f'<div class="val">{gnn_status}</div><div class="lbl">GNN Service</div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.caption("Consumer lag and partition data reads directly from Kafka via `docker exec` -- if Docker isn't reachable this will show 'Unavailable', not a guessed number.")

elif page == "Hop Map":
    API_BASE_URL = "http://localhost:8000"

    st.markdown('<div class="panel-head">Trace Money-Flow Network</div>', unsafe_allow_html=True)
    in_col, depth_col, btn_col = st.columns([3, 1, 1])
    with in_col:
        hop_account = st.text_input("Account ID", value="", placeholder="e.g. 10042B660",
                                     label_visibility="collapsed")
    with depth_col:
        hop_depth_input = st.number_input("Max depth", min_value=1, max_value=5, value=3,
                                           label_visibility="collapsed")
    with btn_col:
        fetch_clicked = st.button("Trace Network", type="primary", use_container_width=True)

    cluster_toggle = st.checkbox("Cluster low-signal leaf accounts", value=True,
                                  help="Groups outer-ring accounts with no risk score into a single marker, instead of drawing each one separately.")

    with st.expander("Path mode — trace the route between two specific accounts"):
        path_col1, path_col2 = st.columns(2)
        with path_col1:
            path_target = st.text_input("Target account ID", value="", key="path_target")
        st.caption("Finds a path from the traced account above to this target, using only the network already loaded. If no path is found, increase max depth and re-trace first.")

    if "hop_map_data" not in st.session_state:
        st.session_state.hop_map_data = None

    if fetch_clicked and hop_account.strip():
        try:
            resp = requests.get(
                f"{API_BASE_URL}/hops/{hop_account.strip()}",
                params={"max_depth": int(hop_depth_input), "max_nodes": 50},
                timeout=8
            )
            if resp.status_code == 200:
                st.session_state.hop_map_data = resp.json()
            elif resp.status_code == 503:
                st.session_state.hop_map_data = None
                st.error("Redis not available (api.py reports 503). Is the backend running?")
            else:
                st.session_state.hop_map_data = None
                st.error(f"API returned {resp.status_code}: {resp.text[:200]}")
        except requests.exceptions.ConnectionError:
            st.session_state.hop_map_data = None
            st.error("Could not reach api.py at localhost:8000. Confirm it's running: `python api.py`")
        except requests.exceptions.Timeout:
            st.session_state.hop_map_data = None
            st.error("Request to /hops timed out after 8s.")
        except Exception as e:
            st.session_state.hop_map_data = None
            st.error(f"Unexpected error: {e}")

    data = st.session_state.hop_map_data

    if not data:
        st.markdown(
            '<div class="placeholder-panel">Enter an account ID and click Trace Network '
            'to visualize its money-flow graph.</div>',
            unsafe_allow_html=True
        )
    else:
        nodes = data.get("nodes", [])
        edges = data.get("edges", [])
        summary = data.get("summary", {})

        root_id = hop_account.strip()
        direct_connections = len({
            (e['to_account'] if e['from_account'] == root_id else e['from_account'])
            for e in edges if e['from_account'] == root_id or e['to_account'] == root_id
        })

        s1, s2, s3, s4, s5 = st.columns(5)
        def hop_kpi(col, value, label, style="neutral"):
            col.markdown(f'<div class="kpi {style}"><div class="val">{value}</div><div class="lbl">{label}</div></div>',
                        unsafe_allow_html=True)
        hop_kpi(s1, data.get("total_nodes", 0), "Nodes Traced")
        hop_kpi(s2, direct_connections, "Direct Connections")
        hop_kpi(s3, summary.get("bank_count", 0), "Banks Involved")
        hop_kpi(s4, summary.get("confirmed_fraud_nodes", 0), "Confirmed Fraud", "danger")
        hop_kpi(s5, f"Rs.{summary.get('total_amount_moved', 0):,.0f}", "Total Moved")

        st.markdown(f"<div style='font-size:0.75rem; color:var(--ink-soft); margin:0.5rem 0 1rem 0;'>"
                    f"{summary.get('investigation_note', '')}</div>", unsafe_allow_html=True)

        def tier_color(tier):
            if not tier: return '#9AA7B2'
            if 'L3' in tier: return '#8C2F2F'
            if 'L2' in tier: return '#9C6B1D'
            if 'L1' in tier: return '#3F6B4A'
            return '#5D7A8C'

        @st.cache_data(ttl=60)
        def get_top_pattern(account_id):
            try:
                raw = r.lindex(f"cep_alerts:{account_id}", 0)
                if raw:
                    return json.loads(raw).get('pattern_type', '')
            except Exception:
                pass
            return ''

        node_patterns = {n['account_id']: get_top_pattern(n['account_id']) for n in nodes}

        cluster_map = {}
        clustered_ids = set()
        real_nodes = nodes
        if cluster_toggle and nodes:
            max_depth_present = max(n['hop_depth'] for n in nodes)
            candidates = [n for n in nodes
                          if n['hop_depth'] == max_depth_present
                          and n.get('risk_score') is None
                          and not n.get('is_confirmed_fraud')]
            if len(candidates) >= 4:
                cluster_id = f"__cluster_{max_depth_present}"
                for n in candidates:
                    cluster_map[n['account_id']] = cluster_id
                    clustered_ids.add(n['account_id'])
                real_nodes = [n for n in nodes if n['account_id'] not in clustered_ids]

        by_depth = {}
        for n in real_nodes:
            by_depth.setdefault(n['hop_depth'], []).append(n)
        if cluster_map:
            cluster_depth = max(n['hop_depth'] for n in nodes)
            by_depth.setdefault(cluster_depth, []).append({
                'account_id': list(cluster_map.values())[0],
                'hop_depth': cluster_depth,
                '__is_cluster': True,
                '__cluster_count': len(clustered_ids),
            })

        MIN_ARC_SPACING = 1.15
        BASE_RING_GAP = 2.2
        positions = {}
        prev_radius = 0.0
        for depth in sorted(by_depth.keys()):
            group = by_depth[depth]
            count = len(group)
            radius = 0.0 if depth == 0 else max(prev_radius + BASE_RING_GAP, (count * MIN_ARC_SPACING) / (2 * math.pi))
            prev_radius = radius
            for i, n in enumerate(group):
                angle = (2 * math.pi * i / count) if count > 0 else 0
                positions[n['account_id']] = (radius * math.cos(angle), radius * math.sin(angle))

        for aid, cid in cluster_map.items():
            positions[aid] = positions.get(cid, (0, 0))

        max_radius = max((abs(v) for xy in positions.values() for v in xy), default=5) + 1.5

        def find_cycle_edges(node_ids, edge_list):
            adj = {aid: [] for aid in node_ids}
            for e in edge_list:
                fa, ta = e['from_account'], e['to_account']
                if fa in adj and ta in adj:
                    adj[fa].append(ta)
            visited, rec_stack, path_stack = set(), set(), []
            cyc_edges = set()
            def dfs(u):
                visited.add(u); rec_stack.add(u); path_stack.append(u)
                for v in adj.get(u, []):
                    if v not in visited:
                        dfs(v)
                    elif v in rec_stack and v in path_stack:
                        idx = path_stack.index(v)
                        for i in range(idx, len(path_stack) - 1):
                            cyc_edges.add((path_stack[i], path_stack[i + 1]))
                        cyc_edges.add((path_stack[-1], v))
                rec_stack.discard(u); path_stack.pop()
            for aid in adj:
                if aid not in visited:
                    dfs(aid)
            return cyc_edges

        all_ids = [n['account_id'] for n in nodes]
        cycle_edges = find_cycle_edges(all_ids, edges)

        path_edges = set()
        path_found = None
        if path_target.strip():
            adj = {}
            for e in edges:
                adj.setdefault(e['from_account'], []).append(e['to_account'])
            start, goal = hop_account.strip(), path_target.strip()
            queue, seen_bfs, parent = [start], {start}, {}
            while queue:
                cur = queue.pop(0)
                if cur == goal:
                    break
                for nxt in adj.get(cur, []):
                    if nxt not in seen_bfs:
                        seen_bfs.add(nxt); parent[nxt] = cur; queue.append(nxt)
            if goal in parent or goal == start:
                path = [goal]
                while path[-1] != start:
                    path.append(parent[path[-1]])
                path.reverse()
                path_found = path
                for i in range(len(path) - 1):
                    path_edges.add((path[i], path[i + 1]))
            else:
                st.warning(f"No path found from {start} to {goal} within the currently loaded {len(nodes)}-node network. Try increasing max depth and re-tracing.")

        fig, ax = plt.subplots(figsize=(10, 9))
        ax.set_facecolor('#FAFAF8')
        fig.patch.set_facecolor('white')

        for e in edges:
            fa, ta = e['from_account'], e['to_account']
            fa_pos = positions.get(cluster_map.get(fa, fa))
            ta_pos = positions.get(cluster_map.get(ta, ta))
            if fa_pos and ta_pos:
                amt = max(e.get('amount', 0), 1)
                width = min(4.0, max(0.5, math.log10(amt + 1) / 2.2))
                is_cycle = (fa, ta) in cycle_edges
                is_path = (fa, ta) in path_edges
                color = '#9C6B1D' if is_cycle else ('#1B3A63' if is_path else '#B9C2C9')
                alpha = 0.9 if (is_cycle or is_path) else 0.35
                zorder = 5 if (is_cycle or is_path) else 2
                ax.annotate('', xy=ta_pos, xytext=fa_pos,
                            arrowprops=dict(arrowstyle='-|>', color=color, lw=width,
                                           alpha=alpha, shrinkA=8, shrinkB=8), zorder=zorder)

        def should_label(n):
            if n.get('__is_cluster'):
                return True
            if n['hop_depth'] == 0:
                return True
            if n.get('is_confirmed_fraud'):
                return True
            tier = n.get('risk_tier') or ''
            return 'L2' in tier or 'L3' in tier

        for depth, group in by_depth.items():
            for n in group:
                x, y = positions[n['account_id']]
                if n.get('__is_cluster'):
                    ax.scatter([x], [y], s=420, c='#D8DEE2', edgecolors='#9AA7B2',
                              linewidths=1.2, zorder=3, marker='o')
                    ax.annotate(f"+{n['__cluster_count']}", (x, y), fontsize=9, ha='center',
                               va='center', color='#4B5A66', fontweight='bold', zorder=4)
                    continue
                color = tier_color(n.get('risk_tier'))
                size = 300 if n['hop_depth'] == 0 else 130
                edgecolor = '#8C2F2F' if n.get('is_confirmed_fraud') else 'white'
                edgewidth = 2.2 if n.get('is_confirmed_fraud') else 0.7
                marker_shape = 's' if n['account_id'] in [ee for pair in path_edges for ee in pair] else 'o'
                ax.scatter([x], [y], s=size, c=color, edgecolors=edgecolor, linewidths=edgewidth,
                          zorder=3, marker=marker_shape)
                if should_label(n):
                    label = n['account_id'][:9]
                    bank = n.get('bank', '')
                    if bank and bank != 'Unknown':
                        label += f"\nbank:{bank[:8]}"
                    pattern = node_patterns.get(n['account_id'], '')
                    if pattern:
                        label += f"\n{pattern[:12]}"
                    ax.annotate(label, (x, y), fontsize=6.2, ha='center', va='center',
                               xytext=(0, -13 if n['hop_depth'] > 0 else -16),
                               textcoords='offset points', color='#14202B',
                               fontfamily='monospace', zorder=4, linespacing=1.4)

        edge_amount_positions = []
        for e in edges:
            fa, ta = e['from_account'], e['to_account']
            fa_pos = positions.get(cluster_map.get(fa, fa))
            ta_pos = positions.get(cluster_map.get(ta, ta))
            if fa_pos and ta_pos:
                edge_amount_positions.append((e.get('amount', 0), fa_pos, ta_pos))
        edge_amount_positions.sort(key=lambda t: -t[0])
        for amt, fa_pos, ta_pos in edge_amount_positions[:5]:
            mx, my = (fa_pos[0] + ta_pos[0]) / 2, (fa_pos[1] + ta_pos[1]) / 2
            ax.annotate(f"Rs.{amt:,.0f}", (mx, my), fontsize=6, color='#1B3A63',
                       ha='center', va='center', fontweight='bold', zorder=6,
                       bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='#D8DEE2', lw=0.4))

        ax.set_xlim(-max_radius, max_radius)
        ax.set_ylim(-max_radius, max_radius)
        ax.set_aspect('equal')
        ax.axis('off')
        legend_handles = [
            mpatches.Patch(color='#8C2F2F', label='L3 — Auto Freeze'),
            mpatches.Patch(color='#9C6B1D', label='L2 — Soft Hold'),
            mpatches.Patch(color='#3F6B4A', label='L1 — Review'),
            mpatches.Patch(color='#5D7A8C', label='CLEAR'),
            mpatches.Patch(color='#9AA7B2', label='No score data'),
        ]
        if cycle_edges:
            legend_handles.append(mpatches.Patch(color='#9C6B1D', label='Detected cycle'))
        if path_edges:
            legend_handles.append(mpatches.Patch(color='#1B3A63', label='Traced path'))
        ax.legend(handles=legend_handles, loc='upper left', fontsize=7, framealpha=0.9)

        st.pyplot(fig)
        plt.close()

        caption_bits = ["Accounts outlined in dark red are confirmed fraud.",
                        "Grey nodes have no risk score yet — receiver-only accounts.",
                        "Edge thickness reflects transaction amount (log scale)."]
        if cycle_edges:
            caption_bits.append("Gold edges are part of a detected cycle in this network.")
        if path_edges:
            caption_bits.append(f"Navy squares/edges trace the path to {path_target.strip()}.")
        if cluster_map:
            caption_bits.append(f"{len(clustered_ids)} low-signal leaf accounts collapsed into one marker — uncheck the box above to see them individually.")
        st.markdown(f"<div style='font-size:0.68rem; color:var(--ink-soft); margin:-0.5rem 0 1rem 0;'>"
                    f"{' '.join(caption_bits)}</div>", unsafe_allow_html=True)

        st.markdown('<div class="panel-head">Node Detail</div>', unsafe_allow_html=True)
        sorted_nodes = sorted(nodes, key=lambda n: (n.get('risk_score') is None, -(n.get('risk_score') or 0)))
        rows = ""
        for n in sorted_nodes[:30]:
            score = n.get('risk_score')
            score_str = f"{score:.4f}" if score is not None else "—"
            conf = '<span style="color:var(--danger); font-weight:700;">YES</span>' if n.get('is_confirmed_fraud') else '—'
            pattern = node_patterns.get(n['account_id'], '') or '—'
            rows += (f"<tr><td>{n['account_id'][:14]}</td><td style='text-align:right'>{score_str}</td>"
                     f"<td>{n.get('risk_tier') or '—'}</td><td>{pattern}</td>"
                     f"<td style='text-align:right'>{n['hop_depth']}</td><td>{conf}</td></tr>")
        st.markdown(
            f"""<table class="ledger-table">
            <tr><th>Account</th><th style="text-align:right">Risk Score</th><th>Tier</th><th>Pattern</th><th style="text-align:right">Hop Depth</th><th>Confirmed</th></tr>
            {rows}</table>""", unsafe_allow_html=True
        )

elif page in NOT_YET_BUILT:
    st.markdown(
        f"""<div class="placeholder-panel">
        <div style="font-family:'Source Serif 4', serif; font-size:1.1rem;">{page} — not yet implemented</div>
        <div style="font-size:0.8rem; margin-top:0.5rem;">This page exists in the reviewed UI mockup. Real data wiring
        is built one page at a time — tell me when to build this one.</div>
        </div>""", unsafe_allow_html=True
    )

st.markdown(
    f"<div style='text-align:right; font-size:0.65rem; color:var(--ink-soft); margin-top:2rem; "
    f"padding-top:0.5rem; border-top:1px solid var(--line);'>"
    f"Last updated: {datetime.now().strftime('%H:%M:%S')} &nbsp;|&nbsp; Mule Account Detection System v0.3 (UI redesign)"
    f"</div>", unsafe_allow_html=True
)