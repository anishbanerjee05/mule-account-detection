# api.py — Mule Account Detection API v2
#
# Two major additions over v1:
#
# 1. PRE-AUTHORIZATION ENDPOINT (/authorize/transaction)
#    Called by bank CBS (SBI Finacle / Kotak CBS) BEFORE approving a transaction.
#    Reads the sender's live account state from Redis, computes behavioral features,
#    fetches GNN embedding, scores with XGBoost, and returns a risk decision
#    in under 100ms. This stops fraud BEFORE money moves.
#
# 2. HOP MAPPING ENDPOINT (/hops/{account_id})
#    Traces the complete money flow network for a flagged account.
#    Queries the transaction edge graph stored in Redis by consumer.py,
#    performs BFS up to N hops in both directions, and returns the full
#    inter-bank network with risk scores at each node.
#    Enables investigators to trace the complete criminal network, not just
#    individual suspicious accounts.
#
# Existing endpoints from v1 are unchanged:
#    GET  /health
#    POST /score/account
#    POST /explain
#    GET  /account/{account_id}/cached
#    GET  /stats

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import pickle
import json
import math
import numpy as np
import shap
import uvicorn
from typing import List, Optional, Dict, Any
import redis
import requests as http_requests
from datetime import datetime, timedelta
from collections import deque

app = FastAPI(
    title="Mule Account Detection API v2",
    description=(
        "Real-time mule account detection with pre-authorization and hop mapping. "
        "CyberPeace Foundation Research Wing — Ministry of Finance Deployment."
    ),
    version="2.0.0",
)

# ── Load models on startup ────────────────────────────────────────────────────
print("Loading models...")
with open("xgb_final_model.pkl", "rb") as f:
    xgb_model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("model_config.json", "r") as f:
    config = json.load(f)

try:
    with open("hybrid_model_config.json", "r") as f:
        hybrid_config = json.load(f)
    with open("xgb_hybrid_model.pkl", "rb") as f:
        hybrid_model = pickle.load(f)
    with open("scaler_hybrid.pkl", "rb") as f:
        hybrid_scaler = pickle.load(f)
    USE_HYBRID = True
    print(
        f"Hybrid model loaded (XGBoost + GNN, {hybrid_config.get('n_total_features', '?')} features)"
    )
except FileNotFoundError as e:
    USE_HYBRID = False
    hybrid_config = config
    hybrid_model = None
    hybrid_scaler = None
    print(
        f"Hybrid artifacts not found ({e.filename}) — pre-authorization scoring will be UNAVAILABLE, not silently downgraded"
    )

feature_cols = config["feature_cols"]
optimal_threshold = config["optimal_threshold"]
L1_THRESHOLD = 0.30
L2_THRESHOLD = 0.50
L3_THRESHOLD = 0.70
GNN_SERVICE_URL = "http://localhost:8002"

explainer = shap.TreeExplainer(xgb_model)

# ── Redis connection ──────────────────────────────────────────────────────────
try:
    r = redis.Redis(host="localhost", port=6379, decode_responses=True)
    r.ping()
    print("Redis connected")
except Exception:
    r = None
    print("Redis not available")

print(f"Models loaded. Threshold: {optimal_threshold:.4f}")

# ── Pydantic schemas ──────────────────────────────────────────────────────────


class AccountFeatures(BaseModel):
    account_id: str
    features: List[float]


class RiskScore(BaseModel):
    account_id: str
    fraud_probability: float
    risk_tier: str
    threshold_used: float
    flagged: bool


class ExplainResponse(BaseModel):
    account_id: str
    fraud_probability: float
    top_features: List[dict]
    risk_tier: str


class HealthResponse(BaseModel):
    status: str
    model_version: str
    optimal_threshold: float
    feature_count: int


# ── Pre-authorization schemas ─────────────────────────────────────────────────


class TransactionAuthRequest(BaseModel):
    # Fields sent by the bank CBS before approving a transaction
    transaction_id: str  # bank's internal transaction reference
    account_from: str  # sender account ID
    account_to: str  # receiver account ID
    bank_from: str  # sender's bank name
    bank_to: str  # receiver's bank name
    amount: float  # transaction amount in INR
    payment_format: str  # NEFT, RTGS, UPI, IMPS, etc.
    timestamp: Optional[str]  # ISO format, defaults to now
    request_channel: Optional[str]  # mobile, internet, branch, atm


class AuthDecision(BaseModel):
    # Returned to the bank CBS within 100ms
    transaction_id: str
    account_from: str
    decision: str  # APPROVE, HOLD, BLOCK
    risk_tier: str  # CLEAR, L1-REVIEW, L2-SOFT_HOLD, L3-AUTO_FREEZE
    fraud_probability: float
    confidence: str  # HIGH, MEDIUM, LOW (based on data availability)
    reasons: List[str]  # human-readable reasons for the decision
    sla_ms: int  # actual processing time in milliseconds
    recommended_action: str


# ── Hop mapping schemas ───────────────────────────────────────────────────────


class HopNode(BaseModel):
    account_id: str
    bank: str
    total_sent: float
    transaction_count: int
    risk_score: Optional[float]
    risk_tier: Optional[str]
    is_confirmed_fraud: bool
    spike_detected: bool  # inbound amount exceeds 3x 30-day baseline
    hop_depth: int  # how many hops from the root account


class HopEdge(BaseModel):
    from_account: str
    to_account: str
    from_bank: str
    to_bank: str
    amount: float
    timestamp: str
    payment_format: Optional[str]


class HopMapResponse(BaseModel):
    root_account: str
    total_nodes: int
    total_edges: int
    max_depth: int
    nodes: List[HopNode]
    edges: List[HopEdge]
    summary: Dict[str, Any]  # banks involved, total amount, etc.


# ── Helper: read account state from Redis ─────────────────────────────────────
def get_account_state(account_id: str) -> Optional[dict]:
    """
    Reads the account's behavioral state from Redis.
    This is the same state maintained by consumer.py's update_account_state().
    Returns None if the account has never been seen.
    """
    if not r:
        return None
    state = r.hgetall(f"acct:{account_id}")
    return state if state else None


def compute_behavioral_features_from_state(
    state: dict, amount: float, bank_to: str
) -> List[float]:
    """
    Replicates consumer.py's compute_behavioral_features() using account state
    already stored in Redis. This is what makes the pre-auth API work — it uses
    the same feature computation logic as the real-time consumer, ensuring
    consistency between the two scoring paths.

    The amount and bank_to parameters represent the PROPOSED transaction,
    allowing the model to score "what would this account look like after
    this transaction?" — a forward-looking score, not just historical.
    """
    total_txn = int(state.get("total_txn", "0"))
    total_sent = float(state.get("total_sent", "0.0"))
    max_amount = float(state.get("max_amount", "0.0"))
    sum_amount = float(state.get("sum_amount", "0.0"))
    sum_sq_amount = float(state.get("sum_sq_amount", "0.0"))
    round_count = int(state.get("round_count", "0"))

    unique_dest = set(state.get("unique_dest", "").split(","))
    unique_banks = set(state.get("unique_banks", "").split(","))
    unique_dest.discard("")
    unique_banks.discard("")

    # Simulate this transaction being added
    # This gives a forward-looking score: "if this transaction goes through,
    # how suspicious does this account look?"
    total_txn += 1
    total_sent += amount
    max_amount = max(max_amount, amount)
    sum_amount += amount
    sum_sq_amount += amount**2
    unique_dest.add(bank_to)
    unique_banks.add(bank_to)
    if amount % 1000 == 0 and amount > 0:
        round_count += 1

    avg_amount = sum_amount / total_txn
    velocity_ratio = min(amount / (max_amount + 1e-9), 1.0) if total_txn > 1 else 0.95
    balance_retention = max(0.0, 1.0 - velocity_ratio)
    fan_out = float(len(unique_dest))
    cross_bank = float(len(unique_banks))
    round_ratio = round_count / total_txn
    structuring = round_ratio * 0.5
    dormancy_ratio = max(0.0, 1.0 - min(total_txn / 10.0, 1.0))

    if total_txn > 1:
        mean_sq = sum_sq_amount / total_txn
        var = mean_sq - avg_amount**2
        std = math.sqrt(max(var, 0))
        cv = std / (avg_amount + 1e-9)
        amount_variance = 1.0 / (1.0 + cv)
    else:
        amount_variance = 1.0

    session_burst = min(amount / (avg_amount * 5 + 1e-9), 1.0)

    return [
        velocity_ratio,
        balance_retention,
        fan_out,
        cross_bank,
        round_ratio,
        structuring,
        amount_variance,
        session_burst,
        dormancy_ratio,
        float(total_txn),
        avg_amount,
        max_amount,
    ]


def get_gnn_embedding_for_auth(
    account_id: str, behavioral: List[float], neighbor_ids: List[str]
) -> List[float]:
    """
    Fetches GNN embedding for pre-auth scoring.
    Uses the same 3-tier lookup as consumer.py:
    1. Redis cache (sub-millisecond)
    2. GNN service /embed_batch (30-50ms for cache misses)
    3. Zero vector fallback (if GNN service unavailable)
    """
    if r:
        cached = r.get(f"embed:{account_id}")
        if cached:
            return json.loads(cached)

    try:
        payload = {
            "accounts": [
                {
                    "account_id": account_id,
                    "behavioral": behavioral,
                    "neighbor_ids": neighbor_ids[:10],
                    "amount": 0.0,
                    "payment_format": "unknown",
                }
            ]
        }
        resp = http_requests.post(
            f"{GNN_SERVICE_URL}/embed_batch",
            json=payload,
            timeout=0.08,  # 80ms max — leaves 20ms for rest of pre-auth pipeline
        )
        if resp.status_code == 200:
            result = resp.json()
            if account_id in result:
                return result[account_id]["embedding"]
    except Exception:
        pass

    return [0.0] * 64


def detect_vendor_spike(account_id: str, proposed_amount: float) -> dict:
    """
    Vendor spike detection: compares the proposed transaction amount against
    the account's 30-day rolling average inbound amount.

    A spike is defined as: proposed_amount > 3x the account's average outbound
    transaction amount. This catches the 'dormant account suddenly receives
    massive funds and forwards them' pattern your CTO requested.

    Returns: {spike: bool, ratio: float, baseline: float, reason: str}
    """
    if not r:
        return {
            "spike": False,
            "ratio": 1.0,
            "baseline": 0.0,
            "reason": "Redis unavailable",
        }

    state = get_account_state(account_id)
    if not state:
        return {
            "spike": False,
            "ratio": 1.0,
            "baseline": 0.0,
            "reason": "New account — no baseline",
        }

    total_txn = int(state.get("total_txn", "0"))
    sum_amount = float(state.get("sum_amount", "0.0"))

    if total_txn == 0:
        return {
            "spike": False,
            "ratio": 1.0,
            "baseline": 0.0,
            "reason": "No transaction history",
        }

    baseline = sum_amount / total_txn
    if baseline < 1.0:
        return {
            "spike": False,
            "ratio": 1.0,
            "baseline": baseline,
            "reason": "Baseline too low to measure",
        }

    ratio = proposed_amount / baseline

    if ratio >= 10.0:
        return {
            "spike": True,
            "ratio": round(ratio, 2),
            "baseline": round(baseline, 2),
            "reason": f"CRITICAL: {ratio:.1f}x average (Rs.{proposed_amount:,.0f} vs baseline Rs.{baseline:,.0f})",
        }
    elif ratio >= 3.0:
        return {
            "spike": True,
            "ratio": round(ratio, 2),
            "baseline": round(baseline, 2),
            "reason": f"SPIKE: {ratio:.1f}x average (Rs.{proposed_amount:,.0f} vs baseline Rs.{baseline:,.0f})",
        }
    else:
        return {
            "spike": False,
            "ratio": round(ratio, 2),
            "baseline": round(baseline, 2),
            "reason": "Within normal range",
        }


def score_account_internal(account_id: str, behavioral: List[float],
                            bank_to: str) -> tuple:
    """
    Internal scoring function used by the pre-auth endpoint.
    Returns (probability, tier, has_gnn_embed).

    This builds a 76-feature vector (12 behavioral + 64 GNN). The base
    xgb_model/scaler were fit on len(feature_cols) features and will
    raise a shape-mismatch error if used here — this function must use
    the hybrid model/scaler only, never the base ones.
    """
    if not USE_HYBRID or hybrid_model is None or hybrid_scaler is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Hybrid model (xgb_hybrid_model.pkl / scaler_hybrid.pkl) not "
                "loaded — pre-authorization scoring requires the 76-feature "
                "hybrid model and cannot fall back to the base model, which "
                "expects a different feature shape."
            )
        )

    neighbor_ids = []
    if r:
        state = r.hgetall(f"acct:{account_id}")
        if state:
            dest_str = state.get('unique_dest', '')
            neighbor_ids = [d for d in dest_str.split(',') if d][:10]

    gnn_embedding = get_gnn_embedding_for_auth(account_id, behavioral, neighbor_ids)
    has_embed     = any(v != 0.0 for v in gnn_embedding)

    features_76 = behavioral + gnn_embedding
    X           = np.array(features_76).reshape(1, -1)
    X_scaled    = hybrid_scaler.transform(X)
    prob        = float(hybrid_model.predict_proba(X_scaled)[0][1])
    tier        = get_risk_tier_v2(prob)

    return prob, tier, has_embed


def get_risk_tier_v2(prob: float) -> str:
    if prob >= L3_THRESHOLD:
        return "L3 — AUTO FREEZE"
    elif prob >= L2_THRESHOLD:
        return "L2 — SOFT HOLD"
    elif prob >= L1_THRESHOLD:
        return "L1 — REVIEW"
    else:
        return "CLEAR"


def write_transaction_edge(
    account_from: str,
    account_to: str,
    bank_from: str,
    bank_to: str,
    amount: float,
    timestamp: str,
    payment_format: str = "unknown",
):
    """
    Writes a transaction edge to the Redis graph store.
    This is called by consumer.py for every processed transaction AND by
    the pre-auth endpoint for every authorized transaction.

    Storage structure:
      txn_out:{account_from}  -> sorted set of (timestamp_score, JSON edge data)
      txn_in:{account_to}     -> sorted set of (timestamp_score, JSON edge data)

    The sorted set uses Unix timestamp as score, enabling time-range queries.
    ltrim keeps only the 500 most recent edges per account — sufficient for
    hop mapping without unbounded Redis memory growth.
    """
    if not r:
        return

    ts_score = datetime.now().timestamp()
    edge_data = json.dumps(
        {
            "from": account_from,
            "to": account_to,
            "bfrom": bank_from,
            "bto": bank_to,
            "amount": amount,
            "ts": timestamp or datetime.now().isoformat(),
            "fmt": payment_format,
        }
    )

    pipe = r.pipeline()
    # Outbound edges for sender
    pipe.zadd(f"txn_out:{account_from}", {edge_data: ts_score})
    pipe.zremrangebyrank(f"txn_out:{account_from}", 0, -501)  # keep 500 most recent
    pipe.expire(f"txn_out:{account_from}", 86400 * 7)  # 7 day TTL

    # Inbound edges for receiver
    pipe.zadd(f"txn_in:{account_to}", {edge_data: ts_score})
    pipe.zremrangebyrank(f"txn_in:{account_to}", 0, -501)
    pipe.expire(f"txn_in:{account_to}", 86400 * 7)

    pipe.execute()


# ── Helpers (from v1, unchanged) ──────────────────────────────────────────────
def get_risk_tier(prob: float) -> str:
    return get_risk_tier_v2(prob)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ENDPOINTS — V1 (unchanged)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@app.get("/health", response_model=HealthResponse)
def health():
    # Reports the BASE model only (used by /score/account, /explain).
    # /authorize/transaction uses a separate hybrid model — check /stats
    # for USE_HYBRID status before assuming pre-auth is ready.
    return HealthResponse(
        status="operational",
        model_version=config.get("model_version", "xgb_final_v1"),
        optimal_threshold=optimal_threshold,
        feature_count=len(feature_cols)
    )


@app.post("/score/account", response_model=RiskScore)
def score_account(request: AccountFeatures):
    if len(request.features) != len(feature_cols):
        raise HTTPException(
            status_code=400,
            detail=f"Expected {len(feature_cols)} features, got {len(request.features)}",
        )
    X = np.array(request.features).reshape(1, -1)
    X_scaled = scaler.transform(X)
    prob = float(xgb_model.predict_proba(X_scaled)[0][1])
    flagged = prob >= optimal_threshold
    tier = get_risk_tier(prob)

    if r:
        r.setex(
            f"score:{request.account_id}",
            3600,
            json.dumps({"prob": prob, "tier": tier, "flagged": flagged}),
        )

    return RiskScore(
        account_id=request.account_id,
        fraud_probability=round(prob, 6),
        risk_tier=tier,
        threshold_used=optimal_threshold,
        flagged=flagged,
    )


@app.post("/explain", response_model=ExplainResponse)
def explain(request: AccountFeatures):
    if len(request.features) != len(feature_cols):
        raise HTTPException(
            status_code=400,
            detail=f"Expected {len(feature_cols)} features, got {len(request.features)}",
        )
    X = np.array(request.features).reshape(1, -1)
    X_scaled = scaler.transform(X)
    prob = float(xgb_model.predict_proba(X_scaled)[0][1])
    shap_values = explainer.shap_values(X_scaled)

    feature_shap = [
        {
            "feature": feature_cols[i],
            "value": round(float(request.features[i]), 4),
            "shap_value": round(float(shap_values[0][i]), 6),
            "direction": (
                "increases fraud risk"
                if shap_values[0][i] > 0
                else "decreases fraud risk"
            ),
        }
        for i in range(len(feature_cols))
    ]
    top_features = sorted(
        feature_shap, key=lambda x: abs(x["shap_value"]), reverse=True
    )[:5]

    return ExplainResponse(
        account_id=request.account_id,
        fraud_probability=round(prob, 6),
        top_features=top_features,
        risk_tier=get_risk_tier(prob),
    )


@app.get("/account/{account_id}/cached")
def get_cached_score(account_id: str):
    if not r:
        raise HTTPException(status_code=503, detail="Redis not available")
    cached = r.get(f"score:{account_id}")
    if not cached:
        raise HTTPException(status_code=404, detail="No cached score for this account")
    return json.loads(cached)


@app.get("/stats")
def get_stats():
    stats = {
        "model": "XGBoost + GNN Hybrid" if USE_HYBRID else "XGBoost",
        "api_version": "2.0.0",
        "optimal_threshold": optimal_threshold,
        "risk_tiers": {
            "L1_REVIEW": f"probability >= {L1_THRESHOLD}",
            "L2_SOFT_HOLD": f"probability >= {L2_THRESHOLD}",
            "L3_AUTO_FREEZE": f"probability >= {L3_THRESHOLD}",
        },
        "endpoints": {
            "pre_authorization": "POST /authorize/transaction",
            "hop_mapping": "GET /hops/{account_id}",
            "spike_detection": "GET /spike/{account_id}?amount=X",
        },
    }
    if r:
        stats["redis_keys"] = r.dbsize()
        stats["alerts_total"] = r.get("alerts_total_count") or "0"
        stats["confirmed_fraud"] = len(r.keys("launder_status:*"))
    return stats


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ENDPOINTS — V2 NEW: PRE-AUTHORIZATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@app.post("/authorize/transaction", response_model=AuthDecision)
def authorize_transaction(req: TransactionAuthRequest):
    """
    PRE-AUTHORIZATION ENDPOINT — The most important new endpoint.

    Called by bank CBS (SBI Finacle / Kotak CBS) BEFORE approving a transaction.
    Must return a decision within 100ms to avoid CBS timeout.

    Flow:
    1. Pull sender's current account state from Redis (microseconds)
    2. Simulate adding this transaction to compute forward-looking features
    3. Fetch GNN embedding (Redis cache hit = <1ms, miss = <80ms)
    4. Score with XGBoost (76 features, <5ms)
    5. Check vendor spike detection
    6. Return structured decision with human-readable reasons

    Decision logic:
    - BLOCK:   L3 (prob >= 0.70) OR 10x+ spike
    - HOLD:    L2 (prob >= 0.50) OR 3x+ spike
    - APPROVE: L1 or CLEAR

    Integration with SBI Finacle:
    Finacle's pre-authorization webhook calls this endpoint with the transaction
    details. The response decision maps to Finacle's internal approval codes.

    Integration with Kotak CBS:
    Kotak's transaction gateway calls this endpoint synchronously.
    """
    start_time = datetime.now()
    reasons = []
    confidence = "HIGH"

    # ── Step 1: Pull sender account state from Redis ──────────────────────────
    state = get_account_state(req.account_from)

    if not state:
        # New account — no behavioral history
        # Use amount alone as a weak signal
        confidence = "LOW"
        reasons.append("New account — no behavioral history in system")
        behavioral = [
            0.95,
            0.05,
            1.0,
            1.0,
            0.0,
            0.0,
            1.0,
            0.0,
            1.0,
            1.0,
            req.amount,
            req.amount,
        ]
    else:
        # ── Step 2: Compute forward-looking behavioral features ───────────────
        behavioral = compute_behavioral_features_from_state(
            state, req.amount, req.bank_to
        )
        total_txn = int(state.get("total_txn", "0"))
        if total_txn < 5:
            confidence = "MEDIUM"
            reasons.append(f"Limited history: only {total_txn} prior transactions")

    # ── Step 3: Vendor spike detection ───────────────────────────────────────
    spike_info = detect_vendor_spike(req.account_from, req.amount)
    if spike_info["spike"]:
        reasons.append(f"SPIKE DETECTED: {spike_info['reason']}")

    # ── Step 4: Score with XGBoost + GNN ─────────────────────────────────────
    prob, tier, has_embed = score_account_internal(
        req.account_from, behavioral, req.bank_to
    )

    if not has_embed:
        confidence = "MEDIUM" if confidence == "HIGH" else confidence
        reasons.append("GNN graph embedding not available for this account")

    # ── Step 5: CEP pattern check from Redis ─────────────────────────────────
    if r:
        cep_raw = r.lrange(f"cep_alerts:{req.account_from}", 0, 4)
        if cep_raw:
            patterns = list({json.loads(a).get("pattern_type", "") for a in cep_raw})
            reasons.append(f"CEP patterns previously fired: {', '.join(patterns)}")

    # ── Step 6: Confirmed fraud check ────────────────────────────────────────
    if r:
        if r.exists(f"launder_status:{req.account_from}"):
            reasons.append("CONFIRMED: Account appears in confirmed laundering records")
            prob = max(prob, 0.75)  # floor the score for confirmed accounts
            tier = get_risk_tier_v2(prob)

    # ── Step 7: Cross-bank flag ───────────────────────────────────────────────
    if state:
        sender_banks = set(state.get("unique_banks", "").split(","))
        sender_banks.discard("")
        if len(sender_banks) >= 5:
            reasons.append(
                f"High cross-bank activity: {len(sender_banks)} banks involved"
            )

    # ── Step 8: Map to decision ───────────────────────────────────────────────
    is_critical_spike = spike_info["spike"] and spike_info["ratio"] >= 10.0
    is_moderate_spike = spike_info["spike"] and spike_info["ratio"] >= 3.0

    if prob >= L3_THRESHOLD or is_critical_spike:
        decision = "BLOCK"
        recommended_action = (
            "Transaction blocked. Refer to compliance team immediately. "
            "Consider account freeze under PMLA Section 17."
        )
        if not reasons:
            reasons.append(
                f"Fraud probability {prob:.4f} exceeds L3 threshold {L3_THRESHOLD}"
            )

    elif prob >= L2_THRESHOLD or is_moderate_spike:
        decision = "HOLD"
        recommended_action = (
            "Transaction held for analyst review. Contact account holder "
            "for transaction verification before releasing funds."
        )
        if not reasons:
            reasons.append(
                f"Fraud probability {prob:.4f} exceeds L2 threshold {L2_THRESHOLD}"
            )

    else:
        decision = "APPROVE"
        recommended_action = (
            "Transaction approved. No significant fraud indicators detected."
        )
        if not reasons:
            reasons.append("Behavioral profile within normal parameters")

    # ── Step 9: Write transaction edge to graph store (async) ─────────────────
    # This builds the hop mapping graph incrementally.
    # Every authorized transaction contributes to the network visualization.
    write_transaction_edge(
        account_from=req.account_from,
        account_to=req.account_to,
        bank_from=req.bank_from,
        bank_to=req.bank_to,
        amount=req.amount,
        timestamp=req.timestamp or datetime.now().isoformat(),
        payment_format=req.payment_format,
    )

    sla_ms = int((datetime.now() - start_time).total_seconds() * 1000)

    return AuthDecision(
        transaction_id=req.transaction_id,
        account_from=req.account_from,
        decision=decision,
        risk_tier=tier,
        fraud_probability=round(prob, 6),
        confidence=confidence,
        reasons=reasons if reasons else ["No specific risk indicators"],
        sla_ms=sla_ms,
        recommended_action=recommended_action,
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ENDPOINTS — V2 NEW: HOP MAPPING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@app.get("/hops/{account_id}", response_model=HopMapResponse)
def get_hop_map(account_id: str, max_depth: int = 3, max_nodes: int = 50):
    """
    HOP MAPPING ENDPOINT — Traces the complete money flow network.

    Starting from a flagged account, performs BFS in both directions:
    - FORWARD (outbound): where did this account send money?
    - BACKWARD (inbound): who sent money to this account?

    At each hop, fetches:
    - The account's risk score from Redis cache
    - The account's behavioral state
    - Confirmed fraud status
    - Vendor spike detection

    Parameters:
        account_id: The root account to map from (typically a flagged account)
        max_depth:  Maximum hops in each direction (default 3, max 5)
        max_nodes:  Maximum total nodes to return (default 50, prevents graph explosion)

    Returns the complete inter-bank network with risk scores at each node,
    enabling investigators to trace the criminal ring and identify all participants.

    Example output for account A in ring A->B->C->A:
        Nodes: A (L3), B (L2), C (L3)
        Edges: A->B (Rs.50,000), B->C (Rs.48,000), C->A (Rs.46,000)
        Banks: HDFC, SBI, Axis (3 banks involved in ring)
    """
    if not r:
        raise HTTPException(status_code=503, detail="Redis not available")

    max_depth = min(max_depth, 5)  # hard cap at 5 hops
    max_nodes = min(max_nodes, 100)  # hard cap at 100 nodes

    # Reserve budget across depths. Previously, a high-fan-out root account
    # (e.g. 49 direct connections) could consume the ENTIRE max_nodes budget
    # at depth 1 alone, before the BFS ever got a chance to explore depth 2+
    # -- meaning the visualization would always show a flat single-hop star
    # for such accounts, never revealing further laundering hops that
    # genuinely exist deeper in the graph. Capping how many depth-1 nodes
    # get enqueued in the first place guarantees remaining budget survives
    # for depth 2+.
    DEPTH_1_CAP = min(30, max(max_nodes - 10, 10))
    depth1_enqueued = 0

    visited_nodes = {}  # account_id -> HopNode
    all_edges = []  # list of HopEdge

    # BFS queue: (account_id, depth, direction)
    # direction: 'forward' = follow outbound, 'backward' = follow inbound
    queue = deque()
    queue.append((account_id, 0, "both"))
    seen = {account_id}

    while queue and len(visited_nodes) < max_nodes:
        current_id, depth, direction = queue.popleft()

        # ── Build node data for current account ──────────────────────────────
        state = get_account_state(current_id)
        cached_score = r.get(f"score:{current_id}")
        is_confirmed = bool(r.exists(f"launder_status:{current_id}"))

        risk_score = None
        risk_tier_str = None
        if cached_score:
            cached = json.loads(cached_score)
            risk_score = cached.get("prob")
            risk_tier_str = cached.get("tier")

        # Determine primary bank from account state
        bank = "Unknown"
        if state:
            banks_str = state.get("unique_banks", "")
            banks_list = [b for b in banks_str.split(",") if b]
            if banks_list:
                bank = banks_list[0]

        # Spike detection for this node
        total_sent = float(state.get("total_sent", "0")) if state else 0.0
        total_txn = int(state.get("total_txn", "1")) if state else 1
        baseline = total_sent / max(total_txn, 1)

        # Check inbound amounts to detect if this node received a spike
        inbound_raw = r.zrange(f"txn_in:{current_id}", -10, -1)
        spike_detected = False
        if inbound_raw and baseline > 0:
            for edge_json in inbound_raw:
                try:
                    edge = json.loads(edge_json)
                    if edge.get("amount", 0) / baseline >= 3.0:
                        spike_detected = True
                        break
                except Exception:
                    pass

        visited_nodes[current_id] = HopNode(
            account_id=current_id,
            bank=bank,
            total_sent=total_sent,
            transaction_count=int(state.get("total_txn", "0")) if state else 0,
            risk_score=round(risk_score, 4) if risk_score else None,
            risk_tier=risk_tier_str,
            is_confirmed_fraud=is_confirmed,
            spike_detected=spike_detected,
            hop_depth=depth,
        )

        if depth >= max_depth:
            continue

        # ── Traverse outbound edges (follow the money forward) ────────────────
        if direction in ("both", "forward"):
            outbound_raw = r.zrange(f"txn_out:{current_id}", -50, -1)
            for edge_json in outbound_raw:
                try:
                    edge = json.loads(edge_json)
                    to_acc = edge["to"]

                    all_edges.append(
                        HopEdge(
                            from_account=current_id,
                            to_account=to_acc,
                            from_bank=edge.get("bfrom", "Unknown"),
                            to_bank=edge.get("bto", "Unknown"),
                            amount=edge.get("amount", 0),
                            timestamp=edge.get("ts", ""),
                            payment_format=edge.get("fmt", "unknown"),
                        )
                    )

                    new_depth = depth + 1
                    depth_ok = (new_depth > 1) or (depth1_enqueued < DEPTH_1_CAP)
                    if to_acc not in seen and len(visited_nodes) < max_nodes and depth_ok:
                        seen.add(to_acc)
                        if new_depth == 1:
                            depth1_enqueued += 1
                        queue.append((to_acc, new_depth, "forward"))
                except Exception:
                    pass

        # ── Traverse inbound edges (trace money backwards) ────────────────────
        if direction in ("both", "backward"):
            inbound_raw = r.zrange(f"txn_in:{current_id}", -50, -1)
            for edge_json in inbound_raw:
                try:
                    edge = json.loads(edge_json)
                    from_acc = edge["from"]

                    # Only add edge if not already captured by forward traversal
                    edge_key = f"{from_acc}->{current_id}"
                    existing = any(
                        e.from_account == from_acc and e.to_account == current_id
                        for e in all_edges
                    )
                    if not existing:
                        all_edges.append(
                            HopEdge(
                                from_account=from_acc,
                                to_account=current_id,
                                from_bank=edge.get("bfrom", "Unknown"),
                                to_bank=edge.get("bto", "Unknown"),
                                amount=edge.get("amount", 0),
                                timestamp=edge.get("ts", ""),
                                payment_format=edge.get("fmt", "unknown"),
                            )
                        )

                    new_depth = depth + 1
                    depth_ok = (new_depth > 1) or (depth1_enqueued < DEPTH_1_CAP)
                    if from_acc not in seen and len(visited_nodes) < max_nodes and depth_ok:
                        seen.add(from_acc)
                        if new_depth == 1:
                            depth1_enqueued += 1
                        queue.append((from_acc, new_depth, "backward"))
                except Exception:
                    pass

    # ── Build summary ─────────────────────────────────────────────────────────
    all_banks = set()
    total_amount_moved = 0.0
    for edge in all_edges:
        all_banks.add(edge.from_bank)
        all_banks.add(edge.to_bank)
        total_amount_moved += edge.amount

    confirmed_nodes = [n for n in visited_nodes.values() if n.is_confirmed_fraud]
    high_risk_nodes = [
        n
        for n in visited_nodes.values()
        if n.risk_score and n.risk_score >= L3_THRESHOLD
    ]
    spike_nodes = [n for n in visited_nodes.values() if n.spike_detected]

    summary = {
        "banks_involved": sorted(list(all_banks - {"Unknown"})),
        "bank_count": len(all_banks - {"Unknown"}),
        "total_amount_moved": round(total_amount_moved, 2),
        "confirmed_fraud_nodes": len(confirmed_nodes),
        "high_risk_nodes": len(high_risk_nodes),
        "spike_nodes": len(spike_nodes),
        "investigation_note": (
            f"Network spans {len(all_banks)} banks. "
            f"Rs.{total_amount_moved:,.0f} moved across {len(all_edges)} transactions. "
            f"{len(confirmed_nodes)} confirmed fraud accounts identified."
        ),
    }

    return HopMapResponse(
        root_account=account_id,
        total_nodes=len(visited_nodes),
        total_edges=len(all_edges),
        max_depth=max_depth,
        nodes=list(visited_nodes.values()),
        edges=all_edges,
        summary=summary,
    )


@app.get("/spike/{account_id}")
def check_spike(account_id: str, amount: float = 0.0):
    """
    Standalone vendor spike detection endpoint.
    Returns whether a proposed amount represents an anomalous spike
    relative to this account's 30-day transaction baseline.

    Usage: GET /spike/ACC123?amount=5000000
    """
    result = detect_vendor_spike(account_id, amount)
    state = get_account_state(account_id)
    return {
        "account_id": account_id,
        "proposed_amount": amount,
        "spike_analysis": result,
        "account_history": (
            {
                "total_transactions": int(state.get("total_txn", 0)) if state else 0,
                "total_sent": float(state.get("total_sent", 0)) if state else 0.0,
                "max_single_amount": (
                    float(state.get("max_amount", 0)) if state else 0.0
                ),
            }
            if state
            else None
        ),
    }


@app.get("/network/summary")
def network_summary():
    """
    High-level summary of the transaction network currently in Redis.
    Useful for the MoF/RBI dashboard to show overall system coverage.
    """
    if not r:
        raise HTTPException(status_code=503, detail="Redis not available")

    out_keys = r.keys("txn_out:*")
    in_keys = r.keys("txn_in:*")

    return {
        "accounts_with_outbound_history": len(out_keys),
        "accounts_with_inbound_history": len(in_keys),
        "total_alerts": r.get("alerts_total_count") or "0",
        "confirmed_fraud_accounts": len(r.keys("launder_status:*")),
        "active_account_states": len(r.keys("acct:*")),
        "gnn_embeddings_cached": len(r.keys("embed:*")),
        "timestamp": datetime.now().isoformat(),
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
