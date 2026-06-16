from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import json
import numpy as np
import pandas as pd
import shap
import uvicorn
from typing import List, Optional
import redis
import os

app = FastAPI(
    title="Mule Account Detection API",
    description="Real-time mule account detection — CyberPeace Foundation",
    version="1.0.0"
)

# ── Load models on startup ────────────────────────────────────────────────────
print("Loading models...")
with open("xgb_final_model.pkl", "rb") as f:
    xgb_model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("model_config.json", "r") as f:
    config = json.load(f)

feature_cols = config["feature_cols"]
optimal_threshold = config["optimal_threshold"]

explainer = shap.TreeExplainer(xgb_model)

# ── Redis connection ──────────────────────────────────────────────────────────
try:
    r = redis.Redis(host="localhost", port=6379, decode_responses=True)
    r.ping()
    print("Redis connected")
except:
    r = None
    print("Redis not available — running without state store")

print(f"Models loaded. Optimal threshold: {optimal_threshold:.4f}")
print(f"Features: {len(feature_cols)}")

# ── Request/Response schemas ──────────────────────────────────────────────────
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

# ── Helpers ───────────────────────────────────────────────────────────────────
def get_risk_tier(prob: float) -> str:
    if prob >= 0.97:
        return "L3 — AUTO FREEZE"
    elif prob >= 0.50:
        return "L2 — SOFT HOLD"
    elif prob >= 0.30:
        return "L1 — REVIEW"
    else:
        return "CLEAR"

# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse)
def health():
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
            detail=f"Expected {len(feature_cols)} features, got {len(request.features)}"
        )

    # Scale and predict
    X = np.array(request.features).reshape(1, -1)
    X_scaled = scaler.transform(X)
    prob = float(xgb_model.predict_proba(X_scaled)[0][1])
    flagged = prob >= optimal_threshold
    tier = get_risk_tier(prob)

    # Cache result in Redis
    if r:
        r.setex(
            f"score:{request.account_id}",
            3600,
            json.dumps({"prob": prob, "tier": tier, "flagged": flagged})
        )

    return RiskScore(
        account_id=request.account_id,
        fraud_probability=round(prob, 6),
        risk_tier=tier,
        threshold_used=optimal_threshold,
        flagged=flagged
    )

@app.post("/explain", response_model=ExplainResponse)
def explain(request: AccountFeatures):
    if len(request.features) != len(feature_cols):
        raise HTTPException(
            status_code=400,
            detail=f"Expected {len(feature_cols)} features, got {len(request.features)}"
        )

    X = np.array(request.features).reshape(1, -1)
    X_scaled = scaler.transform(X)
    prob = float(xgb_model.predict_proba(X_scaled)[0][1])
    shap_values = explainer.shap_values(X_scaled)

    # Build top features list
    feature_shap = [
        {
            "feature": feature_cols[i],
            "value": round(float(request.features[i]), 4),
            "shap_value": round(float(shap_values[0][i]), 6),
            "direction": "increases fraud risk" if shap_values[0][i] > 0 else "decreases fraud risk"
        }
        for i in range(len(feature_cols))
    ]
    top_features = sorted(feature_shap, key=lambda x: abs(x["shap_value"]), reverse=True)[:5]

    return ExplainResponse(
        account_id=request.account_id,
        fraud_probability=round(prob, 6),
        top_features=top_features,
        risk_tier=get_risk_tier(prob)
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
        "model": "XGBoost",
        "optimal_threshold": optimal_threshold,
        "features": feature_cols,
        "risk_tiers": {
            "L1_REVIEW": "probability >= 0.30",
            "L2_SOFT_HOLD": "probability >= 0.50",
            "L3_AUTO_FREEZE": "probability >= 0.97"
        }
    }
    if r:
        stats["redis_keys"] = r.dbsize()
    return stats

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)