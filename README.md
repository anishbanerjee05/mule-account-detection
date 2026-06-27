# Mule Account Detection System

A production-grade machine learning system for detecting money mule accounts in financial transaction networks. Built during a research internship at CyberPeace Foundation, with planned deployment across all RBI-regulated banks in India.

**Demo:** August 1, 2026 — SBI and Kotak Mahindra Bank  
**Target:** National deployment across 700+ RBI-regulated banks

---

## What is a Mule Account?

A mule account is a bank account used by criminal networks to receive and forward stolen funds, creating an obfuscation layer between the crime and the criminal. Detection is challenging because mule accounts:

- Pass KYC verification at account opening
- Remain dormant before activation
- Individually show no single transaction above reporting thresholds
- Only reveal their criminal nature when analysed as a coordinated network

---

## System Architecture

The system is a four-layer detection pipeline running on Apache Kafka:

```
Bank CBS (SBI Finacle / Kotak CBS)
        |
Schema Adapter (schema_adapter.py)
        |
Kafka (4 partitions, timestamped topic per session)
        |
Consumer Pipeline (consumer.py) — batch mode, size=100
    |-- Account State            → Redis (rolling 24h window)
    |-- Behavioral Features (12) computed in-process
    |-- GNN Embeddings (64)      → batch call to gnn_service.py
    |-- XGBoost Scoring (76)     in-process
    |-- CEP Engine (6 patterns)  → Redis rolling windows
    |-- Alerts                   → Redis → Dashboard
        |
GNN Microservice (gnn_service.py, port 8002)
    |-- Tier 1: Redis cache (sub-millisecond)
    |-- Tier 2: Persistent store (embeddings_store.parquet, 579,735 embeddings)
    |-- Tier 3: Live GNN forward pass (fallback for unknown accounts)
        |
Dashboard (dashboard.py, Streamlit)
    |-- Live alert queue (deduplicated, CEP-only catches pinned)
    |-- SHAP feature attribution (behavioral 59% / GNN 41% split)
    |-- Account summary with risk tier
    |-- STR generation in FIU-IND format
        |
MLOps (mlops.py)
    |-- PSI-based drift detection
    |-- Analyst feedback retraining (5x sample weights)
    |-- Model registry (SQLite)
```

---

## Model Performance

### Hybrid XGBoost + GNN (Production Model)

| Metric | Value |
|--------|-------|
| AUC-ROC | **0.8516** |
| F1 Score | 0.1436 |
| Optimal Threshold | 0.3070 |
| Features | 76 (12 behavioral + 64 GNN embeddings) |
| Improvement vs standalone XGBoost | +6.59pp AUC-ROC |

### CEP Pattern Engine (6 Empirically Validated Patterns)

All patterns validated against IBM AML ground truth labels across 6.9M transactions:

| Pattern | Flagged | Confirmed | Precision | Lift |
|---------|---------|-----------|-----------|------|
| GATHER_SCATTER | 16 | 15 | **93.75%** | 124.8x |
| FAN_OUT_SPIKE | 31 | 15 | **48.39%** | 64.4x |
| SIMPLE_CYCLE | 116 | 15 | **12.93%** | 17.2x |
| FAN_IN | 9,041 | 594 | 6.57% | 8.7x |
| STRUCTURING | 637 | 26 | 4.08% | 5.4x |
| SCATTER_GATHER | 176 | 6 | 3.41% | 4.5x |

**Overall CEP: 6.53% precision, 8.7x lift over base rate, 28s for 6.9M rows**

### Throughput Performance

| Configuration | Throughput | vs Baseline |
|--------------|------------|-------------|
| Baseline (sequential, no batch) | 27 TPS | 1x |
| Batch GNN (size=100, 1 consumer) | 500 TPS | 18.6x |
| Batch GNN (size=100, 2 consumers) | 1,000 TPS | 37x |
| Target (4 consumers, 4 partitions) | 2,000 TPS | 74x |

---

## CEP Pattern Validation

Patterns were validated using an offline DuckDB validator (`cep_duckdb_validator.py`) running pure SQL window functions against the full 6.9M row IBM AML dataset. Dropped patterns after empirical diagnosis:

- **RAPID_FORWARD**: Unfalsifiable at minute-level timestamp resolution
- **CONCENTRATED_DESTINATION**: Near-zero fraud/non-fraud separation
- **BURST_ACTIVATION**: Zero fires in full dataset validation
- **PASS_THROUGH_RATIO**: Directionally backwards (fraud LESS symmetric)
- **AMOUNT_SYMMETRY**: No signal, 0.83% precision at base rate
- **LAYERING_CHAIN**: Fraud/non-fraud ratio distributions identical

---

## MuleGNN Architecture

```
Input: 15 node features (12 behavioral + 3 graph-level)
GCNConv(15 -> 64)         -- 1-hop neighborhood aggregation
GCNConv(64 -> 64)         -- 2-hop neighborhood aggregation
EdgeProp Linear(2 -> 64)  -- edge feature integration
FC(128 -> 64)             -- classification head
FC(64 -> 2)               -- fraud probability output
Total parameters: 13,762
Training: 300 epochs, Adam lr=0.01, weighted CrossEntropyLoss
Embeddings: 579,735 accounts, stored in embeddings_store.parquet (69.4MB)
```

---

## Production Deployment Roadmap

### Phase 1 (Aug 2026 - Mar 2027): SBI + Kotak Pilot
- Current Python architecture on bank infrastructure
- High-risk transaction types: international NEFT, large RTGS, cross-bank UPI > Rs.50,000
- Volume: 5-10 million transactions/day

### Phase 2 (Apr 2027 - Mar 2028): Top 10 Banks
- Apache Flink replacing Python consumer
- Federated learning across all participating banks
- Full UPI coverage
- Volume: 500 million transactions/day

### Phase 3 (Apr 2028+): National Deployment
- All 700+ UPI-connected banks
- NPCI integration for cross-bank graph analysis
- Temporal GNN updated in real time
- Volume: 1 billion+ transactions/day

---

## Quick Start

```powershell
# Clone repository
git clone https://github.com/anishbanerjee05/mule-account-detection.git
cd mule-account-detection

# Production environment (Python 3.14)
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# GNN environment (Python 3.11.9 + CUDA 12.1)
python311 -m venv venv311
venv311\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install torch_geometric pandas numpy fastapi uvicorn

# Start Docker infrastructure
docker-compose up -d

# Run full demo (single command)
.\demo.ps1

# Start producer
python producer.py
```

---

## Key Files

| File | Description |
|------|-------------|
| `consumer.py` | Kafka consumer with batch GNN embedding, XGBoost scoring, CEP |
| `gnn_service.py` | GNN microservice with 3-tier embedding lookup + /embed_batch |
| `cep_engine.py` | CEP v2 with 6 IBM AML-aligned patterns |
| `cep_duckdb_validator.py` | Offline pattern validation (6.9M rows in 28s) |
| `dashboard.py` | Streamlit analyst dashboard with SHAP + STR generation |
| `schema_adapter.py` | SBI Finacle + Kotak CBS YAML config adapters |
| `mlops.py` | Drift detection, retraining, model registry |
| `demo.ps1` | Single-command demo launcher |

---

## Datasets

### IBM AML Small (Primary)
- 6,924,049 transactions, 705,903 unique accounts
- 5,304 confirmed laundering accounts (0.7514% base rate)
- Download: https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml

---

## Regulatory Alignment

PMLA 2002 | RBI Master Directions on KYC (2024) | RBI Governor Directive on Mule Accounts (July 2024) | DPDP Act 2023 | RBI FREE-AI Framework (August 2025) | FATF Recommendation 16

---

## Research Foundation

| Reference | Contribution |
|-----------|-------------|
| Altman et al. (2023). Realistic Synthetic Financial Transactions for AML. ArXiv:2306.16424 | IBM AML dataset, 8 typology definitions |
| Tam et al. (2019). Identifying Illicit Accounts in Large Scale E-payment Networks. ArXiv:1906.05546 | EdgeProp mechanism |
| NVIDIA AI Blueprint for Fraud Detection (2024) | GNN + XGBoost hybrid architecture validation |
| RBI Governor Directive (July 2024) | Regulatory urgency, PMLA alignment |
| RBI FREE-AI Framework (August 2025) | Explainability compliance |

---

## Author

**Anish Banerjee**  
B.Tech Computer Science, DBS Global University  
B.Sc. Data Science, IIT Madras  
CyberPeace Foundation Research Wing Intern, June-September 2026  
GitHub: [anishbanerjee05](https://github.com/anishbanerjee05)