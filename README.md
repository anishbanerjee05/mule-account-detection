# Mule Account Detection System

A machine learning system for detecting money mule accounts in financial transaction networks, built as a streaming pipeline (Kafka, Redis) around a hybrid XGBoost + Graph Neural Network model and a rule-based Complex Event Processing (CEP) engine. Developed during a research internship at the CyberPeace Foundation Research Wing.

**Pilot target:** SBI and Kotak Mahindra Bank
**Long-term target:** RBI-regulated banks in India

---

## Contents

1. [Reproducing this project from a fresh clone](#1-reproducing-this-project-from-a-fresh-clone)
2. [Data and model artifacts](#2-data-and-model-artifacts)
3. [The GNN embedding workflow](#3-the-gnn-embedding-workflow)
4. [Which model and configuration the pipeline uses](#4-which-model-and-configuration-the-pipeline-uses)
5. [Running components individually](#5-running-components-individually)
6. [System architecture](#6-system-architecture)
7. [Results](#7-results)
8. [Repository map](#8-repository-map)
9. [Troubleshooting](#9-troubleshooting)
10. [Known limitations](#10-known-limitations)
11. [Validation status](#11-validation-status)
12. [Background: datasets, roadmap, references](#12-background)

---

## 1. Reproducing this project from a fresh clone

### What you need

| Requirement | Notes |
|-------------|-------|
| Windows 10/11 with PowerShell | The launcher (`demo.ps1`) is PowerShell. The Python code itself is portable, but only the Windows launch path has been tested. |
| Docker Desktop (running) | Provides Kafka, Zookeeper and Redis. The Live Monitoring dashboard page also calls `docker exec`. |
| Python 3.14 | Environment `venv`: consumer, producer, dashboard, optional APIs. |
| Python 3.11 | Environment `venv311`: GNN service and notebooks 04-05. |
| About 16 GB RAM | The development machine had 16.8 GB, of which only about 6 GB was free with Docker running. The GNN service therefore runs 2 workers, each loading the full embedding store (579,735 vectors) into memory. `consumer.py` also reads the whole 620 MB dataset CSV at startup to build its label set. |
| About 3 GB free disk | Excluding the PyTorch download. |
| NVIDIA GPU (optional) | Only needed to retrain the GNN. Serving and scoring run on CPU. |

### Step by step

Run every command from the repository root. Scripts load files by relative path.

**1. Clone**

```powershell
git clone https://github.com/anishbanerjee05/mule-account-detection.git
cd mule-account-detection
```

**2. Create the two environments**

```powershell
py -3.14 -m venv venv
.\venv\Scripts\python.exe -m pip install -r requirements.txt

py -3.11 -m venv venv311
.\venv311\Scripts\python.exe -m pip install -r requirements-gnn.txt
```

`requirements-gnn.txt` installs the CUDA 12.1 build of PyTorch (a large download that also runs on CPU-only machines). For a smaller CPU-only install, first run:

```powershell
.\venv311\Scripts\python.exe -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
```

and remove the `--extra-index-url` and `torch==` lines from `requirements-gnn.txt` before installing the rest.

To run the notebooks, additionally install `requirements-notebooks.txt` on top of the relevant environment (notebooks 01-03b in `venv`, notebooks 04-05 in `venv311`).

**3. Obtain the dataset** (not in the repository: 620 MB). See [Section 2](#2-data-and-model-artifacts). Place it in the repository root as `ibm_aml_small.csv`.

**4. Obtain the embedding store** (69.4 MiB, distributed as a release asset). Download it into the repository root:

```powershell
Invoke-WebRequest -Uri "https://github.com/anishbanerjee05/mule-account-detection/releases/download/V1.0/embeddings_store.parquet" -OutFile embeddings_store.parquet
```

The tag is case-sensitive (`V1.0`). Check the file against the SHA-256 table in [Section 2](#2-data-and-model-artifacts).

The three model files (`best_gnn_model.pt`, `xgb_hybrid_model.pkl`, `scaler_hybrid.pkl`) and `hybrid_model_config.json` are committed to this repository, so no action is needed for them.

**5. Allow the launcher to run for this session**

```powershell
Set-ExecutionPolicy -Scope Process Bypass
```

**6. Start everything**

```powershell
.\demo.ps1 -producers
```

`demo.ps1` resolves its location from the script itself, so it works from any clone path. It performs these phases:

| Phase | Action |
|-------|--------|
| 0 | Checks that Docker, both virtual environments and every required file exist, and lists any that are missing. |
| 1 | Starts the Docker services (`docker compose up -d`) if they are not running. |
| 2 | Flushes Redis, creates a fresh timestamped Kafka topic with 4 partitions, verifies it is empty, and records its name in `active_topic.txt`. |
| 3 | Starts the GNN service (port 8002) in a new window and waits until it reports that the embedding store loaded. |
| 3b | Loads the embeddings into Redis (`warm_redis_cache.py`) if the cache is cold. |
| 4 | Starts 2 consumer windows (change with `-consumers N`). |
| 5 | Starts the dashboard and opens http://localhost:8501. With `-producers`, also starts the producer. |

Other switches: `.\demo.ps1 -skip_reset` reuses the existing topic and Redis state; `.\demo.ps1` without `-producers` leaves the producer to be started by hand (`.\venv\Scripts\python.exe producer.py`).

**7. Check it is working**

- GNN service: http://localhost:8002/stats should show `"persistent_store_loaded": true`.
- Dashboard: http://localhost:8501 (Dashboard, Live Monitoring, Flagged Accounts and Hop Map pages are built).
- Consumer windows print the processed and flagged counts as transactions are scored.

**8. Stop**

Close the PowerShell windows opened by the launcher, then `docker compose down`.

---

## 2. Data and model artifacts

### Artifact table

| File | Size | In git? | How to obtain | Used by |
|------|------|---------|---------------|---------|
| `best_gnn_model.pt` | 0.06 MB | Yes | Committed | `gnn_service.py` |
| `xgb_hybrid_model.pkl` | 1.6 MB | Yes | Committed | `consumer.py`, dashboards, `mlops.py` |
| `scaler_hybrid.pkl` | 2 KB | Yes | Committed | `consumer.py`, dashboards, `mlops.py` |
| `hybrid_model_config.json` | 1 KB | Yes | Committed | `consumer.py`, dashboards, `mlops.py` |
| `embeddings_store.parquet` | 69.4 MiB (72,775,892 bytes) | No | https://github.com/anishbanerjee05/mule-account-detection/releases/download/V1.0/embeddings_store.parquet | `gnn_service.py`, `warm_redis_cache.py` |
| `ibm_aml_small.csv` | 620 MB | No | Kaggle download below | `producer.py`, `consumer.py`, notebooks |

### The dataset (`ibm_aml_small.csv`)

Source: the IBM "Transactions for Anti Money Laundering (AML)" dataset on Kaggle: https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml

1. Download the dataset and take the **LI-Small** transactions file (`LI-Small_Trans.csv`). The pipeline was built on the file with 6,924,049 transactions. [Verify against your download: the file has 6,924,049 data rows plus a header.]
2. Rename it to `ibm_aml_small.csv` and place it in the repository root.
3. No other preparation is needed for the streaming pipeline. The producer reads these columns: `Account`, `Account.1`, `From Bank`, `To Bank`, `Amount Paid`, `Amount Received`, `Payment Currency`, `Receiving Currency`, `Payment Format`, `Is Laundering`. It streams only the payment formats ACH, Cheque, Credit Card, Cash and Bitcoin.

CSV files are excluded from git (`.gitignore`).

### Checksums

The pickle files execute code when loaded. Verify what you received before running it:

```powershell
Get-FileHash best_gnn_model.pt, xgb_hybrid_model.pkl, scaler_hybrid.pkl, embeddings_store.parquet -Algorithm SHA256
```

| File | SHA-256 |
|------|---------|
| `best_gnn_model.pt` | `86D4168E8F0BA427FFAB92275BA49BA5502AF90CD48AE01BF2F0C4C2290204C3` |
| `xgb_hybrid_model.pkl` | `22CEAABE140E20167B17A71A21D5D1AA70534D680608E125D2A2C576FCE670F2` |
| `scaler_hybrid.pkl` | `215D956AC24E913EC13459E27AA085977AAEBE092F48F9119600C1B785C36562` |
| `embeddings_store.parquet` | `1462544E6DD9E21BCF9D6AFDE9B596C14C15B2B56DEF0B89A65579E5241CE22D` |

The model files were saved with `xgboost` 3.2.0, `scikit-learn` 1.9.0, `pandas` 3.0.3 and `numpy` 2.4.x, which `requirements.txt` pins. Different versions may fail to load the pickles or behave differently.

Note: `mlops.py` rewrites `xgb_hybrid_model.pkl`, `scaler_hybrid.pkl` and `hybrid_model_config.json` when it retrains the model. After a retraining demonstration these files differ from the committed copies and their checksums will no longer match. Restore them with `git restore xgb_hybrid_model.pkl scaler_hybrid.pkl hybrid_model_config.json`.

### Regenerating the artifacts from scratch

Retraining is not needed to run the system, and a retrained model may score slightly differently from the committed one. To regenerate:

| Step | Run | Environment | Produces |
|------|-----|-------------|----------|
| 1 | Download the dataset as above | n/a | `ibm_aml_small.csv` |
| 2 | `02b_IBM_EDA.ipynb` | `venv` | `ibm_feature_matrix.csv` (git-ignored) |
| 3 | `04_graph_construction.ipynb` | `venv311` | The transaction graph used by notebook 05 |
| 4 | `05_GNN.ipynb` (needs Docker/Redis running; GPU recommended) | `venv311` | `best_gnn_model.pt`, `graph_data.pkl`, embeddings in Redis, then the hybrid model files `xgb_hybrid_model.pkl`, `scaler_hybrid.pkl`, `hybrid_model_config.json` |
| 5 | `export_embeddings.py` | `venv311` | `embeddings_store.parquet` [verify: inputs of this script] |

Notebooks 01, 02 and 03 are the PaySim-based exploration and baseline models (`model_config.json`). They are not needed by the pipeline.

---

## 3. The GNN embedding workflow

The trained MuleGNN (`best_gnn_model.pt`, produced by `05_GNN.ipynb`) is run once over the full transaction graph by `export_embeddings.py`, which writes a 64-dimensional embedding for each of the 579,735 training accounts to `embeddings_store.parquet`. This file is the durable source of truth.

At runtime `gnn_service.py` (FastAPI, port 8002) resolves an account's embedding in three tiers:

| Tier | Source | Behavior |
|------|--------|----------|
| 1 | Redis cache | Key `embed:<account_id>`, 6-hour TTL when written by the service. Sub-millisecond. |
| 2 | In-memory copy of `embeddings_store.parquet` | Loaded once per worker at startup. A hit backfills Redis. |
| 3 | Live single-account forward pass | For accounts outside the trained graph. Bounded to 5 concurrent passes. |

`warm_redis_cache.py` pre-loads Redis from the parquet once, so the first requests already hit Tier 1. The consumer then concatenates the 12 behavioral features it computes in-process with the 64-dimensional embedding and scores the 76-feature vector with `xgb_hybrid_model.pkl`.

Redis is configured with a 512 MB limit and LRU eviction (`docker-compose.yml`). If entries are evicted, the next request falls back to Tier 2, so results stay the same and only latency changes.

**Tier 3 is an approximation, not the trained graph model.** For an account that is not in the graph, the service applies the trained layer weights to that single account's features and averages whatever neighbor embeddings are already available, because there is no graph around a brand-new account. Responses from this path are flagged `is_provisional: true`. Accounts from the IBM dataset replay are served by Tiers 1 and 2; Tier 3 matters for new accounts arriving through the webhook receiver.

The service loads `best_gnn_model.pt` with `strict=True`, remapping the PyG layer names (`conv1.lin.weight`, `edge_proj.*`) to the service's layer names, so a naming mismatch fails at startup instead of silently leaving layers untrained.

---

## 4. Which model and configuration the pipeline uses

### Active (used by the main pipeline)

| File | Role |
|------|------|
| `xgb_hybrid_model.pkl`, `scaler_hybrid.pkl`, `hybrid_model_config.json` | Production scorer: XGBoost on 76 features (12 behavioral + 64 GNN embedding), threshold 0.3070. Loaded by `consumer.py`. |
| `best_gnn_model.pt` | GNN weights, loaded by `gnn_service.py`. |
| `embeddings_store.parquet` | Pre-computed account embeddings (Tier 2). |
| `cep_engine.py` | Rule-based CEP engine, called from `consumer.py`. |
| `dashboard_v2_light_theme.py` | The dashboard launched by `demo.ps1`. |

### Historical or experimental (not used by the main pipeline)

| File | What it is |
|------|------------|
| `model_config.json` | PaySim-trained XGBoost config (`xgb_final_v1`, AUC-ROC 0.9148), from the first modeling phase. The model files for it (`xgb_final_model.pkl`, `scaler.pkl`) are not distributed. |
| `ibm_model_config.json` | IBM-only XGBoost baseline (12 features, AUC-ROC 0.7857), from `03b_IBM_ML.ipynb`. The reference point for the hybrid model's improvement. |
| `api.py` | Standalone scoring API with SHAP explanations, built on the PaySim model. Needs `xgb_final_model.pkl` and `scaler.pkl`, which are not in the repository, so it will not start from a fresh clone. Legacy. |
| `dashboard.py` | Earlier dashboard, with SHAP attribution and STR (suspicious transaction report) generation. Still the only dashboard with those features; run with `streamlit run dashboard.py`. |
| Notebooks 01-03b | Exploration and baseline models on PaySim and IBM AML. |

---

## 5. Running components individually

`demo.ps1` is the recommended route. To start each piece by hand (Windows PowerShell, from the repository root):

**Docker services**

```powershell
docker compose up -d
```

Starts `zookeeper`, `kafka` (localhost:9092) and `redis` (localhost:6379). The compose file also defines a `postgres` container left over from an earlier design; nothing in the pipeline uses it (MLOps uses SQLite).

**Kafka topic.** The producer and consumers read the topic name from `active_topic.txt`:

```powershell
$t = "transactions-raw-" + (Get-Date -Format "yyyyMMddHHmmss")
docker exec kafka kafka-topics --create --topic $t --bootstrap-server localhost:9092 --partitions 4 --replication-factor 1
[System.IO.File]::WriteAllText("$PWD\active_topic.txt", $t)
```

**GNN service** (start after Docker, since it connects to Redis at import time)

```powershell
.\venv311\Scripts\python.exe start_gnn.py
```

Wait for `"persistent_store_loaded": true` at http://localhost:8002/stats. Memory note: run it with 2 workers or fewer on a 16 GB machine.

**Warm the Redis cache**

```powershell
.\venv311\Scripts\python.exe warm_redis_cache.py
```

**Consumers** (one window each; run 2)

```powershell
.\venv\Scripts\python.exe consumer.py
```

**Producer**

```powershell
.\venv\Scripts\python.exe producer.py
```

**Dashboard**

```powershell
.\venv\Scripts\streamlit.exe run dashboard_v2_light_theme.py
```

**Optional services**

| Service | Command | Port |
|---------|---------|------|
| Bank webhook receiver (SBI / Kotak adapters) | `.\venv\Scripts\python.exe webhook_receiver.py` | 8001 |
| Legacy scoring API | see Section 4 (needs extra files) | 8000 |

### Configuration

| Setting | Where |
|---------|-------|
| Redis and Kafka addresses (localhost:6379, localhost:9092) | Hard-coded in `consumer.py`, `producer.py`, `gnn_service.py`, dashboards |
| Kafka topic | `active_topic.txt`, written by `demo.ps1`, read through `topic_config.py` |
| GNN service address (localhost:8002) | `consumer.py` |
| Risk tiers | `consumer.py`: L1 review at 0.30, L2 soft hold at 0.50, L3 auto freeze at 0.70 |
| Bank schema mappings | `schema_configs/sbi_finacle.yaml`, `schema_configs/kotak_cbs.yaml` |
| Dashboard theme | `.streamlit/config.toml` |

---

## 6. System architecture

```
Bank CBS (SBI Finacle / Kotak CBS)
        |
Schema Adapter (schema_adapter.py, webhook_receiver.py)
        |
Kafka (4 partitions, timestamped topic per session)
        |
Consumer Pipeline (consumer.py) - batch mode, size 100
    |-- Account state            -> Redis (rolling 24h window)
    |-- Behavioral features (12) computed in-process
    |-- GNN embeddings (64)      -> batch call to gnn_service.py
    |-- XGBoost scoring (76)     in-process
    |-- CEP engine (10 patterns) -> Redis rolling windows
    |-- Alerts                   -> Redis -> Dashboard
        |
GNN Microservice (gnn_service.py, port 8002)
    |-- Tier 1: Redis cache
    |-- Tier 2: persistent store (embeddings_store.parquet, 579,735 embeddings)
    |-- Tier 3: live forward pass (approximation, unknown accounts only)
        |
Dashboard (dashboard_v2_light_theme.py, Streamlit, port 8501)
    |-- Dashboard, Live Monitoring, Flagged Accounts, Hop Map
    |-- Not yet built: Risk Analysis, Reports, Settings
        |
MLOps (mlops.py)
    |-- PSI-based drift detection
    |-- Analyst-feedback retraining (5x sample weights)
    |-- Model registry (SQLite)
```

### Dashboard pages

| Page | Status |
|------|--------|
| Dashboard | Built |
| Live Monitoring | Built. Real-time feed plus Kafka and GNN health panel; reads consumer lag through `docker exec`, so Docker must be running. |
| Flagged Accounts | Built. Search, tier and pattern filters, CSV export of the filtered set. |
| Hop Map | Built. |
| Risk Analysis, Reports, Settings | Placeholders (not yet built) |

### MuleGNN architecture

```
Input: 15 node features (12 behavioral + 3 graph-level)
GCNConv(15 -> 64)         -- 1-hop neighborhood aggregation
GCNConv(64 -> 64)         -- 2-hop neighborhood aggregation
EdgeProp Linear(2 -> 64)  -- edge feature integration
FC(128 -> 64)             -- embedding layer (output stored as the 64-d embedding)
FC(64 -> 2)               -- fraud probability output
Training: 300 epochs, Adam lr=0.01, weighted CrossEntropyLoss
```

---

## 7. Results

### Hybrid XGBoost + GNN (production model)

| Metric | Value |
|--------|-------|
| AUC-ROC | **0.8516** |
| F1 score | 0.1436 |
| Optimal threshold | 0.3070 |
| Features | 76 (12 behavioral + 64 GNN embeddings) |
| Improvement vs standalone XGBoost on IBM AML | +6.59 points AUC-ROC (0.7857 -> 0.8516) |

### CEP pattern engine

`cep_engine.py` runs **10 active patterns**: FAN_IN, FAN_OUT_SPIKE, GATHER_SCATTER, SCATTER_GATHER, SIMPLE_CYCLE, STRUCTURING, RANDOM_STRUCTURING, SMURFING_RING, TEMPORAL_LAYERING and CROSS_BANK_ACCELERATION. Eight were validated empirically against IBM AML ground-truth labels over 6.9M transactions (offline validator `cep_duckdb_validator.py`). Measured results for six of them:

| Pattern | Flagged | Confirmed | Precision | Lift |
|---------|---------|-----------|-----------|------|
| GATHER_SCATTER | 16 | 15 | 93.75% | 124.8x |
| FAN_OUT_SPIKE | 31 | 15 | 48.39% | 64.4x |
| SIMPLE_CYCLE | 116 | 15 | 12.93% | 17.2x |
| FAN_IN | 9,041 | 594 | 6.57% | 8.7x |
| STRUCTURING | 637 | 26 | 4.08% | 5.4x |
| SCATTER_GATHER | 176 | 6 | 3.41% | 4.5x |

As reported by the validator for these patterns: 6.53% precision overall, 8.7x lift over the base rate, 28 seconds for 6.9M rows.

**Retired patterns.** Three were removed from the engine after empirical diagnosis: RAPID_FORWARD (unfalsifiable at the dataset's minute-level timestamp resolution), CONCENTRATED_DESTINATION (near-zero fraud/non-fraud separation) and BURST_ACTIVATION (zero fires in the full validation run). Three further candidates were dropped during validation: PASS_THROUGH_RATIO (directionally backwards), AMOUNT_SYMMETRY (no signal, 0.83% precision at base rate) and LAYERING_CHAIN (identical ratio distributions).

### Throughput

Throughput has not been stabilized. Measured values on the development machine (16.8 GB RAM, Docker stack running) fall in a range of **25 to 90 transactions per second** depending on configuration. Earlier figures of several hundred TPS cited in project material were not reproduced and should not be relied on. A throughput regression after adding the CEP engine is unresolved (a revert to `SIMPLE_CYCLE` measured 35 TPS). The batched GNN lookup removed the per-transaction HTTP cost that capped the original consumer at about 27 TPS; the remaining bottleneck is under investigation. In the fresh-clone run of 2026-10-21 the two consumers together scored 42.8 TPS over one 60-second window while the producer wrote at roughly 170 messages per second, so consumer lag grows for as long as the producer runs.

---

## 8. Repository map

### Pipeline (runtime)

| File | Description |
|------|-------------|
| `demo.ps1` | Single-command launcher (see Section 1) |
| `docker-compose.yml` | Zookeeper, Kafka, Redis (and an unused Postgres) |
| `producer.py` | Streams the IBM AML CSV into the Kafka topic |
| `consumer.py` | Kafka consumer: state, features, GNN lookup, XGBoost scoring, CEP, risk tiers |
| `gnn_service.py`, `start_gnn.py` | GNN microservice and its launcher |
| `warm_redis_cache.py` | Loads `embeddings_store.parquet` into Redis |
| `export_embeddings.py` | Generates `embeddings_store.parquet` from the trained GNN |
| `cep_engine.py`, `account_baseline.py` | CEP engine and the per-account baseline module it imports |
| `topic_config.py` | Reads the active Kafka topic name |
| `dashboard_v2_light_theme.py` | Active dashboard |
| `mlops.py` | Drift detection (PSI), retraining, model registry |
| `schema_adapter.py`, `webhook_receiver.py`, `schema_configs/` | Bank schema adapters and webhook ingestion |
| `requirements.txt`, `requirements-gnn.txt`, `requirements-notebooks.txt` | Dependencies per environment |

### Notebooks

| Notebook | Description |
|----------|-------------|
| `01_EDA.ipynb`, `02_features.ipynb`, `03_ML_classification.ipynb` | PaySim exploration, features, classification |
| `02b_IBM_EDA.ipynb`, `03b_IBM_ML.ipynb` | IBM AML features and baseline ML |
| `04_graph_construction.ipynb` | Transaction graph construction |
| `05_GNN.ipynb` | GNN training, embedding extraction, hybrid model training |

### Validation and demonstration tools (optional, not part of the pipeline)

| File | Description |
|------|-------------|
| `cep_duckdb_validator.py` | Offline pattern validation with DuckDB (6.9M rows in about 28 s) |
| `cep_polars_validator.py` | Offline validation with Polars |
| `cep_offline_validator.py`, `cep_v3_validator.py` | Offline CEP replays through `cep_engine.py` |
| `cep_ground_truth_check.py`, `cep_summary.py` | Ground-truth comparison and post-run summaries |
| `cep_diagnostic.py`, `cep_diagnostic_v2.py` | Diagnostics used to retire patterns |
| `cross_bank_simulation.py`, `run_cross_bank_demo_fast.py` | Cross-bank federated alerting simulation (`python cross_bank_simulation.py 100000`) |
| `reset_demo_environment.ps1`, `monitor_ram_logged.ps1` | Environment reset and RAM logging helpers |

The validators additionally need `duckdb` and `polars`, which are not in the requirements files: `pip install duckdb polars`.

### Legacy

`api.py`, `dashboard.py`, `model_config.json`, `ibm_model_config.json`. See Section 4.

---

## 9. Troubleshooting

| Symptom | Cause and fix |
|---------|---------------|
| `demo.ps1` is blocked from running | Run `Set-ExecutionPolicy -Scope Process Bypass` first. |
| `ABORT: Required files are missing` | The listed files are not in the repository root. See Section 2 (dataset and embedding store). |
| `ConnectionError ... localhost:6379` when starting the GNN service | Redis is not running. Start Docker Desktop, then `docker compose up -d`. |
| GNN service starts but `persistent_store_loaded` is `false` | `embeddings_store.parquet` is missing or empty. Place the file in the repository root and restart the service. |
| GNN service restarts repeatedly | Memory pressure or slow startup. Use 2 workers or fewer and close other applications. |
| `FileNotFoundError: xgb_hybrid_model.pkl` | You are not running from the repository root. |
| Pickle loading errors or warnings | Library versions differ from the pinned ones. Reinstall from `requirements.txt` in a clean environment. |
| `Could not load IBM AML labels` in the consumer | `ibm_aml_small.csv` is missing. The consumer still runs but cannot mark confirmed laundering accounts. |
| Dashboard Live Monitoring shows no Kafka data | Docker is not running, or no consumer has subscribed yet. |

---

## 10. Known limitations

- **Throughput** is 25-90 TPS on the development machine, with an unresolved regression after adding CEP (see Section 7).
- **SMURFING_RING** shows an N/A z-score in the integrated pipeline although it passes in isolated validation. The cause has not been identified.
- **Tier 3 embeddings** are an approximation for accounts outside the trained graph (see Section 3).
- **The pre-authorization API endpoint** exists in the code but has not been exercised with a live HTTP request.
- **Dashboard:** Risk Analysis, Reports and Settings pages are placeholders. SHAP attribution and STR generation exist only in the legacy `dashboard.py`.
- **Federated alerting, not federated learning.** The cross-bank simulation shares account-level alert signals between banks. It does not share model gradients. A national-scale design (per-bank models aggregated into a national score) conflicts with this and is an open design decision.
- **Cold start:** the tiered framework for banks with little or no history is designed but not implemented.
- **Model hot-swap during retraining** (continuing to score while a new model is trained) is not implemented.
- **Platform:** only the Windows/PowerShell launch path has been tested.
- **Hard-coded local addresses** for Redis, Kafka and the GNN service (see Section 5).
- **Dataset caveat:** IBM AML is synthetic. Performance on real bank data will differ.

---

## 11. Validation status

Fresh-clone end-to-end run: **2026-10-21; Windows (build 10.0.26200), Windows PowerShell 5.1, Python 3.14.4 and 3.11, Docker Desktop; repository commit `bd58db6`. Outcome: completed, with the caveats listed below.**

| Check | Result |
|-------|--------|
| Clean clone into an empty folder | Passed. Cloned from GitHub and updated to `bd58db6`. |
| Both environments installed from the requirements files | Passed. Import checks succeeded in both (`venv`; `venv311` with torch 2.5.1+cu121). The pip output was not captured in the run transcript. |
| Embedding store placed per Section 2 | Passed. Downloaded from the Release URL; the SHA-256 of all four artifacts matched the table in Section 2. |
| Dataset placed per Section 2 | Not re-validated. `ibm_aml_small.csv` was copied from an earlier Kaggle download, so the Kaggle steps in Section 2 were not re-run in this test. |
| `demo.ps1 -producers` completed all phases | Passed on the fourth attempt (earlier attempts stopped at Phase 0, 1 and 2; see below). Kafka topic created with 4 empty partitions; GNN service healthy; Redis warmed; 2 consumers, dashboard and producer started. |
| Transactions scored | Passed. `/stats` reported `persistent_store_loaded: true` with 579,735 embeddings; the consumer group `mule-detector-prod` consumed 1,400 messages in the first minutes and 2,570 in a 60-second window (42.8 TPS with 2 consumers) while the producer had written 52,319. Consumer lag grew (Section 7). |
| Alerts visible on the dashboard | Not verified in this run. The dashboard process was started in Phase 5; its pages were not inspected. |

### Errors encountered and how they were resolved

1. **Environments without packages.** Environments existed but the packages were not installed after an interrupted install. Resolved by running the `pip install` commands in Section 1 to completion.
2. **`ibm_aml_small.csv` missing.** Phase 0 of `demo.ps1` stopped and named the file (expected behaviour). Resolved by placing the file in the repository root.
3. **Docker not running.** Phase 1 stopped with a message to start Docker Desktop (expected behaviour).
4. **Kafka crash after a Docker restart.** Kafka exited at startup with `NodeExists` because ZooKeeper still held the previous broker registration. `demo.ps1` now waits for the broker and restarts it if it is not running. The crash was reproduced by killing and restarting the broker; the recovery path of the fixed launcher was not re-run end to end.
5. **Kafka topic never created.** The topic-creation command was missing from `demo.ps1`, so the offset check failed. Fixed in commit `bd58db6` (create with retries); the following run passed.
6. **Cosmetic.** On a first run PowerShell prints a red `docker : error: no such object: redis` before the launcher starts the containers, and the closing banner lists "Start producer" hints even with `-producers`. Neither affects the result.

### Not covered by this run

The recovery path in item 4, the Kaggle download steps, the dashboard pages (unless stated in the table), and flagged-account output were not exercised here.

---
## 12. Background

### Datasets

**IBM AML Small (primary)**

- 6,924,049 transactions, 705,903 unique accounts
- 5,304 confirmed laundering accounts (0.7514% base rate)
- Download: https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml

**PaySim (exploration only)**

- 6,362,620 transactions, 0.1291% fraud rate
- Download: https://www.kaggle.com/datasets/ealaxi/paysim1

### What is a mule account?

A mule account is a bank account used by criminal networks to receive and forward stolen funds, creating an obfuscation layer between the crime and the criminal. Detection is difficult because mule accounts pass KYC at opening, remain dormant before activation, individually show no transaction above reporting thresholds, and reveal their nature only when analysed as a coordinated network.

### Deployment roadmap (planning, not delivered)

- **Phase 1: SBI and Kotak pilot.** The current Python architecture on bank infrastructure.
- **Phase 2: top 10 banks.** Apache Flink replacing the Python consumer; federated alerting (account-level signal sharing) across participating banks; broader UPI coverage.
- **Phase 3: national deployment.** All UPI-connected banks, NPCI integration for cross-bank graph analysis, and a temporal GNN updated in real time.

### Regulatory alignment

PMLA 2002 | RBI Master Directions on KYC (2024) | RBI Governor Directive on Mule Accounts (July 2024) | DPDP Act 2023 | RBI FREE-AI Framework (August 2025) | FATF Recommendation 16

### Research foundation

| Reference | Contribution |
|-----------|-------------|
| Altman et al. (2023). Realistic Synthetic Financial Transactions for AML. ArXiv:2306.16424 | IBM AML dataset, typology definitions |
| Tam et al. (2019). Identifying Illicit Accounts in Large Scale E-payment Networks. ArXiv:1906.05546 | EdgeProp mechanism |
| NVIDIA AI Blueprint for Fraud Detection (2024) | GNN + XGBoost hybrid architecture validation |
| RBI Governor Directive (July 2024) | Regulatory urgency, PMLA alignment |
| RBI FREE-AI Framework (August 2025) | Explainability compliance |

### Author

**Anish Banerjee**
B.Tech Computer Science, DBS Global University
B.Sc. Data Science, IIT Madras
CyberPeace Foundation Research Wing Intern, June-September 2026
GitHub: [anishbanerjee05](https://github.com/anishbanerjee05)
