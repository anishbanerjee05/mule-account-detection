# Mule Account Detection System

A four-phase machine learning architecture for detecting money mule accounts in financial transaction networks. Built as part of a research internship at CyberPeace Foundation, with planned production deployment at SBI and Kotak Bank (August 2026).

\---

## What is a Mule Account?

A mule account is a bank account used by criminal networks to receive and forward stolen funds, creating an obfuscation layer between the crime and the criminal. Detection is challenging because mule accounts:

* Pass KYC verification at account opening
* Remain dormant before activation
* Individually show no single transaction above reporting thresholds
* Only reveal their criminal nature when analysed as a coordinated network

\---

## Architecture Overview

The system is structured as a four-phase detection funnel. Each phase addresses the specific failure mode of the previous one.

|Phase|Method|Dataset|Key Result|
|-|-|-|-|
|Phase 1|Feature Engineering + Rule Engine|PaySim + IBM AML|20 features, F1 baseline 0.0231|
|Phase 2|XGBoost + SHAP|PaySim + IBM AML|F1 0.2956 (PaySim), F1 0.0796 (IBM AML)|
|Phase 3|Kafka + Flink + Behavioral Biometrics|—|In progress — target July 2026|
|Phase 4|GCN + EdgeProp + Ring Detection|IBM AML|AUC-ROC 0.8140, 590 rings detected|

\---

## Performance Results

### PaySim Dataset

|Model|F1|AUC-ROC|Improvement vs Baseline|
|-|-|-|-|
|Rule Engine (baseline)|0.0231|—|1.0x|
|Logistic Regression|0.0210|0.7897|-0.9x|
|Random Forest|0.0341|0.8835|1.5x|
|XGBoost (default threshold)|0.0640|0.9148|2.8x|
|**XGBoost (optimal threshold 0.967)**|**0.2956**|**0.9148**|**12.8x**|

### IBM AML Dataset

|Model|F1|AUC-ROC|Improvement vs Baseline|
|-|-|-|-|
|Rule Engine (baseline)|0.0155|—|1.0x|
|Logistic Regression|0.0177|0.7016|1.1x|
|Random Forest|0.0291|0.7944|1.9x|
|XGBoost (optimal threshold 0.708)|0.0796|0.7857|5.1x|
|**GNN (optimal threshold 0.845)**|**0.0822**|**0.8140**|**5.3x**|

### Graph Neural Network — Ring Detection

|Metric|Value|
|-|-|
|Total accounts flagged|2,889|
|Rings detected (2+ connected flagged accounts)|590|
|Rings with 10+ accounts|8|
|Largest ring detected|385 accounts (80 confirmed laundering)|
|XGBoost rings detected|0|

\---

## Datasets

### PaySim

* 6,362,620 transactions over 30 days
* 0.1291% fraud rate (8,213 fraud cases)
* Fraud only in TRANSFER and CASH\_OUT transaction types
* Download: https://www.kaggle.com/datasets/ealaxi/paysim1

### IBM AML Small

* 6,924,049 transactions
* 0.0515% laundering rate (3,565 cases)
* 41,814 unique sender banks — real cross-institutional structure
* Fan-out degree 12.16x fraud/legit separation
* Download: https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml

\---

## Key Findings

**Finding 1 — Search space reduction:** Fraud only occurs in TRANSFER and CASH\_OUT transactions on PaySim — reduces detection scope by 56% using domain knowledge alone, before any model is applied.

**Finding 2 — Balance drain:** 98.1% of fraud transactions drain the sender balance to exactly zero. 49.8% of destination accounts also drain immediately — confirming the hopping/layering pattern.

**Finding 3 — Fan-out degree:** 1.00x separation on PaySim (synthetic limitation) vs 12.16x on IBM AML. Real mule accounts send to 24.5 unique destinations on average vs 2.0 for legitimate accounts.

**Finding 4 — Feature importance is dataset-dependent:** Top SHAP features differ completely across datasets. PaySim: account\_age\_at\_activation (1.56), max\_amount (0.96). IBM AML: dormancy\_ratio (1.59), total\_transactions (1.50). session\_burst\_score = 0.000 on PaySim but 0.731 on IBM AML. No single feature universally dominates — validates the multi-layer architecture.

**Finding 5 — Threshold optimisation:** XGBoost F1 improves from 0.0640 to 0.2956 on PaySim purely through threshold selection (0.50 → 0.967). The model flags 975 accounts instead of 35,400 with 39.7% precision at the optimal threshold.

**Finding 6 — GNN beats XGBoost on AUC-ROC:** GNN AUC-ROC 0.8140 vs XGBoost 0.7857 on IBM AML. Graph structure adds measurable discriminative value. GNN crosses the XGBoost baseline at epoch 80 and maintains superiority through epoch 300.

**Finding 7 — 590 coordinated rings invisible to ML:** The GNN detected 590 mule rings that XGBoost could not see. The largest ring has 385 accounts — each individually scoring below the alert threshold, but the star topology is unmistakable at the graph level.

\---

## Engineered Features (20 total)

|Feature|PaySim Signal|IBM AML Signal|
|-|-|-|
|Velocity ratio|1.05x|0.99x|
|Balance retention rate|1.17x|3.48x|
|Fan-out degree|1.00x|12.16x|
|Cross-bank degree|N/A|1.81x|
|Dormancy spike score|3.22x|N/A|
|Round amount ratio|31.00x|0.01x|
|Structuring score|0.64x|1.88x|
|Dormancy ratio|3.04x|0.66x|
|Total transactions|1.00x|21.28x|
|Session burst score|1.00x|0.73x (SHAP)|

\---

## MuleGNN Architecture

```
Input: 15 node features (12 engineered + 3 graph-level)
GCNConv(15 → 64)          — 1-hop neighborhood aggregation
GCNConv(64 → 64)          — 2-hop neighborhood aggregation
EdgeProp Linear(2 → 64)   — edge feature integration (Tam et al. 2019)
FC(128 → 64)              — classification head
FC(64 → 2)                — fraud probability output
Total parameters: 13,762
GPU: NVIDIA RTX 4050 6.4GB VRAM, CUDA 12.1
Training: 300 epochs, Adam lr=0.01, weighted CrossEntropyLoss (pos\\\_weight=242)
```

\---

## Notebooks

|Notebook|Description|Status|
|-|-|-|
|01\_EDA.ipynb|Exploratory Data Analysis — PaySim|✅ Complete|
|02\_features.ipynb|Feature Engineering — PaySim (20 features)|✅ Complete|
|02b\_IBM\_EDA.ipynb|Feature Engineering — IBM AML (13 features)|✅ Complete|
|03\_ML\_classification.ipynb|ML Pipeline — PaySim|✅ Complete|
|03b\_IBM\_ML.ipynb|ML Pipeline — IBM AML|✅ Complete|
|04\_graph\_construction.ipynb|Graph Construction — IBM AML|✅ Complete|
|05\_GNN.ipynb|GNN Training — IBM AML (GPU)|✅ Complete|

\---

## Production Deployment Timeline

Target: **July 29, 2026** — fully containerised system ready for SBI and Kotak Bank integration.

|Week|Period|Deliverable|
|-|-|-|
|1|June 13–19|Docker infrastructure — Kafka, Redis, PostgreSQL|
|2|June 20–26|Schema adapter + Kafka producer + normalisation consumer|
|3|June 27–July 3|FastAPI model server (XGBoost + GNN + SHAP endpoints)|
|4|July 4–10|Flink VelocityScorer + CEP Pattern Detector|
|5|July 11–17|MLOps pipeline — drift monitor, retraining, model registry|
|6|July 18–24|Grafana dashboard + React analyst workstation + STR generator|
|7|July 25–29|SBI schema adapter + full demo polish|

\---

## Setup

```bash
git clone https://github.com/anishbanerjee05/mule-account-detection.git
cd mule-account-detection

# Phase 1-2: Python 3.14 (CPU)
python -m venv venv
venv\\\\Scripts\\\\activate
pip install -r requirements.txt

# Phase 4 (GNN): Python 3.11 + CUDA
# Requires NVIDIA GPU with CUDA 12.1
C:\\\\Users\\\\ASUS\\\\AppData\\\\Local\\\\Programs\\\\Python\\\\Python311\\\\python.exe -m venv venv311
venv311\\\\Scripts\\\\activate
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install torch\\\_geometric pandas numpy matplotlib scikit-learn xgboost shap networkx jupyter ipykernel
```

Download datasets to the project root:

* PaySim: `paysim.csv/PS\\\_20174392719\\\_1491204439457\\\_log.csv`
* IBM AML: `ibm\\\_aml\\\_small.csv`

\---

## Research Foundation

|Reference|Contribution to Architecture|
|-|-|
|Tam et al. (2019). Identifying Illicit Accounts in Large Scale E-payment Networks. ArXiv:1906.05546|EdgeProp mechanism, i.i.d. limitation analysis|
|Liu et al. (2018). Heterogeneous GNN for Malicious Account Detection. ACM CIKM 2018|Alipay deployment evidence, heterogeneous graph design|
|Sultana et al. (2025). detectGNN: GNNs for Enhanced Fraud Detection. ArXiv:2503.22681|Performance benchmarks|
|RBI Governor Directive. (July 2024)|Regulatory urgency, PMLA alignment|
|MediaNama RTI Report. (December 2025)|MuleHunter.AI capabilities documentation|
|RBI FREE-AI Framework. (August 2025)|Explainability compliance requirements|

\---

## Regulatory Alignment

PMLA 2002 | RBI Master Directions | RBI Governor Directive July 2024 | DPDP Act 2023 | RBI FREE-AI Framework August 2025

\---

## Author

**Anish Banerjee**
B.Tech Computer Science, DBS Global University
B.Sc. Data Science, IIT Madras
CyberPeace Foundation Research Wing Intern, June–September 2026

GitHub: [anishbanerjee05](https://github.com/anishbanerjee05)

