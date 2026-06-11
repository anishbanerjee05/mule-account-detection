\# Mule Account Detection System



A four-phase machine learning architecture for detecting money mule accounts in financial transaction networks. Built as part of a research internship at CyberPeace Foundation.



\## What is a Mule Account?

A mule account is a bank account used by criminal networks to receive and forward stolen funds, creating an obfuscation layer between the crime and the criminal. Detection is challenging because mule accounts pass KYC verification, remain dormant before activation, and individually show no obvious fraud signals.



\## Architecture Overview

The system is structured as a detection funnel across four progressive phases:



| Phase | Method | Capability |

|-------|--------|------------|

| Phase 1 | Feature Engineering + Rule Engine | Baseline detection, PMLA-aligned rules |

| Phase 2 | XGBoost + SHAP | ML classification with explainability |

| Phase 3 | Kafka + Flink + Behavioral Biometrics | Real-time streaming detection |

| Phase 4 | GCN + EdgeProp | Graph-level mule ring detection |



\## Dataset

PaySim — synthetic mobile money transaction dataset modelled on real M-Pesa data.

\- 6,362,620 transactions over 30 days

\- 0.13% fraud rate

\- Download from: https://www.kaggle.com/datasets/ealaxi/paysim1



\## Key Findings from EDA

\- Fraud ONLY occurs in TRANSFER and CASH\_OUT transactions (56% search space reduction)

\- 98.1% of fraud transactions drain the sender balance to exactly zero

\- 99.8% of fraud accounts send to exactly 1 unique destination

\- Average fraud amount is 4.7x higher than legitimate transactions

\- 25 temporal spike events detected above 2-sigma threshold



\## Research Foundation

\- Tam et al. (2019) — Identifying Illicit Accounts in Large Scale E-payment Networks. ArXiv:1906.05546

\- Liu et al. (2018) — Heterogeneous GNN for Malicious Account Detection. ACM CIKM 2018

\- Sultana et al. (2025) — detectGNN: GNNs for Enhanced Fraud Detection. ArXiv:2503.22681



\## Notebooks

| Notebook | Description | Status |

|----------|-------------|--------|

| 01\_EDA.ipynb | Exploratory Data Analysis | Complete |

| 02\_features.ipynb | Feature Engineering | In Progress |

| 03\_ML\_classification.ipynb | ML Pipeline | Pending |

| 04\_graph\_construction.ipynb | Graph Construction | Pending |

| 05\_GNN.ipynb | Graph Neural Network | Pending |



\## Setup

```bash

git clone https://github.com/anishbanerjee05/mule-account-detection.git

cd mule-account-detection

python -m venv venv

venv\\Scripts\\activate

pip install -r requirements.txt

```



\## Regulatory Alignment

PMLA 2002 | RBI Master Directions | RBI Governor Directive July 2024 | DPDP Act 2023 | RBI FREE-AI Framework August 2025



\## Author

Anish Banerjee

B.Tech CSE, DBS Global University | B.Sc. Data Science, IIT Madras

CyberPeace Foundation Research Wing Intern, June–September 2026

