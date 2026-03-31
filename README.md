# 📦 CLD ML Pipeline

## 🧬 Overview

This project simulates a realistic **Cell Line Development (CLD)** workflow and builds machine learning models to:

- Predict **late-stage clone performance** from early screening data
- Optimize **clone selection strategies** under different business objectives
- Compare **legacy vs optimized platform behavior**
- Provide a sandbox for testing **next-gen interventions** (e.g., CRISPR, multi-omics)

The pipeline includes:

- Synthetic data generator (biologically grounded)
- Feature engineering (early passage signals → model-ready features)
- ML models (regression + ranking)
- Utility-based clone selection framework

---

## 🎯 Goals

- Reproduce realistic CLD phenomena:
  - Productivity decay across passages
  - Trade-offs between growth, productivity, and quality
  - Hidden late-stage liabilities
- Evaluate how well early data predicts:
  - Stability (titer drop)
  - Late productivity
  - Late aggregation (quality)
- Simulate different development strategies:
  - **Biosimilar mode** (productivity-focused)
  - **Novel/ADC mode** (quality-aware)

---

## 🏗️ Project Structure
'''
CLD_ML_PIPELINE/
│
├── src/
│   ├── data_generation/
│   │   └── generate_synthetic_cld.py
│   ├── feature_engineering/
│   ├── modeling/
│   └── evaluation/
│
├── notebooks/
│   ├── 02c_feature_engineering.ipynb
│   ├── 02d_modeling.ipynb
│   └── 03b_evaluation.ipynb
│
├── data/
│   ├── schema/
│   └── synthetic/
│
└── README.md
'''
---

## 🧪 Synthetic Data Generator

### Key Design Principles

The generator simulates biologically plausible CLD behavior:

- **Productivity (P)**: right-skewed distribution (few high producers)
- **Stability (S)**: governs decay across passages
- **Quality (Q)**: aggregation increases with stress and intrinsic liability

---

## ⚙️ Platform Scenarios

### 🔴 Legacy Platform

- High heterogeneity (position effects, silencing, integration randomness)
- Presence of:
  - **Super jackpot clones** (rare, high P + high S)
  - **Aggressive clones** (high early P, unstable later)
- Strong decoupling between early and late behavior
- Higher noise and unpredictability

---

### 🟢 Optimized Platform (✅ Frozen)

- Reduced variability (targeted integration-like behavior)
- No super jackpot clones
- Small **residual aggressive subgroup (~1%)**
- Late-stage outcomes include:
  - Hidden stochastic effects
  - Partial decoupling from early signals

---

## 📊 Modeling Targets

From early passage data (P1–P10), we predict:

| Target        | Description |
|--------------|------------|
| `drop`       | Relative titer loss (stability proxy) |
| `late_titer` | Productivity at late passages |
| `late_agg`   | Aggregation (quality proxy) |

---

## 📈 Evaluation Framework

### Metrics

- Regression:
  - MAE
  - R²
- Ranking:
  - Spearman correlation
- Selection:
  - Top-K overlap
  - Precision@K
  - NDCG@K

---

## 🧠 Utility-Based Selection

We define utility functions to simulate real decision-making:

### Biosimilar Mode
- Focus: maximize productivity
- Low penalty on aggregation

### Novel / ADC Mode
- Balanced:
  - Productivity
  - Stability
  - Quality (aggregation penalty)

---

## ✅ Current Status (Optimized Scenario)

The optimized generator has been calibrated and **frozen** with the following properties:

- Late outcomes are:
  - Predictable but **not trivial**
  - Influenced by **hidden late-only factors**
- Residual failure modes exist:
  - Small aggressive subgroup (~1%)
- Model performance:
  - Strong but imperfect ranking
  - Realistic degradation from early → late prediction

### Key Observations

- `late_titer`: moderately predictable
- `late_agg`: harder due to noise + hidden effects
- `drop`: remains difficult (biologically realistic)

---

## 🔍 Validation Checks

- ✅ No feature leakage
- ✅ Permutation tests confirm real signal usage
- ✅ Utility optimization validated on held-out test set

---

## 🚀 Next Steps

### 1️⃣ Legacy Scenario Realism Expansion

Enhance the **legacy generator** to better reflect:

- Stronger heterogeneity
- Larger clone-to-clone variability
- More pronounced early vs late decoupling
- Increased noise in quality and productivity
- Clearer distinction between:
  - Biosimilar vs Novel/ADC strategies

---

### 2️⃣ CRISPR Intervention Layer

Simulate targeted interventions:

- Knock-in / knock-out effects
- Stability improvement
- Stress reduction
- Clone rescue vs unintended trade-offs

**Goal:**
Evaluate how interventions shift clone ranking and selection outcomes

---

### 3️⃣ Multi-Omics Integration

Introduce additional data layers:

- Transcriptomics (expression burden, folding stress)
- Epigenetics (silencing risk)
- Proteomics (secretion efficiency)

**Goal:**
Improve prediction of late-stage behavior beyond early phenotypes

---

### 4️⃣ Decision Optimization

- Multi-objective optimization
- Portfolio selection under constraints
- Active learning for clone selection

---

## 💡 Long-Term Vision

Build a simulation + ML framework that can:

- Reproduce real CLD complexity
- Test experimental strategies virtually
- Quantify value of additional data (omics, assays)
- Support decision-making in bioprocess development

---

## 🧑‍🔬 Author Notes

This project is designed as a **research + simulation sandbox**, not just a modeling exercise.

> “If the model performs perfectly, the data is unrealistic.”