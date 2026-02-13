# CLD-ML-Pipeline

Simulation-driven clone selection framework for Cell Line Development (CLD)

## Vision

Cell Line Development (CLD) remains one of the most resource-intensive stages in biologics manufacturing.
Despite advances in AI for drug discovery, predictive modeling for biomanufacturing platforms is still underdeveloped.

This project aims to build a Simulation-Driven Laboratory (SDL) framework for CLD, enabling:
	•	Early-stage clone ranking
	•	Late-stage outcome prediction
	•	Multi-objective decision simulation
	•	Platform engineering scenario testing

The long-term goal is to reduce CLD time, cost, and experimental burden using predictive modeling.

## Concept

We simulate a realistic CLD process including:
	•	Clone-to-clone productivity variability (log-normal)
	•	Stability-driven productivity decay
	•	Expression burden effects
	•	Stress accumulation
	•	Aggregation (quality proxy)
	•	Batch effects
	•	Culture mode differences (fed-batch vs perfusion)
	•	Copy number (ddPCR-like)
	•	Platform factor (targeted integration vs legacy)
	•	Clone-specific decay heterogeneity

The framework allows:

Early data (passages 3–10) → Late prediction (passages 24–30)

## Project Architecture

Synthetic DB Generation
        ↓
Feature Engineering (Early-only)
        ↓
Multi-target ML Models
        ↓
Ranking & Top-K Evaluation
        ↓
Decision Simulation (Retention %)
        ↓
Platform Scenario Testing

## Repository Structure

data/
    schema/
    synthetic/
        raw/
        processed/

notebooks/
    01_explore_synthetic_data.ipynb
    02_feature_engineering.ipynb
    02d_feature_engineering_v2.ipynb
    02c_build_late_labels.ipynb
    03b_multitarget_models.ipynb
    04_clone_drop_simulation.ipynb
    04b_predicted_late_selection.ipynb

src/
    data_generation/
        generate_synthetic_cld.py

## Synthetic Data Generator

The generator simulates:

**Latent variables per clone:**
	•	Productivity (P)
	•	Stability (S)
	•	Quality potential (Q)
	•	Platform factor (G)
	•	Copy number (CN)
	•	Clone-specific decay sensitivity (k_decay_i)

**Assays:**
	•	Titer
	•	VCD
	•	Viability
	•	Aggregation
	•	ddPCR copy number

**Derived label:**
	•	Productivity drop % (early vs late)

The generator supports scenario switches:
enable_platform=True
enable_copy_number=True
enable_clone_decay_variation=True

This allows ablation and sensitivity analysis.

## 📈 Modeling Strategy

We train 3 regression models:
	1.	Stability drop
	2.	Late-stage titer
	3.	Late-stage aggregation

Evaluation includes:
	•	R²
	•	MAE
	•	Spearman rank correlation
	•	Top-K overlap
	•	Top-K enrichment rate

## 🎯 Multi-Objective Selection

We simulate commercial clone selection using:

'''Utility = a*z(late_titer)
        - b*z(drop)
        - c*z(late_aggregation)'''

We evaluate:
	•	Top 5 / 10 / 20 retention
	•	True-good recall
	•	Ranking robustness



## License
Apache License 2.0
