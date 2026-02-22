"""
generate_synthetic_cld.py - This module contains functions to generate synthetic CLD (Cloud Layer Data) for testing and development purposes.

Goal:
- Create a synthetic (virtual) CLD dataset in SQLite following my ERD.
- Provide realistic relationships:
    * Productivity is right-skewed across clones (few high producers).
    * Productivity decays across passages depending on stability.
    * High expression burden can reduce growth/viability early,
    but as productivity decays, growth/viability can recover late.
    * Quality proxy (aggregation) worsens when intrinsic quality is low and/or stress is high.
- Provide hidden truth (P/S/Q) in CSV for validation, not in the DB tables, to avoid ML leakage.

Tables populated:
- product, host_cell, vector, cell_line
- batch
- clone
- passage
- process_condition
- assay_result
- stability_test
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from dataclasses import dataclass
from datetime import date, timedelta
import argparse

import numpy as np
import pandas as pd

# --------------------------------------
# Configuration object (all knobs in one place)
# --------------------------------------
@dataclass(frozen=True)
class Config:
    # Reproducibility
    seed : int = 42

    # Dataset sizes
    n_clones : int = 2000
    n_passages : int = 30

    # Phase labeling (to prevent ML leakage)
    early_max : int = 10
    late_min : int = 26

    # Stability label window
    stability_early_start: int = 3
    stability_early_end: int = 10
    stability_late_start: int = 26
    stability_late_end: int = 30

    # Latent Distributions
    # Productivity: Log-normal distribution
    mu_logP : float = 4.0  # log-mean
    sigma_logP : float = 0.7  # log-stddev

    # Stability: Beta distribution -> bound [0, 1]
    alpha_stability : float = 7.0
    beta_stability : float = 3.0

    # Quality potential: near high, clipped to [0, 1]
    mu_quality_potential : float = 0.8
    sigma_quality_potential : float = 0.15

    # Productivity decay sensitivity
    k_decay : float = 0.05

    # --------------------------------------
    # Feature switches (ablation-ready)
    # --------------------------------------

    enable_platform: bool = True
    enable_copy_number: bool = True
    enable_clone_decay_variation: bool = True

    # --------------------------------------
    # Platform factor (project-level + clonal residual)
    # --------------------------------------
    # Interpretation:
    # - G_project models a platform upgrade (e.g., targeted integration + anti-silencing + secretion tuning)
    # - g_clone models residual clone-to-clone variability
    
    G_project: float = 0.0 # 0.0 = baseline platform, 1.0 = upgraded platform (we can change later)
    g_clone_sd: float = 0.35 # residual variability; reduce this for stronger targeted integration

    platform_strength_P: float = 0.25 # effect of platform on productivity latent P
    platform_strength_S: float = 0.20 # effect of platform on stability latent S
    platform_strength_Q: float = 0.15 # effect of platform on quality latent Q
    platform_stress_relief: float = 0.10 # platform reduces stress slightly (lower agg risk)

    # --------------------------------------
    # Platform scenario adjustments (legacy vs optimized)
    # --------------------------------------
    enable_platform_groups: bool = True
    frac_optimized: float = 0.30 # fraction of clones on optimized platform (used only if grouping is enabled)

    G_legacy: float = 0.0
    G_optimized: float = 1.0

    optimized_decay_mult: float = 0.7 # optimized platform reduces decay sensitivity by this factor
    optimized_cn_effect_mult: float = 1.25 # optimized platform increases CN effect on productivity by this factor (e.g., better expression machinery can better leverage higher CN)

    # --------------------------------------
    # Copy number (ddPCR-like assay) effect on productivity
    # --------------------------------------

    cn_mean: float = 3.0  # average copy number
    cn_sigma: float = 0.35 # log-normal sigma
    cn_max: int = 20    # max copy number
    cn_effect_P: float = 0.18 # Copy number increase P (diminishing returns via log1p)
    cn_penalty_S: float = 0.05 # Copy number slightly decrease S (higher CNV instability - burden)
    cn_penalty_Q: float = 0.03 # Copy number slightly decrease Q (higher burden)
    ddpcr_noise_sd: float = 0.35 # measurement noise in obserbved Copy Number

    # --------------------------------------
    # Clone-specific decay variation
    # --------------------------------------
    k_decay_sd: float = 0.25 # k_i = k_decay * exp(N(0, k_decay_sd)) 

    # Scaling from latent P -> physical units
    alpha_titer : float = 0.03 # latetn units to g/L
    base_vcd : float = 15e6  # cells/mL

    # Measurement noise levels (std dev) - random per data point
    titer_noise_sd : float = 0.15  # g/L
    vcd_noise_sd : float = 0.8e6   # cells/mL
    viability_noise_sd : float = 1.5  # fraction
    aggregation_noise_sd : float = 0.3  # fraction

    # Batch effect SDs (systemic offsets per passage-run)
    batch_titer_sd : float = 0.05  # g/L
    batch_vcd_sd : float = 0.3e6   # cells/mL
    batch_viability_sd : float = 0.4  # fraction
    batch_aggregation_sd : float = 0.02  # fraction

    # Burden/adaptation parameters
    burden_coeff: float = 0.35  # how strongly burden affects VCD
    adapt_vcd: float = 0.06   # VCD improvement with passage
    adapt_viab: float = 1.2    # Viability improvement with passage

    # Stress model affects quality (aggregation)
    stress_base: float = 0.08
    env_stress_slope: float = 0.002 # mild passage-related stress increase
    expr_stress: float = 0.06 # expression burden stress (depends on P_ip)

    # Quality proxy parameters
    gamma_agg_intrinsic: float = 20.0
    delta_agg_stress: float = 6.0

config = Config()

# --------------------------------------
# Helper functions
# --------------------------------------
def phase_label(p: int) -> str:
    """Label passage so ML can train on early and predict late."""
    if p <= config.early_max:
        return "early"
    elif p >= config.late_min:
        return "late"
    else:
        return "mid"
    
def load_schema_sql(schema_path:Path) -> str:
    """Load the SQL schema from file.
    
    Assumes schema _path points to a file containing only valid SQL
    (e.g., CREATE TABLE statements).
    """
    if not schema_path.exists():
        raise FileNotFoundError(f"Schema file not found: {schema_path}")
    
    return schema_path.read_text(encoding="utf-8")

def ensure_fresh_db(db_path: Path) -> None:
    """Delete existing DB so each run creates a fresh dataset."""
    if db_path.exists():
        db_path.unlink()

def apply_scenario(base: Config, scenario: str) -> Config:
    """
    Return a new Config object with scenario-specific overrides.
    We keep everything else identical for fair comparisons.
    """

    scenario = scenario.lower()
    if scenario == "legacy":
        return Config(
            **{**base.__dict__,
               "frac_optimized": 0.0,
               "g_clone_sd": 0.60,
               "k_decay_sd": 0.35,
               "platform_stress_relief": 0.05,
               "cn_sigma": 0.45,
               "cn_effect_P": 0.18,
               "cn_penalty_S": 0.06,
               "cn_penalty_Q": 0.04,
               "optimized_decay_mult": 1.0,
               "optimized_cn_effect_mult": 1.0,
            })
    elif scenario == "optimized":
        return Config(
            **{**base.__dict__,
               "frac_optimized": 1.0,
               "g_clone_sd": 0.15,
               "k_decay_sd": 0.15,
               "platform_stress_relief": 0.20,
               "cn_sigma": 0.25,
               "cn_effect_P": 0.22,
               "cn_penalty_S": 0.03,
               "cn_penalty_Q": 0.02,
               "optimized_decay_mult": 0.65,
               "optimized_cn_effect_mult": 1.25,
            })
    else:
        raise ValueError("scenario must be 'legacy' or 'optimized'")

#-----------------------------------------
# Main synthetic data generation function
#-----------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic CLD dataset.")
    parser.add_argument("--scenario", type=str, default="legacy", choices=["legacy", "optimized"], help="Which scenario configuration to use (legacy, optimized)")
    args = parser.parse_args()

    global config
    config = apply_scenario(config, args.scenario)

    np.random.seed(config.seed)

    # Repo paths
    root = Path(__file__).resolve().parents[2]
    schema_path = root / "data" / "schema" / "cld_schema.sql"
    out_dir = root / "data" / "synthetic" / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)
    db_path = out_dir / f"cld_{config.n_clones}clones_{args.scenario}.db"

    # Ensure fresh DB
    ensure_fresh_db(db_path)

    # Create DB and tables
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("PRAGMA foreign_keys = ON;") # Enforce FK constraints

    schema_sql = load_schema_sql(schema_path)
    cur.executescript(schema_sql)
    conn.commit()

    # --------------------------------------
    # Step 0 : Insert metadata tables (product, host_cell, vector, cell_line)
    # --------------------------------------
    pd.DataFrame([{
        "product_id": "P001",
        "modality": "mAb",
        "target_indication" : "PD-1 like oncology",
    }]).to_sql("product", conn, if_exists="append", index=False)

    pd.DataFrame([{
        "host_id": "CHO-K1",
        "species" : "Cricetulus griseus",
        "genetic_background" : "CHO-K1 reference",
        }]).to_sql("host_cell", conn, if_exists="append", index=False) 

    pd.DataFrame([{
        "vector_id": "V001",
        "backbone" : "pcDNA3.1",
        "promoter" : "CMV",
        "signal_peptide" : "IgG kappa",
        "copy_number_est": 1,
    }]).to_sql("vector", conn, if_exists="append", index=False) 

    pd.DataFrame([{
        "cell_line_id": "CL001",
        "host_id": "CHO-K1",
        "vector_id": "V001",
        "selection_marker" : "Neomycin",
        "product_id" : "P001",
        "transfection_date" : "2026-01-24",
        "transfection_method" : "Lipofectamine",
    }]).to_sql("cell_line", conn, if_exists="append", index=False)

    # --------------------------------------
    # Step 0b : Create batches (one assay batch per passage number)
    # --------------------------------------
    base_date = date(2026, 1, 24)
    batch_rows = []
    batch_effects = {}

    for p in range(1, config.n_passages + 1):
        batch_id = f"B_P{p:02d}"
        run_date = (base_date + timedelta(days=(p - 1) * 7)).isoformat()  # Weekly intervals

        batch_rows.append({
            "batch_id": batch_id,
            "experiment_type": "assay_run",
            "run_date": run_date,
            "platform": "ELISA | Vi-CELL | SEC-HPLC",
            "operator": "sim-operator_01"
        })

        # Hidden systematic offsets for each assay in this batch
        batch_effects[batch_id] = {
            "titer": np.random.normal(0, config.batch_titer_sd),
            "vcd": np.random.normal(0, config.batch_vcd_sd),
            "viability": np.random.normal(0, config.batch_viability_sd),
            "aggregation": np.random.normal(0, config.batch_aggregation_sd),
        }
    # Insert batches into DB
    pd.DataFrame(batch_rows).to_sql("batch", conn, if_exists="append", index=False)
    
    # Store batch effects truths for validation
    pd.DataFrame([{"batch_id": k, **v} for k, v in batch_effects.items()]).to_csv(
        out_dir / f"batch_effects_truths_{config.n_clones}_{args.scenario}.csv", index=False)
    
    # --------------------------------------
    # Step 1 : Generate clones + latent truth (Productivity, Stability, Quality)
    # NOTE: P/S/Q are NOT stored in the DB to avoid ML leakage.
    # --------------------------------------

    clone_ids = [f"CLONE_{i:04d}" for i in range(1, config.n_clones + 1)]

    #productivities = np.random.lognormal(mean = config.mu_logP, sigma= config.sigma_logP, size=config.n_clones)
    #stabilities = np.random.beta(a = config.alpha_stability, b = config.beta_stability, size=config.n_clones)
    #quality_potentials = np.clip(
    #    np.random.normal(loc = config.mu_quality_potential, scale = config.sigma_quality_potential, size=config.n_clones),
    #    0.0, 1.0)
    
    # ------------------------------
    # Latent truth generation (P/S/Q) + platform + copy number + clone-specific decay
    # Notes:
    # - We do NOT store these latents in DB tables (avoid ML leakage)
    # - We export them to CSV for validation and debugging
    # ------------------------------

    # Base latents
    P_base = np.random.lognormal(mean = config.mu_logP, sigma= config.sigma_logP, size=config.n_clones)
    S_base = np.random.beta(a = config.alpha_stability, b = config.beta_stability, size=config.n_clones)
    Q_base = np.clip(
        np.random.normal(loc = config.mu_quality_potential, scale = config.sigma_quality_potential, size=config.n_clones),
        0.0, 1.0)
    
    # Platform factor: scenario group (legacy vs optimized) + clone residual
    # - is_opt: 1 if clone is on optimized platform, else 0
    # - G_base: baseline platform quality for each clone based on scenario group
    if config.enable_platform:
        if config.enable_platform_groups:    
            is_opt = (np.random.rand(config.n_clones) < config.frac_optimized).astype(int)
            G_base = np.where(is_opt == 1, config.G_optimized, config.G_legacy)
        else:
            is_opt = np.zeros(config.n_clones, dtype=int)
            G_base = np.full(config.n_clones, config.G_project)
        
        g_clone = np.random.normal(0.0, config.g_clone_sd, size=config.n_clones)
        G_raw = np.clip(G_base + g_clone, -2.0, 2.0)  # prevent extreme values
        G = 1.0 / (1.0 + np.exp(-G_raw))  # sigmoid to bound between 0 and 1
    else:
        is_opt = np.zeros(config.n_clones, dtype=int)
        G = np.full(config.n_clones, 0.5) # no platform effect


    # Copy number (true + observed ddPCR)
    if config.enable_copy_number:
        CN_true = np.random.lognormal(mean = np.log(config.cn_mean), sigma = config.cn_sigma, size = config.n_clones)
        CN_true = np.clip(np.round(CN_true), 1, config.cn_max).astype(int)
        CN_obs = np.clip(np.round(CN_true + np.random.normal(0.0, config.ddpcr_noise_sd, size=config.n_clones)), 0, config.cn_max *2).astype(int)
    else:
        CN_true = np.ones(config.n_clones, dtype=int)
        CN_obs = CN_true.copy()

    # Apply platform/copy-number effects to P/S/Q
    # - CN increases productivity (dimisnihing returns)
    # - CN slightly penalizes stability and quality (expression burden)
    # CN effect can be stronger on optimized platform (better expression machinery / integration design)
    if config.enable_platform and config.enable_platform_groups:
        cn_mult = np.where(is_opt == 1, config.optimized_cn_effect_mult, 1.0)
    else:
        cn_mult = 1.0

    P = P_base * np.exp(config.platform_strength_P * (G - 0.5)) * np.exp((config.cn_effect_P * cn_mult) * np.log1p(CN_true))
    S = np.clip(S_base + config.platform_strength_S * (G - 0.5) - config.cn_penalty_S * np.log1p(CN_true), 0.0, 1.0)
    Q = np.clip(Q_base + config.platform_strength_Q * (G - 0.5) - config.cn_penalty_Q * np.log1p(CN_true), 0.0, 1.0)

    # Clone-specific decay sensitivity
    if config.enable_clone_decay_variation:
        k_decay_i = config.k_decay * np.exp(np.random.normal(0.0, config.k_decay_sd, size=config.n_clones))
    else:
        k_decay_i = np.full(config.n_clones, config.k_decay)

    # Optimized platform reduces silencing/decay speed (better epigenetic stability)
    if config.enable_platform and config.enable_platform_groups:
        k_decay_i = np.where(is_opt == 1, k_decay_i * config.platform_decay_mult, k_decay_i)

    productivities = P
    stabilities = S
    quality_potentials = Q

    latents = pd.DataFrame({
        "clone_id": clone_ids,
        "productivity": productivities,
        "stability": stabilities,
        "quality_potential": quality_potentials,
        "G_platform": G,
        "platform_group": is_opt, # 0 = legacy, 1 = optimized
        "CN_true": CN_true,
        "CN_obs": CN_obs,
        "k_decay_i": k_decay_i,
    })
    
    
    latents.to_csv(out_dir / f"clone_latent_truths_{config.n_clones}_{args.scenario}.csv", index=False)

    # Lookup maps for fast access inside passage loop
    cn_obs_by_clone = dict(zip(latents["clone_id"], latents["CN_obs"]))
    kdecay_by_clone = dict(zip(latents["clone_id"], latents["k_decay_i"]))
    g_by_clone = dict(zip(latents["clone_id"], latents["G_platform"]))

    pd.DataFrame({
        "clone_id": clone_ids,
        "cell_line_id": "CL001",
        "isolation_method": "limiting_dilution",
        "clone_rank": None
    }).to_sql("clone", conn, if_exists="append", index=False)

    # Assign culture mode per clone (simple scenario; optional realism)
    culture_mode_by_clone = {cid: ('fed-batch' if np.random.rand() < 0.85 else "perfusion")
                             for cid in clone_ids}
    
    # --------------------------------------
    # Step 2 / 3/ 4 : Generate passages, process conditions, and assay results
    # --------------------------------------

    # Data rows to insert
    passage_rows = []
    process_condition_rows = []
    assay_result_rows = []

    # For labels and clone ranking
    early_titer_sum = {cid: 0.0 for cid in clone_ids}
    early_titer_n = {cid: 0 for cid in clone_ids}

    # For stbility label using averages
    stab_early_sum = {cid: 0.0 for cid in clone_ids}
    stab_early_n = {cid: 0 for cid in clone_ids}
    stab_late_sum = {cid: 0.0 for cid in clone_ids}
    stab_late_n = {cid: 0 for cid in clone_ids}

    mean_P = float(np.mean(productivities))

    for _, row in latents.iterrows():
        cid = row["clone_id"]
        mode = culture_mode_by_clone[cid]

        for p in range(1, config.n_passages + 1):
            passage_id = f"{cid}_P{p:02d}"

            # Passage table row
            passage_rows.append({
                "passage_id": passage_id,
                "clone_id": cid,
                "passage_number": p,
                "culture_duration": 7,
                "phase" : phase_label(p),
            })

            # Process_Condition table row (per passage)
            temp = 37.0 + np.random.normal(0, 0.15)
            pH = 7.0 + np.random.normal(0, 0.05)
            feed_strategy = "standard" if mode == "fed-batch" else "perfusion"
            
            process_condition_rows.append({
                "condition_id" : f"COND_{passage_id}",
                "passage_id": passage_id,
                "culture_mode": mode,
                "temp": temp if mode == "fed-batch" else (temp - 0.5),
                "pH": pH,
                "feed_strategy": feed_strategy,
                "medium": "Chemically defined CHO medium"
            })

            # Effective productivity after stability-driven decay
            # This is the key CLD phenomenon
            # unstable clones (low S) lose expression across passages

            # Clone-specific decay sensitivity (k_decay_i) makes decay speed heterogenous across clones
            k_i = kdecay_by_clone[cid]
            P_ip = row["productivity"] * np.exp(-k_i * (1 - row["stability"]) * p)

            # Expression burden depends on current productivity (P_ip)
            expr_burden = (P_ip / mean_P)

            # A mild "environment stress" increases slightly with passage,
            # while expression stress depends on current burden
            stress = (config.stress_base + 
                      config.env_stress_slope * p +
                      config.expr_stress * expr_burden)
            
            # Platform can reduce stress slightly (better folding/secretion robustness)
            if config.enable_platform:
                factor = 1.0 - config.platform_stress_relief * (g_by_clone[cid] - 0.5)
                factor = np.clip(factor, 0.8, 1.2)  # prevent extreme relief
                stress *= factor

            
            if mode == "perfusion":
                stress *= 0.9  # Perfusion reduces stress slightly
            
            # Batch ID per passage
            batch_id = f"B_P{p:02d}"
            be = batch_effects[batch_id]

            # --------------------------------
            # Assays
            # --------------------------------

            # Titer depends on effective productivity + noise + batch effect
            titer_true = config.alpha_titer * P_ip
            titer = max(0.0, titer_true
                         + np.random.normal(0, config.titer_noise_sd) + be["titer"])
            
            # VCD: increases with passage (adaptation)
            # As P_ip decays, burden decreases, so VCD can recover later
            adaptation_factor = (1.0 + config.adapt_vcd * np.log1p(p))
            burden_factor = np.exp(-config.burden_coeff * expr_burden) # higher burden -> smaller VCD
            vcd_true = config.base_vcd * adaptation_factor * burden_factor
            vcd = max(0.0, vcd_true + np.random.normal(0, config.vcd_noise_sd) + be["vcd"])

            # Viability: tends to improve with passage adaptation
            # and improves when burden decreases (P_pi decays)
            viab_true = 95.0 + config.adapt_viab * np.log1p(p) - 2.0 * expr_burden
            viability = float(np.clip(viab_true + np.random.normal(0, config.viability_noise_sd) + be["viability"], 0.0, 100.0))

            # Quality proxy (aggregation): worse when intrinsic quality is low (1 - Q))
            agg_true = (config.gamma_agg_intrinsic * (1 - row["quality_potential"]) + config.delta_agg_stress * stress)
            aggregation = float(np.clip(agg_true + np.random.normal(0, config.aggregation_noise_sd)+ be["aggregation"], 0.0, 100.0))

            # Store assay result row
            assay_result_rows.extend([
                {"assay_id": f"ASSAY_{passage_id}_titer", "passage_id": passage_id, "batch_id": batch_id, "assay_type": "titer", "value": titer, "unit": "g/L", "method": "ELISA"},
                {"assay_id": f"ASSAY_{passage_id}_vcd", "passage_id": passage_id, "batch_id": batch_id, "assay_type": "vcd", "value": vcd, "unit": "cells/mL", "method": "Vi-CELL"},
                {"assay_id": f"ASSAY_{passage_id}_viability", "passage_id": passage_id, "batch_id": batch_id, "assay_type": "viability", "value": viability, "unit": "%", "method": "Vi-CELL"},
                {"assay_id": f"ASSAY_{passage_id}_aggregation", "passage_id": passage_id, "batch_id": batch_id, "assay_type": "aggregation", "value": aggregation, "unit": "%", "method": "SEC-HPLC"},
            ])

            # ddPCR copy number assay (store once per clone at the start of stability early window)
            if config.enable_copy_number and p == config.stability_early_start:
                assay_result_rows.append({
                    "assay_id": f"ASSAY_{passage_id}_ddpcr_cn",
                    "passage_id": passage_id,
                    "batch_id": batch_id,
                    "assay_type": "ddpcr_cn",
                    "value": float(cn_obs_by_clone[cid]),
                    "unit": "copies/cell",
                    "method": "ddPCR",
                })



            # Collect early passage titer stats for ranking
            if p <= config.early_max:
                early_titer_sum[cid] += titer
                early_titer_n[cid] += 1

            # Collect titer for stability label
            if config.stability_early_start <= p <= config.stability_early_end:
                stab_early_sum[cid] += titer
                stab_early_n[cid] += 1
            if config.stability_late_start <= p <= config.stability_late_end:
                stab_late_sum[cid] += titer
                stab_late_n[cid] += 1

        
    # Insert generated rows into SQLite
    pd.DataFrame(passage_rows).to_sql("passage", conn, if_exists="append", index=False)
    pd.DataFrame(process_condition_rows).to_sql("process_condition", conn, if_exists="append", index=False)
    pd.DataFrame(assay_result_rows).to_sql("assay_result", conn, if_exists="append", index=False)

    # --------------------------------------
    # Step 5 : Stability label table
    # --------------------------------------

    stability_rows = []
    for cid in clone_ids:
        n0 = stab_early_n[cid]
        nf = stab_late_n[cid]

        early_mean = (stab_early_sum[cid] / n0) if n0 > 0 else np.nan
        late_mean = (stab_late_sum[cid] / nf) if nf > 0 else np.nan

        if not np.isfinite(early_mean) or early_mean <= 1e-9 or not np.isfinite(late_mean):
            drop = np.nan
        else:
            drop = (early_mean - late_mean) / early_mean

        stability_rows.append({
            "stability_id": f"STB_{cid}",
            "clone_id": cid,
            "start_passage": config.stability_early_start,
            "end_passage": config.stability_late_end,
            "productivity_drop_pct": drop,
            "metric_type": "titer_drop",
            "evaluation_method": f"avg_titer_p{config.stability_early_start}-{config.stability_early_end}_vs_p{config.stability_late_start}-{config.stability_late_end}"
        })

    # Insert stability labels into DB
    pd.DataFrame(stability_rows).to_sql("stability_test", conn, if_exists="append", index=False)

    # --------------------------------------
    # Step 6 : Clone ranking based on early passage performance
    # --------------------------------------

    early_mean_rank = {
        cid: (early_titer_sum[cid] / max(1, early_titer_n[cid]))
        for cid in clone_ids
    }
    sorted_clones = sorted(clone_ids, key=lambda c: early_mean_rank[c], reverse=True)

    updates = [(rank + 1, cid) for rank, cid in enumerate(sorted_clones)]
    cur.executemany("UPDATE clone SET clone_rank = ? WHERE clone_id = ?", updates)
    # Commit and close
    conn.commit()
    conn.close()

    print(f"Generated DB at: {db_path}")
    print(f"- clones : {config.n_clones}, passages/clone : {config.n_passages}")
    print(f"- assay rows : {len(assay_result_rows)}")
    print(f"P quantiles:", np.quantile(latents["productivity"], [0.25, 0.5, 0.75]))
    print(f"S mean:", np.mean(latents["stability"]))
    print(f"Q mean:", np.mean(latents["quality_potential"]))

if __name__ == "__main__":
    main()

