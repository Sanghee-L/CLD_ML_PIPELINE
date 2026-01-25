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
    n_clones : int = 500
    n_passages : int = 30

    # Phase labeling (to prevent ML leakage)
    early_max : int = 7
    late_min : int = 25

    # Stability label window
    p0 : int = 1
    pf : int = 30

    # Stability: Beta distribution -> bound [0, 1]
    alpha_stability : float = 7.0
    beta_stability : float = 3.0

    # Quality potential: near high, clipped to [0, 1]
    mu_quality_potential : float = 0.8
    sigma_quality_potential : float = 0.15

    # Productivity decay sensitivity
    k_decay : float = 0.03

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


#-----------------------------------------
# Main synthetic data generation function
#-----------------------------------------

def main() -> None:
    np.random.seed(config.seed)

    # Repo paths
    root = Path(__file__).resolve().parents[2]
    schema_path = root / "data" / "schema" / "cld_schema.sql"
    out_dir = root / "data" / "synthetic" / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)
    db_path = out_dir / "cld.db"

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
        "host_cell_id": "CHO-K1",
        "vector_id": "V001",
        "selection_marker" : "Neomycin",
        "product_id" : "P001",
        "transfection_date" : datetime(2026, 1, 24),
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
            "titer": np.random.lognormal(0, config.batch_titer_sd),
            "vcd": np.random.normal(0, config.batch_vcd_sd),
            "viability": np.random.normal(0, config.batch_viability_sd),
            "aggregation": np.random.normal(0, config.batch_aggregation_sd),
        }
    # Insert batches into DB
    pd.DataFrame(batch_rows).to_sql("batch", conn, if_exists="append", index=False)
    
    # Store batch effects truths for validation
    pd.DataFrame([{"batch_id": k, **v} for k, v in batch_effects.items()]).to_csv(
        out_dir / "batch_effects_truths.csv", index=False)
    

    # --------------------------------------
    # Step 1 : Generate clones + latent truth (Productivity, Stability, Quality)
    # NOTE: P/S/Q are NOT stored in the DB to avoid ML leakage.
    # --------------------------------------

    clone_ids = [f"CLONE_{i:04d}" for i in range(1, config.n_clones + 1)]

    productivities = np.random.lognormal(mean = config.mu_logP, sigma= config.sigma_logP, size=config.n_clones)
    stabilities = np.random.beta(a = config.alpha_stability, b = config.beta_stability, size=config.n_clones)
    quality_potentials = np.clip(
        np.random.normal(loc = config.mu_quality_potential, scale = config.sigma_quality_potential, size=config.n_clones),
        0.0, 1.0)
    
    latents = pd.DataFrame({
        "clone_id": clone_ids,
        "productivity": productivities,
        "stability": stabilities,
        "quality_potential": quality_potentials,
    })
    latents.to_csv(out_dir / "clone_latent_truths.csv", index=False)

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
    titer_at_p0 = {}
    titer_at_pf = {}

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
                "medium": "Chemically defined CHO medium"
            })

            # Effective productivity after stability-driven decay
            # This is the key CLD phenomenon
            # unstable clones (low S) lose expression across passages
            P_ip = row["productivity"] * np.exp(-config.k_decay * (1 - row["stability"]) * p)

            # Expression burden depends on current productivity (P_ip)
            expr_burden = (P_ip / mean_P)

            # A mild "environment stress" increases slightly with passage,
            # while expression stress depends on current burden
            stress = (config.stress_base +
                      config.env_stress_slope * p +
                      config.expr_stress * expr_burden)
            
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
            agg_true = (config.gamma_agg * (1 - row["quality_potential"]) + config.delta_agg * stress)
            aggregation = float(np.clip(agg_true + np.random.normal(0, config.aggregation_noise_sd)+ be["aggregation"], 0.0, 100.0))

            # Store assay result row
            assay_result_rows.append([
                {"assay_id": f"ASSAY_{passage_id}_titer", "passage_id": passage_id, "batch_id": batch_id, "assay_type": "titer", "value": titer, "units": "g/L", "method": "ELISA"},
                {"assay_id": f"ASSAY_{passage_id}_vcd", "passage_id": passage_id, "batch_id": batch_id, "assay_type": "vcd", "value": vcd, "units": "cells/mL", "method": "Vi-CELL"},
                {"assay_id": f"ASSAY_{passage_id}_viability", "passage_id": passage_id, "batch_id": batch_id, "assay_type": "viability", "value": viability, "units": "%", "method": "Vi-CELL"},
                {"assay_id": f"ASSAY_{passage_id}_aggregation", "passage_id": passage_id, "batch_id": batch_id, "assay_type": "aggregation", "value": aggregation, "units": "%", "method": "SEC-HPLC"},
            ])

            # Collect early passage titer stats for ranking
            if p <= config.early_max:
                early_titer_sum[cid] += titer
                early_titer_n[cid] += 1

            # Collect titer for stability label
            if p == config.p0:
                titer_at_p0[cid] = titer
            if p == config.pf:
                titer_at_pf[cid] = titer
        
    # Insert generated rows into SQLite
    pd.DataFrame(passage_rows).to_sql("passage", conn, if_exists="append", index=False)
    pd.DataFrame(process_condition_rows).to_sql("process_condition", conn, if_exists="append", index=False)
    pd.DataFrame(assay_result_rows).to_sql("assay_result", conn, if_exists="append", index=False)

    # --------------------------------------
    # Step 5 : Stability label table
    # --------------------------------------

    stability_rows = []
    for cid in clone_ids:
        t0 = float(titer_at_p0.get(cid, np.nan))
        tf = float(titer_at_pf.get(cid, np.nan))

        if not np.isfinite(t0) or t0 <= 1e-9 or not np.isfinite(tf):
            drop = np.nan
        else:
            drop = (t0 - tf) / t0

        stability_rows.append({
            "stability_id": f"STB_{cid}",
            "clone_id": cid,
            "start_passage": config.p0,
            "end_passage": config.pf,
            "productivity_drop_pct": drop,
            "metric_type": "titer_drop",
            "evaluation_method": "simulated_passage_decay"
        })

    # Insert stability labels into DB
    pd.DataFrame(stability_rows).to_sql("stability_test", conn, if_exists="append", index=False)

    # --------------------------------------
    # Step 6 : Clone ranking based on early passage performance
    # --------------------------------------

    early_mean = {cid: (early_titer_sum[cid] / max(1, early_titer_n[cid])) for cid in clone_ids}
    sorted_clones = sorted(clone_ids, key=lambda c: early_mean[c], reverse=True)

    updates_rows = [(rank + 1, cid) for rank, cid in enumerate(sorted_clones)]
    cur.executemany("UPDATE clone SET clone_rank = ? WHERE clone_id = ?", updates_rows)

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

