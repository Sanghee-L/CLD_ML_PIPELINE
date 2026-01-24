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
    aggregation_noise_sd : float = 0.03  # fraction

    # Batch effect SDs (systemic offsets per passage-run)
    batch_titer_sd : float = 0.05  # g/L
    batch_vcd_sd : float = 0.3e6   # cells/mL
    batch_viability_sd : float = 0.4  # fraction
    batch_aggregation_sd : float = 0.02  # fraction

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
        "host_cell_id": "CHO-K1",
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
        run_date = base_date + timedelta(days=(p - 1) * 7).isoformat()  # Weekly intervals

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

    pd.DataFrame(batch_rows).to_sql("batch", conn, if_exists="append", index=False)
    pd.DataFrame([{"batch_id": k, **v} for k, v in batch_effects.items()]).to_csv(
        out_dir / "batch_effects_truth.csv", index=False)
    

    # --------------------------------------
    # Step 1 : Generate clones + latent truth (Productivity, Stability, Quality)
    # --------------------------------------
    