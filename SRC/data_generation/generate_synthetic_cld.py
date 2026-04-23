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
    n_clones : int = 5000
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
    # Legacy jackpot subpopulation (two-track)
    # ----------------------------------------
    # Track A: "super jackpot" (very rare, high P + high S + low decay) -> biosimilar-style strategy
    # Track B: "aggressive jackpot" (rare, high P but slightly unstable) -> nobel mAb/ ADC strategy

    enable_jackpot_cluster: bool = True
    
    # Super jackpot (very rare, high P + high S + low decay)
    super_frac: float = 0.015
    super_P_mult: float = 3.0
    super_S_add: float = 0.25
    super_decay_mult: float = 0.40

    # Aggressive jackpot (rare, high P but slightly unstable)
    aggressive_frac: float = 0.035
    aggressive_P_mult: float = 1.8
    aggressive_S_add: float = -0.05
    aggressive_decay_mult: float = 1.10

    # --------------------------------------
    # Phenotype coupling knobs (make early signals more learnable)
    # --------------------------------------

    # Super clone : stronger and cleaner phenotype
    super_early_titer_mult: float = 1.10
    super_late_titer_mult: float = 1.08
    super_burden_relief: float = 0.15 # super clones have less burden, so smaller VCD penalty and better viability
    super_extra_stress_relief: float = 0.08 # super clones have better folding/secretion, so less stress and better quality
    super_agg_shift: float = -0.15 # super clones have better quality, so lower aggregation

    # Aggressive clone : looks good early, then loses productivity stability (false positive)
    # Early: high growth / viability / titer look attractive
    # Late: productivity collapses and quality liability appears 
    aggressive_early_titer_mult: float = 1.12
    aggressive_early_vcd_mult: float = 1.25
    aggressive_early_viab_bonus: float = 3.0
    
    aggressive_early_fade: float = 0.015 # additional productivity decay per passage for aggressive clones
    aggressive_instability_start: int = 7 # passage at which aggressive clones start to become unstable
    aggressive_late_extra_decay: float = 0.06 # aggressive clones have faster decay starting mid-passage, so they lose more productivity late
    
    aggressive_burden_relief: float = 0.10 # aggressive clones have more burden relief as they lose productivity
    aggressive_extra_stress: float = 0.015 # aggressive clones have more expression stress, so slightly worse quality
    aggressive_agg_shift: float = 0.10 # aggressive clones have worse quality, so higher aggregation
    aggressive_agg_noise_mult: float = 1.50 # aggressive clones have more heterogeneous quality, so higher aggregation noise

    # Clone-to-clone variation within aggressive subgroup
    aggressive_early_vcd_sd: float = 0.10
    aggressive_early_viab_sd: float = 0.80
    aggressive_early_titer_sd: float = 0.05
    aggressive_late_decay_jitter_sd: float = 0.02

    # --------------------------------------
    # Optimized residual-failure knobs
    # --------------------------------------

    # Even in optimized platforms, a very small residual subgroup can remain:
    # clones that look fine early but lose performance later due to product/cassette-specific effects
    optimized_residual_aggressive_frac: float = 0.01

    # Mild early boost so they can look competitive in early screening
    optimized_residual_early_titer_mult: float = 1.05

    # Late-emerging instability starts later and is milder than legacy aggressive
    optimized_residual_instability_start: int = 9
    optimized_residual_late_extra_decay: float = 0.03

    # Aggregation impact is weak, mostly stochastic / hidden
    optimized_residual_agg_shift: float = 0.18
    optimized_residual_agg_noise_mult: float = 2.20

    # Hidden late-only clone factors:
    # Not directly visible in early features, used to reduce overly perfect prediction
    optimized_hidden_titer_sd: float = 0.18
    optimized_hidden_agg_sd: float = 1.20

    # Additional late-only stochastic process noise in optimized world
    optimized_late_titer_noise_sd: float = 0.24
    optimized_late_agg_noise_sd: float = 1.10

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

    # --------------------------------------
    # Legacy late aggregation realism knobs
    # --------------------------------------
    # Hidden late-only clone liability
    legacy_late_agg_hidden_sd: float = 2.6

    # Late process variability (batch/process-like instability not visable early)
    legacy_late_agg_process_sd: float = 1.6

    # Additional observation noise at late window
    legacy_late_agg_obs_noise: float = 1.1

    # Mid/late hidden contribution scaling
    legacy_mid_agg_hidden_mult: float = 0.20
    legacy_late_agg_hidden_mult: float = 0.95

    # Additional late decoupling from early features
    legacy_late_agg_extra_decouple_sd: float = 0.9

    # Aggressive-clone late quality worsening
    aggressive_late_agg_shift: float = 0.35
    aggressive_late_agg_noise_mult_late: float = 2.00

    # Quality proxy parameters
    gamma_agg_intrinsic: float = 18.0
    delta_agg_stress: float = 7.0    

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
               "enable_jackpot_cluster": True,
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
               "enable_jackpot_cluster": False,
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

    # ------------------------------
    # Subgroup assignment
    # - legacy: super + aggressive
    # - optimized: no super jackpot, but allow a very small residual aggressive subgroup
    # ------------------------------

    rng = np.random.default_rng(config.seed + 123)

    if args.scenario == "legacy" and config.enable_jackpot_cluster:

        # Disjoint assignment: sample super first, then aggressive from remaining pool
        is_super = (rng.random(config.n_clones) < config.super_frac)
        remaining = ~is_super
        is_aggr = (rng.random(config.n_clones) < config.aggressive_frac) & remaining

        # Apply super jackpot effects
        P = np.where(is_super, P * config.super_P_mult, P)
        S = np.clip(S + is_super.astype(float) * config.super_S_add, 0.0, 1.0)
        k_decay_i = np.where(is_super, k_decay_i * config.super_decay_mult, k_decay_i)

        # Apply aggressive jackpot effects
        P = np.where(is_aggr, P * config.aggressive_P_mult, P)
        S = np.clip(S + is_aggr.astype(float) * config.aggressive_S_add, 0.0, 1.0)
        k_decay_i = np.where(is_aggr, k_decay_i * config.aggressive_decay_mult, k_decay_i)
    
    elif args.scenario == "optimized":

        # No rare "super jackpot" in optimized:
        # the whole platform distribution is already shifted toward better/stable behavior
        is_super = np.zeros(config.n_clones, dtype=bool)

        # But allow a very small residual aggressive subgroup
        is_aggr = (rng.random(config.n_clones) < config.optimized_residual_aggressive_frac)

        # Make them only mildly attractive early and slightly less stable late
        P = np.where(is_aggr, P * 1.05, P)
        S = np.clip(S + is_aggr.astype(float) * (-0.02), 0.0, 1.0)
        k_decay_i = np.where(is_aggr, k_decay_i * 1.08, k_decay_i)

    else:
        is_super = np.zeros(config.n_clones, dtype=bool)
        is_aggr = np.zeros(config.n_clones, dtype=bool)

    # --------------------------------------
    # Clone-level variation within aggressive subgroup
    # --------------------------------------
    aggressive_vcd_boost_i = np.ones(config.n_clones)
    aggressive_viab_bonus_i = np.zeros(config.n_clones)
    aggressive_titer_mult_i = np.ones(config.n_clones)
    aggressive_late_decay_i = np.full(config.n_clones, config.aggressive_late_extra_decay)

    if args.scenario == "legacy":
        aggressive_vcd_boost_i = np.where(
            is_aggr,
            np.clip(
                np.random.normal(
                    config.aggressive_early_vcd_mult,
                    config.aggressive_early_vcd_sd,
                    size=config.n_clones
                ),
                1.00, 1.45
            ),
            1.0
        )

        aggressive_viab_bonus_i = np.where(
            is_aggr,
            np.clip(
                np.random.normal(
                    config.aggressive_early_viab_bonus,
                    config.aggressive_early_viab_sd,
                    size=config.n_clones
                ),
                0.5, 4.5
            ),
            0.0
        )

        aggressive_titer_mult_i = np.where(
            is_aggr,
            np.clip(
                np.random.normal(
                    config.aggressive_early_titer_mult,
                    config.aggressive_early_titer_sd,
                    size=config.n_clones
                ),
                1.00, 1.20
            ),
            1.0
        )

        aggressive_late_decay_i = np.where(
            is_aggr,
            np.clip(
                np.random.normal(
                    config.aggressive_late_extra_decay,
                    config.aggressive_late_decay_jitter_sd,
                    size=config.n_clones
                ),
                0.03, 0.10
            ),
            config.aggressive_late_extra_decay
        )

    # Optimized platform reduces silencing/decay speed (better epigenetic stability)
    # We can apply this reduction to all clones if enable_platform_groups is False (i.e., all clones on optimized platform), or only to the fraction of clones on optimized platform if grouping is enabled.
    if config.enable_platform and config.enable_platform_groups:
        k_decay_i = np.where(is_opt == 1, k_decay_i * config.optimized_decay_mult, k_decay_i)

    productivities = P
    stabilities = S
    quality_potentials = Q

    # Hidden late-only clone factors
    # These are not directly visible from early observed features
    # and make optimized late outcomes less perfectly predictabe.
    if args.scenario == "optimized":
        hidden_titer_late = np.random.normal(
            0.0, config.optimized_hidden_titer_sd, size = config.n_clones
        )
        hidden_agg_late = np.random.normal(
            0.0, config.optimized_hidden_agg_sd, size = config.n_clones
        )

        # Late-only clone-level stochastic process effects
        hidden_titer_process = np.random.normal(
            0.0, config.optimized_late_titer_noise_sd, size = config.n_clones
        )
        hidden_agg_process = np.random.normal(
            0.0, config.optimized_late_agg_noise_sd, size = config.n_clones
        )

    elif args.scenario == "legacy":
        hidden_titer_late = np.zeros(config.n_clones)
        hidden_agg_late = np.random.normal(
            0.0, config.legacy_late_agg_hidden_sd, size = config.n_clones
        )
        hidden_titer_process = np.zeros(config.n_clones)
        hidden_agg_process = np.random.normal(
            0.0, config.legacy_late_agg_process_sd, size = config.n_clones
        )
    else:
        hidden_titer_late = np.zeros(config.n_clones)
        hidden_agg_late = np.zeros(config.n_clones)
        hidden_titer_process = np.zeros(config.n_clones)
        hidden_agg_process = np.zeros(config.n_clones)

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
        "is_super": is_super.astype(int),
        "is_aggressive": is_aggr.astype(int),
        "hidden_titer_late": hidden_titer_late,
        "hidden_agg_late": hidden_agg_late,
        "hidden_titer_process": hidden_titer_process,
        "hidden_agg_process": hidden_agg_process,
        "aggressive_vcd_boost_i": aggressive_vcd_boost_i,
        "aggressive_viab_bonus_i": aggressive_viab_bonus_i,
        "aggressive_titer_mult_i": aggressive_titer_mult_i,
        "aggressive_late_decay_i": aggressive_late_decay_i,
    })
    
    
    latents.to_csv(out_dir / f"clone_latent_truths_{config.n_clones}_{args.scenario}.csv", index=False)

    # Lookup maps for fast access inside passage loop
    cn_obs_by_clone = dict(zip(latents["clone_id"], latents["CN_obs"]))
    kdecay_by_clone = dict(zip(latents["clone_id"], latents["k_decay_i"]))
    g_by_clone = dict(zip(latents["clone_id"], latents["G_platform"]))
    hidden_titer_by_clone = dict(zip(latents["clone_id"], latents["hidden_titer_late"]))
    hidden_agg_by_clone = dict(zip(latents["clone_id"], latents["hidden_agg_late"]))
    hidden_titer_process_by_clone = dict(zip(latents["clone_id"], latents["hidden_titer_process"]))
    hidden_agg_process_by_clone = dict(zip(latents["clone_id"], latents["hidden_agg_process"]))
    aggr_vcd_boost_by_clone = dict(zip(latents["clone_id"], latents["aggressive_vcd_boost_i"]))
    aggr_viab_bonus_by_clone = dict(zip(latents["clone_id"], latents["aggressive_viab_bonus_i"]))
    aggr_titer_mult_by_clone = dict(zip(latents["clone_id"], latents["aggressive_titer_mult_i"]))
    aggr_late_decay_by_clone = dict(zip(latents["clone_id"], latents["aggressive_late_decay_i"]))

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

            # ----------------------------------------------
            # Group-specific phenotype coupling
            # ----------------------------------------------

            is_super_clone = bool(row["is_super"])
            is_aggr_clone = bool(row["is_aggressive"])

            # Base productivity trajectory from latent productivity and stability
            k_i = kdecay_by_clone[cid]
            P_ip = row["productivity"] * np.exp(-k_i * (1 - row["stability"]) * p)

            # Default modifiers
            titer_mult = 1.0
            growth_burden_mult = 1.0
            extra_stress = 0.0
            agg_shift = 0.0
            agg_noise_sd = config.aggregation_noise_sd
            vcd_multiplier = 1.0
            extra_viab_bonus = 0.0

            # ---- Super clone : good early signal + sustained late performance ----
            if is_super_clone:
                if p <= config.early_max:
                    titer_mult *= config.super_early_titer_mult
                else:
                    titer_mult *= config.super_late_titer_mult

                # keep growth/viability reasonably normal despite high production
                growth_burden_mult *= (1.0 - config.super_burden_relief)

                # slightly cleaner/less stressed phenotype
                extra_stress -= config.super_extra_stress_relief
                agg_shift += config.super_agg_shift
            
            # ---- Aggressive clone : looks good early, then collapse productivity stability ----
            elif is_aggr_clone:
                if args.scenario == "legacy":
                    # Early phase: aggressive clones look excellent
                    # in titer, growth, and viability
                    if p <= config.early_max:
                        early_steps = max(0, p - config.stability_early_start)
                        fade = max(0.85, 1.0 - config.aggressive_early_fade * early_steps)
                        
                        titer_mult *= aggr_titer_mult_by_clone[cid] * fade
                        growth_burden_mult *= (1.0 - config.aggressive_burden_relief)
                        vcd_multiplier = aggr_vcd_boost_by_clone[cid]
                        extra_viab_bonus = aggr_viab_bonus_by_clone[cid]
                    else:
                        vcd_multiplier = 1.0
                        extra_viab_bonus = 0.0

                    # Late phase: hidden instability becomes visible
                    if p >= config.aggressive_instability_start:
                        late_steps = p - config.aggressive_instability_start + 1
                        P_ip *= np.exp(-aggr_late_decay_by_clone[cid] * late_steps)
                    
                    # Quality liability becomes more visible later than early
                    extra_stress += config.aggressive_extra_stress

                    # Mild early aggregating penalty
                    agg_shift += 0.5 * config.aggressive_agg_shift
                    agg_noise_sd *= config.aggressive_agg_noise_mult

                    # Additional late-emerging aggregation liability
                    if p >= 16:
                        agg_shift += 0.5 * config.aggressive_late_agg_shift
                        agg_noise_sd *= 1.0 + 0.5 * (config.aggressive_late_agg_noise_mult_late - 1.0)

                    if p >= config.late_min:
                        agg_shift += 0.5 * config.aggressive_late_agg_shift
                        agg_noise_sd *= 1.0 + 0.5 * (config.aggressive_late_agg_noise_mult_late - 1.0)
                
                elif args.scenario == "optimized":

                    # Residual aggressive clones in optimized world:
                    # mildly attractive early, but later quality/productivity liabilities appear
                    if p <= config.early_max:
                        titer_mult *= config.optimized_residual_early_titer_mult

                    if p >= config.optimized_residual_instability_start:
                        late_steps = p - config.optimized_residual_instability_start + 1
                        P_ip *= np.exp(-config.optimized_residual_late_extra_decay * late_steps)

                    # VCD/viability mostly unaffected
                    growth_burden_mult *= 0.97

                    # Aggregation penalty becomes more visible from mid-late passage

                    if p >= 16:
                        agg_shift += 0.5 * config.optimized_residual_agg_shift
                        agg_noise_sd *= 1.0 + 0.5 * (config.optimized_residual_agg_noise_mult - 1.0)

                    if p >= config.late_min:
                        agg_shift += 0.5 * config.optimized_residual_agg_shift
                        agg_noise_sd *= 1.0 + 0.5 * (config.optimized_residual_agg_noise_mult - 1.0)
            
            # Expression burden depends on current effective productivity
            expr_burden = (P_ip / mean_P)

            # Growth/viability use a "relieved" burden so aggressive clones 
            # do not look obviously worse in VCD / viability
            effective_growth_burden = expr_burden * growth_burden_mult

            # Base stress
            stress = (
                config.stress_base 
                + config.env_stress_slope * p
                + config.expr_stress * expr_burden
                + extra_stress
            )

            # Platform effect on stress
            if config.enable_platform:
                factor = 1.0 - config.platform_stress_relief * (g_by_clone[cid] - 0.5)
                factor = np.clip(factor, 0.8, 1.2)  # prevent extreme relief
                stress *= factor

            # Perfusion slightly reduces stress
            if mode == "perfusion":
                stress *= 0.9

            stress = max(0.0, stress) # prevent negative stress
            
            # Batch ID per passage
            batch_id = f"B_P{p:02d}"
            be = batch_effects[batch_id]

            # --------------------------------
            # Assays
            # --------------------------------

            # Titer depends on effective productivity + noise + batch effect
            late_titer_factor = 1.0
            late_titer_additive = 0.0

            # hidden titer effect is added from mid phase (psaage 16)
            if p >= 16:
                late_titer_factor = np.exp(0.5 * hidden_titer_by_clone[cid])

            if p >= config.late_min:
                late_titer_factor *= np.exp(0.5 * hidden_titer_by_clone[cid])
                late_titer_additive += hidden_titer_process_by_clone[cid]
            
            titer_true = (
                config.alpha_titer * P_ip * titer_mult * late_titer_factor
                + late_titer_additive)
            titer = max(0.0, titer_true
                         + np.random.normal(0, config.titer_noise_sd) + be["titer"])
            
            # VCD: adaptation improves over passage, but growth burden is only mildly affected
            adaptation_factor = (1.0 + config.adapt_vcd * np.log1p(p))
            burden_factor = np.exp(-config.burden_coeff * effective_growth_burden) # higher burden -> smaller VCD
            vcd_true = config.base_vcd * adaptation_factor * burden_factor * vcd_multiplier
            vcd = max(0.0, vcd_true + np.random.normal(0, config.vcd_noise_sd) + be["vcd"])

            # Viability: largenly maintained, only weakly burden-sensitive
            viab_true = 95.0 + config.adapt_viab * np.log1p(p) - 2.0 * effective_growth_burden + extra_viab_bonus
            viability = float(np.clip(viab_true + np.random.normal(0, config.viability_noise_sd) + be["viability"], 0.0, 100.0))

            # Aggregation: weak clone effect + stress + noise
            late_agg_shift = 0.0
            extra_agg_noise_sd = 0.0

            # Mid-late hidden quality liability begins earlier than the late label window
            # hidden agg liability accumulate from passage 16
            if args.scenario == "legacy":
                if p >= 16:
                    late_agg_shift += config.legacy_mid_agg_hidden_mult * hidden_agg_by_clone[cid]
                    extra_agg_noise_sd += 0.5 * abs(hidden_agg_process_by_clone[cid])

                if p >= config.late_min:
                    late_agg_shift += config.legacy_late_agg_hidden_mult * hidden_agg_by_clone[cid]
                    extra_agg_noise_sd += abs(hidden_agg_process_by_clone[cid])

            else:
                if p >= 16:
                    late_agg_shift += 0.4 * hidden_agg_by_clone[cid]
                    extra_agg_noise_sd += 0.4 * abs(hidden_agg_process_by_clone[cid])

                if p >= config.late_min:
                    late_agg_shift += hidden_agg_by_clone[cid]
                    extra_agg_noise_sd += abs(hidden_agg_process_by_clone[cid])
            
            late_hidden_decouple = 0.0

            if args.scenario == "legacy":
                # Extra late-only hidden decoupling: makes late aggregation harder to infer from early data
                if p >= config.late_min:
                    late_hidden_decouple = np.random.normal(
                        0.0, config.legacy_late_agg_extra_decouple_sd
                    )
            agg_true = (
                config.gamma_agg_intrinsic * (1 - row["quality_potential"])
                + config.delta_agg_stress * stress
                + agg_shift
                + late_agg_shift
                + late_hidden_decouple
            )

            late_obs_noise = 0.0
            if args.scenario == "optimized" and p >= config.late_min:
                late_obs_noise = 0.35
            elif args.scenario == "legacy" and p >= config.late_min:
                late_obs_noise = config.legacy_late_agg_obs_noise
            
            aggregation = float(
                np.clip(
                    agg_true
                    + np.random.normal(0, agg_noise_sd + extra_agg_noise_sd + late_obs_noise)
                    + be["aggregation"], 0.0, 100.0
                )
            )

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

