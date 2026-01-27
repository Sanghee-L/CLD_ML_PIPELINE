# Synthetic CLD Dataset Generator

This module generates a **synthetic (virtual) Cell Line Development (CLD)** dataset and stores it in **SQLite** following the ERD / schema in this repository.

The purpose is to create a realistic CLD dataset that can be used to develop and demonstrate:
- ML models for **productivity, stability, and quality**
- early clone filtering (“early clone drop”)
- decision-support pipelines that can later evolve into a Self-Driving Lab (SDL)

---

## What this script creates

### Outputs
After running the generator, you will get:

- **SQLite DB**
  - `data/synthetic/raw/cld.db`

- **Hidden truth (for validation only)**
  - `data/synthetic/raw/clone_latents_truth.csv`
    - contains latent variables `P` (productivity), `S` (stability), `Q` (quality potential)
  - `data/synthetic/raw/batch_effects_truth.csv`
    - contains simulated batch offsets for each passage-run

**Important:** `P/S/Q` are intentionally NOT stored in the SQLite tables to avoid ML “data leakage”.
In real labs, true latent values are not directly observable.

---

## Database tables populated

The generator populates the following tables (see `data/schema/cld_schema.sql`):

- `product`, `host_cell`, `vector`, `cell_line` (metadata)
- `batch` (assay runs: date/platform/operator)
- `clone` (clone IDs, ranks)
- `passage` (passage timeline and phase: early/mid/late)
- `process_condition` (culture mode, temp, pH, feed strategy, medium)
- `assay_result` (titer, VCD, viability, aggregation proxy)
- `stability_test` (stability label computed from early vs late average titers)

---

## Biological assumptions (high level)

This simulator encodes typical CLD phenomena:

1) **Productivity is right-skewed**
- Many low producers, few high producers (log-normal distribution)

2) **Productivity decays with passage depending on stability**
- Unstable clones lose expression faster across passages

3) **Growth/viability can recover late**
- As expression burden decreases (due to silencing/decay), cells become fitter

4) **Quality proxy (aggregation) worsens with intrinsic defects and stress**
- Intrinsic quality potential + stress contribute to aggregation risk

5) **Batch effects exist**
- Each passage-run has a small systematic offset (operator/day/platform drift)

---

## How stability is labeled

Stability is computed as:

- Early mean titer = average titer over passages **1–5**
- Late mean titer = average titer over passages **26–30**
- Drop (%):

`drop = (early_mean - late_mean) / early_mean`

This reduces sensitivity to single-passage noise.

---

## How clone ranking is computed

Clone rank is assigned based on **early mean titer** (passages 1–5):
- Highest early mean titer → rank 1
- Next → rank 2
- etc.

---

## How to run

From the repository root:

```bash
python src/data_generation/generate_synthetic_cld.py
