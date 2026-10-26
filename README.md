# Reproducible microalgae PBR image analysis package

This package contains the cleaned input tables and Python scripts to reproduce the PCA, PLSR, Random Forest, SVR, feature-set comparison, and residual/subgroup error analyses.

## Input data

`data/microalgae_pbr_image_level_features.csv`

Main analysis table. One row represents one image. Color features are averaged from four 200 × 200 pixel ROI per image. This is the main dataset for PCA and prediction modeling.

`data/microalgae_pbr_roi_level_features.csv`

ROI-level table. This file is provided for transparency and within-image variability checking. It should not be treated as independent biological replication.

## Methodological decisions

1. Main model uses image-level observations, not ROI-level rows, to avoid pseudo-replication.
2. Only measured OD days are used: Day 0, 2, 4, 6, and 8.
3. OD665 and OD750 are not PCA inputs. They are used only for correlation or supervised prediction targets.
4. Day is not used as an input feature in prediction models.
5. ROI selection score is excluded from models because it is an algorithmic quality score, not a biological color predictor.
6. Model validation uses leave-one-batch-out validation.
7. PLSR is the primary model because the dataset is small and image-derived color features are collinear.
8. Random Forest and SVR are benchmark models.

## How to run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run all analyses:

```bash
python scripts/00_run_all.py
```

Run individually:

```bash
python scripts/01_pca_analysis.py
python scripts/02_plsr_main_model.py
python scripts/03_random_forest_benchmark.py
python scripts/04_svr_benchmark.py
python scripts/05_feature_set_comparison_plsr.py
python scripts/06_residual_and_subgroup_error_analysis.py
```

All outputs are saved in the `outputs/` folder.

## Important interpretation rule

ROI-level observations are technical subsamples. The defensible statistical unit for model evaluation is the image-level observation.

## Previous output reference

The folder `previous_outputs_reference/` contains the zipped outputs generated during the earlier analysis session. These are included so the numerical results used in the discussion can be cross-checked against the reproducible scripts.

The main PLSR script in `scripts/02_plsr_main_model.py` reproduces the reported main PLSR values:

- OD750: R² approximately 0.708, RMSE approximately 0.203, MAE approximately 0.163.
- OD665: R² approximately 0.703, RMSE approximately 0.241, MAE approximately 0.190.

The folder `scripts/original_generation_scripts/` contains original working scripts saved during the analysis session for SVR, green-feature comparison, PBR-type error analysis, and residual analysis.
