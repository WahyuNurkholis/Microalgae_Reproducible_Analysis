Random Forest analysis for microalgae PBR image-based OD estimation

Data used:
- 60 image-level observations from microalgae_pbr_image_level_features.csv.
- 4 ROI per image already averaged into image-level features.
- ROI-level observations were not treated as independent samples.
- Day was excluded from predictors.
- ROI_Selection_Score was excluded because it is an arbitrary QC score.
- Validation: leave-one-batch-out cross-validation with Batch as the group.

Random Forest setting:
- n_estimators = 300
- max_depth = 3
- min_samples_leaf = 3
- max_features = sqrt
- bootstrap = True
- random_state = 42

Feature sets:
- Mean4ROI_only: mean values from four ROIs.
- Mean_and_SD4ROI: mean values plus between-ROI standard deviation features.

Main files:
- RF_LOBO_overall_metrics.csv
- RF_LOBO_metrics_by_batch.csv
- RF_LOBO_predictions.csv
- RF_feature_importance_summary.csv
- RF_vs_PLSR_summary_for_discussion.csv
- RF_true_vs_predicted_OD750.png/svg
- RF_true_vs_predicted_OD665.png/svg
- RF_top_features_OD750.png/svg
- RF_top_features_OD665.png/svg
