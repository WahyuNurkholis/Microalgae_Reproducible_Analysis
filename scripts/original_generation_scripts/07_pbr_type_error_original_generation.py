import os, zipfile, textwrap
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import ttest_rel, wilcoxon, shapiro

# Input prediction file from validated main PLSR analysis
input_pred = Path('/mnt/data/PLSR_environment_error_analysis/PLSR_predictions_with_environment_errors.csv')
if not input_pred.exists():
    input_pred = Path('/mnt/data/tmp_plsr_old/PLSR_fixed3_LOBO_predictions.csv')

outdir = Path('/mnt/data/PLSR_PBR_type_error_analysis')
outdir.mkdir(parents=True, exist_ok=True)

pred = pd.read_csv(input_pred)
# Standardize column names/derived errors
pred['PBR_Label'] = pred['PBR_Type'].replace({'Flat':'FPPBR', 'Tube':'TPBR', 'Tubular':'TPBR', 'Flat panel':'FPPBR'})
pred['Error'] = pred['Predicted'] - pred['Observed']
pred['Residual'] = pred['Observed'] - pred['Predicted']
pred['Abs_Error'] = pred['Error'].abs()
pred['Squared_Error'] = pred['Error']**2
pred['APE_percent'] = np.where(pred['Observed'].abs()>1e-12, pred['Abs_Error']/pred['Observed'].abs()*100, np.nan)

# Helper metrics
rows=[]
for target in sorted(pred['Target'].unique()):
    for pbr, g in pred[pred['Target']==target].groupby('PBR_Label'):
        obs = g['Observed'].to_numpy()
        prd = g['Predicted'].to_numpy()
        mae = mean_absolute_error(obs, prd)
        rmse = np.sqrt(mean_squared_error(obs, prd))
        r2 = r2_score(obs, prd) if len(g)>1 and np.var(obs)>0 else np.nan
        mbe = np.mean(prd-obs)
        rows.append({
            'Target': target,
            'PBR_Type': pbr,
            'n': len(g),
            'Observed_mean': np.mean(obs),
            'Observed_sd': np.std(obs, ddof=1),
            'Observed_min': np.min(obs),
            'Observed_max': np.max(obs),
            'Predicted_mean': np.mean(prd),
            'R2_within_PBR_type': r2,
            'RMSE': rmse,
            'MAE': mae,
            'MBE_pred_minus_obs': mbe,
            'MAPE_percent': np.nanmean(g['APE_percent']),
            'NRMSE_by_mean_percent': rmse/(np.mean(obs)+1e-12)*100,
            'NMAE_by_mean_percent': mae/(np.mean(obs)+1e-12)*100,
            'Mean_Abs_Error': g['Abs_Error'].mean(),
            'Median_Abs_Error': g['Abs_Error'].median()
        })
metrics = pd.DataFrame(rows)
metrics.to_csv(outdir/'PLSR_per_PBR_type_error_metrics.csv', index=False)

# More granular metrics by PBR x environment (may help discussion)
rows=[]
for target in sorted(pred['Target'].unique()):
    for (pbr, env), g in pred[pred['Target']==target].groupby(['PBR_Label','Environment']):
        obs = g['Observed'].to_numpy()
        prd = g['Predicted'].to_numpy()
        mae = mean_absolute_error(obs, prd)
        rmse = np.sqrt(mean_squared_error(obs, prd))
        r2 = r2_score(obs, prd) if len(g)>1 and np.var(obs)>0 else np.nan
        rows.append({
            'Target': target,
            'PBR_Type': pbr,
            'Environment': env,
            'n': len(g),
            'Observed_mean': np.mean(obs),
            'Observed_sd': np.std(obs, ddof=1),
            'R2_within_subgroup': r2,
            'RMSE': rmse,
            'MAE': mae,
            'NRMSE_by_mean_percent': rmse/(np.mean(obs)+1e-12)*100,
            'NMAE_by_mean_percent': mae/(np.mean(obs)+1e-12)*100,
            'MBE_pred_minus_obs': np.mean(prd-obs)
        })
metrics_env = pd.DataFrame(rows)
metrics_env.to_csv(outdir/'PLSR_per_PBR_type_by_environment_metrics.csv', index=False)

# Paired absolute error tests: same batch, environment, day, target, compare FPPBR vs TPBR
paired_rows=[]
detail_rows=[]
for target in sorted(pred['Target'].unique()):
    tmp = pred[pred['Target']==target].copy()
    pivot = tmp.pivot_table(index=['Batch','Environment','Day'], columns='PBR_Label', values='Abs_Error', aggfunc='mean').reset_index()
    if set(['FPPBR','TPBR']).issubset(pivot.columns):
        paired = pivot.dropna(subset=['FPPBR','TPBR']).copy()
        paired['Abs_Error_Diff_FPPBR_minus_TPBR'] = paired['FPPBR'] - paired['TPBR']
        for _, r in paired.iterrows():
            detail_rows.append({'Target': target, **r.to_dict()})
        d = paired['Abs_Error_Diff_FPPBR_minus_TPBR'].to_numpy()
        t_stat, t_p = ttest_rel(paired['FPPBR'], paired['TPBR']) if len(d)>1 else (np.nan,np.nan)
        try:
            w_stat, w_p = wilcoxon(paired['FPPBR'], paired['TPBR'], zero_method='wilcox') if np.any(d!=0) and len(d)>0 else (np.nan,np.nan)
        except ValueError:
            w_stat, w_p = np.nan, np.nan
        try:
            sh_stat, sh_p = shapiro(d) if 3 <= len(d) <= 5000 else (np.nan,np.nan)
        except Exception:
            sh_stat, sh_p = np.nan, np.nan
        paired_rows.append({
            'Target': target,
            'n_pairs': len(paired),
            'FPPBR_mean_abs_error': paired['FPPBR'].mean(),
            'TPBR_mean_abs_error': paired['TPBR'].mean(),
            'Mean_abs_error_diff_FPPBR_minus_TPBR': d.mean(),
            'Median_abs_error_diff_FPPBR_minus_TPBR': np.median(d),
            'Paired_t_stat': t_stat,
            'Paired_t_p': t_p,
            'Wilcoxon_stat': w_stat,
            'Wilcoxon_p': w_p,
            'Shapiro_diff_stat': sh_stat,
            'Shapiro_diff_p': sh_p
        })
paired_tests = pd.DataFrame(paired_rows)
paired_details = pd.DataFrame(detail_rows)
paired_tests.to_csv(outdir/'PLSR_per_PBR_type_paired_abs_error_tests.csv', index=False)
paired_details.to_csv(outdir/'PLSR_per_PBR_type_paired_abs_error_details.csv', index=False)

# Save full predictions with PBR errors
pred.to_csv(outdir/'PLSR_predictions_with_PBR_type_errors.csv', index=False)

# Plot defaults (no fixed colors)
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 13,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.dpi': 300,
    'savefig.dpi': 600,
})

# Scatter true vs predicted by PBR type for each target
for target in sorted(pred['Target'].unique()):
    gtarget = pred[pred['Target']==target]
    fig, ax = plt.subplots(figsize=(7,6))
    for pbr, g in gtarget.groupby('PBR_Label'):
        ax.scatter(g['Observed'], g['Predicted'], s=70, alpha=0.85, edgecolor='black', linewidth=0.6, label=pbr)
    mn = min(gtarget['Observed'].min(), gtarget['Predicted'].min())
    mx = max(gtarget['Observed'].max(), gtarget['Predicted'].max())
    pad = (mx-mn)*0.05 if mx>mn else 0.1
    ax.plot([mn-pad, mx+pad], [mn-pad, mx+pad], linestyle='--', linewidth=2, label='Ideal fit')
    ax.set_xlabel(f'Observed {target}')
    ax.set_ylabel(f'Predicted {target}')
    ax.set_title(f'PLSR prediction by PBR type: {target}')
    ax.legend(frameon=False)
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(outdir/f'PLSR_true_vs_predicted_by_PBR_type_{target}.png', bbox_inches='tight')
    fig.savefig(outdir/f'PLSR_true_vs_predicted_by_PBR_type_{target}.svg', bbox_inches='tight')
    plt.close(fig)

# Boxplot absolute error by PBR type
for target in sorted(pred['Target'].unique()):
    gtarget = pred[pred['Target']==target]
    order = ['FPPBR','TPBR']
    data = [gtarget[gtarget['PBR_Label']==pbr]['Abs_Error'].values for pbr in order]
    fig, ax = plt.subplots(figsize=(6.5,5.5))
    ax.boxplot(data, labels=order, showmeans=True)
    # add jittered points
    rng = np.random.default_rng(42)
    for i, vals in enumerate(data, start=1):
        x = rng.normal(i, 0.04, size=len(vals))
        ax.scatter(x, vals, s=35, alpha=0.75, edgecolor='black', linewidth=0.4)
    ax.set_ylabel(f'Absolute error of {target}')
    ax.set_xlabel('PBR type')
    ax.set_title(f'Absolute prediction error by PBR type: {target}')
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(outdir/f'PLSR_absolute_error_by_PBR_type_{target}.png', bbox_inches='tight')
    fig.savefig(outdir/f'PLSR_absolute_error_by_PBR_type_{target}.svg', bbox_inches='tight')
    plt.close(fig)

# Residual boxplot by PBR type
for target in sorted(pred['Target'].unique()):
    gtarget = pred[pred['Target']==target]
    order = ['FPPBR','TPBR']
    data = [gtarget[gtarget['PBR_Label']==pbr]['Residual'].values for pbr in order]
    fig, ax = plt.subplots(figsize=(6.5,5.5))
    ax.axhline(0, linestyle='--', linewidth=1.5)
    ax.boxplot(data, labels=order, showmeans=True)
    rng = np.random.default_rng(42)
    for i, vals in enumerate(data, start=1):
        x = rng.normal(i, 0.04, size=len(vals))
        ax.scatter(x, vals, s=35, alpha=0.75, edgecolor='black', linewidth=0.4)
    ax.set_ylabel(f'Residual, observed minus predicted {target}')
    ax.set_xlabel('PBR type')
    ax.set_title(f'Residuals by PBR type: {target}')
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(outdir/f'PLSR_residuals_by_PBR_type_{target}.png', bbox_inches='tight')
    fig.savefig(outdir/f'PLSR_residuals_by_PBR_type_{target}.svg', bbox_inches='tight')
    plt.close(fig)

# Summary bar plots for metrics
for metric in ['MAE','RMSE','R2_within_PBR_type','NRMSE_by_mean_percent','NMAE_by_mean_percent']:
    fig, ax = plt.subplots(figsize=(7,5.5))
    plotdf = metrics.copy()
    targets = sorted(plotdf['Target'].unique())
    pbrs = ['FPPBR','TPBR']
    x = np.arange(len(targets))
    width = 0.35
    for j, pbr in enumerate(pbrs):
        vals=[]
        for t in targets:
            subset = plotdf[(plotdf['Target']==t)&(plotdf['PBR_Type']==pbr)]
            vals.append(subset[metric].iloc[0] if len(subset)>0 else np.nan)
        ax.bar(x + (j-0.5)*width, vals, width, label=pbr)
    ax.set_xticks(x)
    ax.set_xticklabels(targets)
    ax.set_ylabel(metric.replace('_',' '))
    ax.set_xlabel('Target')
    ax.set_title(f'PLSR {metric.replace("_", " ")} by PBR type')
    ax.legend(frameon=False)
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(outdir/f'PLSR_{metric}_by_PBR_type_summary.png', bbox_inches='tight')
    fig.savefig(outdir/f'PLSR_{metric}_by_PBR_type_summary.svg', bbox_inches='tight')
    plt.close(fig)

# Write readme
readme = f"""
PLSR per-PBR-type error analysis
================================
Input predictions: {input_pred}
Model context: PLSR fixed 3 components, leave-one-batch-out validation, image-level average of 4 ROI.
PBR labels: Flat -> FPPBR; Tube -> TPBR.
Targets: {', '.join(sorted(pred['Target'].unique()))}

Output files:
- PLSR_per_PBR_type_error_metrics.csv: Main metrics by PBR type and target.
- PLSR_per_PBR_type_by_environment_metrics.csv: Metrics by PBR type, environment, and target.
- PLSR_per_PBR_type_paired_abs_error_tests.csv: Paired tests comparing absolute error between FPPBR and TPBR at matched Batch x Environment x Day.
- PLSR_per_PBR_type_paired_abs_error_details.csv: Matched pair error details.
- PLSR_predictions_with_PBR_type_errors.csv: All predictions with residual and error columns.
- PNG/SVG figures: true-vs-predicted plots, absolute error boxplots, residual plots, and metric summary plots.

Interpretation guide:
Use absolute error and normalized error together. A smaller absolute error does not always mean a more robust model if the observed OD range is narrower. R2 within subgroup may become negative when the model does not explain within-subgroup variation better than the subgroup mean.
"""
(outdir/'analysis_readme.txt').write_text(readme)

# Zip outputs
zip_path = Path('/mnt/data/PLSR_PBR_type_error_analysis.zip')
if zip_path.exists():
    zip_path.unlink()
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
    for p in outdir.iterdir():
        zf.write(p, arcname=f'{outdir.name}/{p.name}')

print('Saved:', zip_path)
print('Metrics:')
print(metrics.to_string(index=False))
print('\nPaired tests:')
print(paired_tests.to_string(index=False))
print('\nMetrics by PBR x Environment:')
print(metrics_env.to_string(index=False))
