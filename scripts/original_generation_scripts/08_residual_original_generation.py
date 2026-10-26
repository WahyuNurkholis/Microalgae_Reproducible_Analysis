import os, math, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import kruskal, mannwhitneyu, spearmanr

# Input: PLSR fixed-3 leave-one-batch-out predictions
input_pred = Path('/mnt/data/tmp_plsr_old/PLSR_fixed3_LOBO_predictions.csv')
outdir = Path('/mnt/data/PLSR_residual_analysis')
outdir.mkdir(parents=True, exist_ok=True)

# Load data
pred = pd.read_csv(input_pred)
# Positive residual = Observed - Predicted = model underprediction
pred['Residual'] = pred['Observed'] - pred['Predicted']
pred['Abs_Error'] = pred['Residual'].abs()
pred['Squared_Error'] = pred['Residual'] ** 2
pred['PBR_Label'] = pred['PBR_Type'].map({'Flat':'FPPBR','Tube':'TPBR'}).fillna(pred['PBR_Type'])
pred['APE_percent'] = np.where(pred['Observed'].abs() > 1e-12, pred['Abs_Error'] / pred['Observed'].abs() * 100, np.nan)

# Helper metrics
def metrics(g):
    obs = g['Observed'].values
    pr = g['Predicted'].values
    res = g['Residual'].values
    n = len(g)
    out = {
        'n': n,
        'Observed_mean': float(np.mean(obs)),
        'Observed_min': float(np.min(obs)),
        'Observed_max': float(np.max(obs)),
        'Predicted_mean': float(np.mean(pr)),
        'Bias_mean_residual': float(np.mean(res)),
        'Residual_SD': float(np.std(res, ddof=1)) if n > 1 else np.nan,
        'MAE': float(np.mean(np.abs(res))),
        'RMSE': float(math.sqrt(np.mean(res**2))),
        'Max_Abs_Error': float(np.max(np.abs(res))),
        'NRMSE_by_mean_percent': float(math.sqrt(np.mean(res**2)) / np.mean(obs) * 100) if np.mean(obs) != 0 else np.nan,
        'NMAE_by_mean_percent': float(np.mean(np.abs(res)) / np.mean(obs) * 100) if np.mean(obs) != 0 else np.nan,
        'MAPE_percent': float(np.nanmean(np.abs(res) / np.where(np.abs(obs) > 1e-12, np.abs(obs), np.nan) * 100))
    }
    # R2 only if enough variance
    if n >= 2 and np.var(obs, ddof=1) > 0:
        out['R2'] = float(r2_score(obs, pr))
    else:
        out['R2'] = np.nan
    # Spearman residual vs predicted, to detect trend in residuals
    if n >= 4:
        r, p = spearmanr(g['Predicted'], g['Residual'])
        out['Spearman_residual_vs_predicted_r'] = float(r)
        out['Spearman_residual_vs_predicted_p'] = float(p)
    else:
        out['Spearman_residual_vs_predicted_r'] = np.nan
        out['Spearman_residual_vs_predicted_p'] = np.nan
    return pd.Series(out)

# Overall residual diagnostics
summary_overall = pred.groupby('Target', as_index=False).apply(metrics, include_groups=False).reset_index(drop=True)
summary_overall.to_csv(outdir/'PLSR_residual_overall_metrics.csv', index=False)

# Group summaries
for group_col, name in [('Batch','batch'),('Environment','environment'),('PBR_Label','PBR_type')]:
    sm = pred.groupby(['Target', group_col]).apply(metrics, include_groups=False).reset_index()
    sm.to_csv(outdir/f'PLSR_residual_metrics_by_{name}.csv', index=False)

# Multi-factor summary useful for limitation section
multi = pred.groupby(['Target','Batch','Environment','PBR_Label']).apply(metrics, include_groups=False).reset_index()
multi.to_csv(outdir/'PLSR_residual_metrics_by_batch_environment_PBR.csv', index=False)

# Statistical checks on absolute error distribution
rows = []
for target, dft in pred.groupby('Target'):
    # Batch: Kruskal-Wallis
    groups = [g['Abs_Error'].values for _, g in dft.groupby('Batch')]
    try:
        stat, p = kruskal(*groups)
    except Exception:
        stat, p = np.nan, np.nan
    rows.append({'Target': target, 'Grouping': 'Batch', 'Test': 'Kruskal-Wallis', 'Statistic': stat, 'p_value': p})
    # Environment: Mann-Whitney U
    groups = [g['Abs_Error'].values for _, g in dft.groupby('Environment')]
    labels = list(dft.groupby('Environment').groups.keys())
    if len(groups)==2:
        stat, p = mannwhitneyu(groups[0], groups[1], alternative='two-sided')
        rows.append({'Target': target, 'Grouping': 'Environment', 'Test': f'Mann-Whitney U ({labels[0]} vs {labels[1]})', 'Statistic': stat, 'p_value': p})
    # PBR type
    groups = [g['Abs_Error'].values for _, g in dft.groupby('PBR_Label')]
    labels = list(dft.groupby('PBR_Label').groups.keys())
    if len(groups)==2:
        stat, p = mannwhitneyu(groups[0], groups[1], alternative='two-sided')
        rows.append({'Target': target, 'Grouping': 'PBR type', 'Test': f'Mann-Whitney U ({labels[0]} vs {labels[1]})', 'Statistic': stat, 'p_value': p})

tests = pd.DataFrame(rows)
tests.to_csv(outdir/'PLSR_abs_error_group_tests.csv', index=False)

# Save full predictions with errors
pred.to_csv(outdir/'PLSR_predictions_with_residuals.csv', index=False)

# Plot settings
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 13,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 600
})

# Residual vs predicted plot per target, with environment labels
for target, dft in pred.groupby('Target'):
    fig, ax = plt.subplots(figsize=(7.2,5.4))
    for env, g in dft.groupby('Environment'):
        ax.scatter(g['Predicted'], g['Residual'], s=65, edgecolor='black', linewidth=0.6, alpha=0.85, label=env)
    ax.axhline(0, linestyle='--', linewidth=1.5)
    ax.set_xlabel(f'Predicted {target}')
    ax.set_ylabel('Residual (Observed - Predicted)')
    ax.set_title(f'Residual vs Predicted {target}')
    ax.legend(frameon=False)
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(outdir/f'PLSR_residual_vs_predicted_{target}.png', bbox_inches='tight')
    fig.savefig(outdir/f'PLSR_residual_vs_predicted_{target}.svg', bbox_inches='tight')
    plt.close(fig)

# Residual vs predicted colored by batch too
for target, dft in pred.groupby('Target'):
    fig, ax = plt.subplots(figsize=(7.2,5.4))
    for batch, g in dft.groupby('Batch'):
        ax.scatter(g['Predicted'], g['Residual'], s=65, edgecolor='black', linewidth=0.6, alpha=0.85, label=batch)
    ax.axhline(0, linestyle='--', linewidth=1.5)
    ax.set_xlabel(f'Predicted {target}')
    ax.set_ylabel('Residual (Observed - Predicted)')
    ax.set_title(f'Residual vs Predicted {target} by Batch')
    ax.legend(frameon=False)
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(outdir/f'PLSR_residual_vs_predicted_by_batch_{target}.png', bbox_inches='tight')
    fig.savefig(outdir/f'PLSR_residual_vs_predicted_by_batch_{target}.svg', bbox_inches='tight')
    plt.close(fig)

# Function for box+scatter absolute error by group
def abs_error_plot(target, group_col, group_label, fname):
    dft = pred[pred['Target']==target]
    order = sorted(dft[group_col].dropna().unique())
    data = [dft.loc[dft[group_col]==o, 'Abs_Error'].values for o in order]
    fig, ax = plt.subplots(figsize=(7.2,5.4))
    ax.boxplot(data, labels=order, showfliers=False)
    rng = np.random.default_rng(42)
    for i, vals in enumerate(data, start=1):
        x = rng.normal(i, 0.045, size=len(vals))
        ax.scatter(x, vals, s=45, edgecolor='black', linewidth=0.5, alpha=0.75)
    ax.set_xlabel(group_label)
    ax.set_ylabel(f'Absolute error of {target}')
    ax.set_title(f'Absolute Error by {group_label} for {target}')
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(outdir/f'{fname}_{target}.png', bbox_inches='tight')
    fig.savefig(outdir/f'{fname}_{target}.svg', bbox_inches='tight')
    plt.close(fig)

for target in pred['Target'].unique():
    abs_error_plot(target, 'Batch', 'Batch', 'PLSR_absolute_error_by_batch')
    abs_error_plot(target, 'Environment', 'Environment', 'PLSR_absolute_error_by_environment')
    abs_error_plot(target, 'PBR_Label', 'PBR type', 'PLSR_absolute_error_by_PBR_type')

# Residual bias by group bar-like point summary with CI based on SD/sqrt(n)
def bias_summary_plot(target, group_col, group_label, fname):
    dft = pred[pred['Target']==target]
    sm = dft.groupby(group_col)['Residual'].agg(['mean','std','count']).reset_index()
    sm['se'] = sm['std'] / np.sqrt(sm['count'])
    fig, ax = plt.subplots(figsize=(7.2,5.4))
    x = np.arange(len(sm))
    ax.errorbar(x, sm['mean'], yerr=sm['se'], fmt='o', capsize=4, markersize=7)
    ax.axhline(0, linestyle='--', linewidth=1.3)
    ax.set_xticks(x)
    ax.set_xticklabels(sm[group_col].astype(str).tolist())
    ax.set_xlabel(group_label)
    ax.set_ylabel('Mean residual (Observed - Predicted)')
    ax.set_title(f'Residual Bias by {group_label} for {target}')
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(outdir/f'{fname}_{target}.png', bbox_inches='tight')
    fig.savefig(outdir/f'{fname}_{target}.svg', bbox_inches='tight')
    plt.close(fig)

for target in pred['Target'].unique():
    bias_summary_plot(target, 'Batch', 'Batch', 'PLSR_residual_bias_by_batch')
    bias_summary_plot(target, 'Environment', 'Environment', 'PLSR_residual_bias_by_environment')
    bias_summary_plot(target, 'PBR_Label', 'PBR type', 'PLSR_residual_bias_by_PBR_type')

# Create a plain text readme with major numbers
with open(outdir/'analysis_readme.txt','w') as f:
    f.write('PLSR Residual Analysis\n')
    f.write('Input: PLSR_fixed3_LOBO_predictions.csv\n')
    f.write('Validation: leave-one-batch-out. Residual = Observed - Predicted.\n')
    f.write('Positive residual indicates model underprediction. Negative residual indicates model overprediction.\n\n')
    f.write('Overall metrics:\n')
    f.write(summary_overall.to_string(index=False))
    f.write('\n\nAbsolute error group tests:\n')
    f.write(tests.to_string(index=False))

# Zip outputs
zip_path = Path('/mnt/data/PLSR_residual_analysis.zip')
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as z:
    for file in outdir.rglob('*'):
        z.write(file, arcname=str(file.relative_to(outdir.parent)))

print('Output folder:', outdir)
print('Zip:', zip_path)
print('\nOverall metrics')
print(summary_overall)
print('\nBy batch')
print(pd.read_csv(outdir/'PLSR_residual_metrics_by_batch.csv'))
print('\nBy environment')
print(pd.read_csv(outdir/'PLSR_residual_metrics_by_environment.csv'))
print('\nBy PBR')
print(pd.read_csv(outdir/'PLSR_residual_metrics_by_PBR_type.csv'))
print('\nTests')
print(tests)
