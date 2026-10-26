import os, re, json, math, warnings, zipfile, pathlib
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import LeaveOneGroupOut, GroupKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.inspection import permutation_importance

infile = '/mnt/data/microalgae_pbr_image_level_features.csv'
out_dir = '/mnt/data/SVR_microalgae_analysis_results'
os.makedirs(out_dir, exist_ok=True)
df = pd.read_csv(infile)

def parse_meta(fn):
    stem=os.path.splitext(str(fn))[0]
    parts=stem.split('_')
    batch=parts[0] if len(parts)>0 else None
    pbr=parts[1] if len(parts)>1 else None
    env=parts[2] if len(parts)>2 else None
    m=re.search(r'Day\s*([0-9]+)', stem, flags=re.I)
    day=int(m.group(1)) if m else np.nan
    return pd.Series({'Batch':batch,'PBR_Type':pbr,'Environment':env,'Day':day})
if 'Batch' not in df.columns:
    meta = df['Image'].apply(parse_meta)
    for c in meta.columns:
        df[c]=meta[c]

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
feature_cols_all = [c for c in numeric_cols if (c.endswith('mean4ROI') or c.endswith('sd4ROI')) and 'ROI_Selection_Score' not in c]
feature_cols_mean = [c for c in feature_cols_all if c.endswith('mean4ROI')]
feature_sets = {
    'Mean_only': feature_cols_mean,
    'Mean_plus_SD': feature_cols_all,
}

targets = ['OD750','OD665']
groups = df['Batch'].values

def metrics(y_true, y_pred):
    return {'R2': r2_score(y_true, y_pred),
            'RMSE': float(np.sqrt(mean_squared_error(y_true, y_pred))),
            'MAE': float(mean_absolute_error(y_true, y_pred))}

# Compact grid for nested LOBO. Dataset is small, so broader tuning is not defensible.
param_grid = {
    'svr__kernel': ['rbf'],
    'svr__C': [1, 10, 100],
    'svr__gamma': ['scale', 0.01, 0.1],
    'svr__epsilon': [0.01, 0.1]
}

summary_rows=[]
fold_rows=[]
pred_rows=[]

for feat_name, feat_cols in feature_sets.items():
    X = df[feat_cols].astype(float).values
    logo = LeaveOneGroupOut()
    for target in targets:
        y = df[target].astype(float).values
        preds = np.full(y.shape, np.nan)
        params_by_fold=[]
        for train_idx, test_idx in logo.split(X, y, groups=groups):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            g_train = groups[train_idx]
            test_batch = sorted(np.unique(groups[test_idx]))[0]
            pipe = Pipeline([('scaler', StandardScaler()), ('svr', SVR())])
            inner_cv = GroupKFold(n_splits=len(np.unique(g_train)))
            gs = GridSearchCV(pipe, param_grid, scoring='neg_mean_absolute_error', cv=inner_cv, n_jobs=1, refit=True)
            gs.fit(X_train, y_train, groups=g_train)
            y_pred = gs.predict(X_test)
            preds[test_idx] = y_pred
            fold_m = metrics(y_test, y_pred)
            fold_rows.append({'Model':'SVR_nested_LOBO','Feature_Set':feat_name,'Target':target,'Test_Batch':test_batch,
                              'n_train':len(train_idx),'n_test':len(test_idx), **fold_m, 'Best_Params':json.dumps(gs.best_params_)})
            params_by_fold.append(gs.best_params_)
            for ii, yp in zip(test_idx, y_pred):
                pred_rows.append({'Model':'SVR_nested_LOBO','Feature_Set':feat_name,'Target':target,
                                  'Image':df.loc[ii,'Image'],'Batch':df.loc[ii,'Batch'],'PBR_Type':df.loc[ii,'PBR_Type'],
                                  'Environment':df.loc[ii,'Environment'],'Day':df.loc[ii,'Day'],
                                  'y_true':y[ii], 'y_pred':float(yp), 'Residual':float(y[ii]-yp)})
        all_m = metrics(y, preds)
        summary_rows.append({'Model':'SVR_nested_LOBO','Feature_Set':feat_name,'Target':target,
                             'n_obs':len(y),'n_features':len(feat_cols), **all_m,
                             'Best_Params_by_fold':json.dumps(params_by_fold)})

summary = pd.DataFrame(summary_rows).sort_values(['Target','R2'], ascending=[True,False])
fold_df = pd.DataFrame(fold_rows)
pred_df = pd.DataFrame(pred_rows)
summary.to_csv(os.path.join(out_dir,'SVR_nested_LOBO_summary.csv'), index=False)
fold_df.to_csv(os.path.join(out_dir,'SVR_nested_LOBO_fold_metrics.csv'), index=False)
pred_df.to_csv(os.path.join(out_dir,'SVR_nested_LOBO_predictions.csv'), index=False)

# Choose best feature set per target by R2
best_table = summary.sort_values(['Target','R2'], ascending=[True,False]).groupby('Target').head(1).reset_index(drop=True)
best_table.to_csv(os.path.join(out_dir,'SVR_best_model_by_target.csv'), index=False)

# Make plots for best feature set per target
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'axes.labelsize':14,'axes.titlesize':14,'xtick.labelsize':11,'ytick.labelsize':11,'legend.fontsize':10})
for _, row in best_table.iterrows():
    target = row['Target']; feat_name=row['Feature_Set']
    pp = pred_df[(pred_df['Target']==target)&(pred_df['Feature_Set']==feat_name)].copy()
    y_true=pp['y_true'].values; y_pred=pp['y_pred'].values
    m=metrics(y_true,y_pred)
    fig, ax = plt.subplots(figsize=(6.5,5.5))
    # Plot by batch
    for batch in sorted(pp['Batch'].unique()):
        tmp=pp[pp['Batch']==batch]
        ax.scatter(tmp['y_true'], tmp['y_pred'], s=60, edgecolor='black', linewidth=0.6, alpha=0.85, label=batch)
    mn = min(y_true.min(), y_pred.min()); mx = max(y_true.max(), y_pred.max())
    pad=(mx-mn)*0.08 if mx>mn else 0.1
    ax.plot([mn-pad,mx+pad],[mn-pad,mx+pad], linestyle='--', linewidth=1.8, label='Ideal fit')
    ax.set_xlabel(f'Measured {target}')
    ax.set_ylabel(f'Predicted {target}')
    ax.set_title(f'SVR prediction of {target}\nR² = {m["R2"]:.3f}, RMSE = {m["RMSE"]:.3f}, MAE = {m["MAE"]:.3f}')
    ax.legend(frameon=False)
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f'SVR_true_vs_predicted_{target}.png'), dpi=600, bbox_inches='tight')
    fig.savefig(os.path.join(out_dir, f'SVR_true_vs_predicted_{target}.svg'), bbox_inches='tight')
    plt.close(fig)

    # residual plot
    fig, ax = plt.subplots(figsize=(6.5,5.2))
    for batch in sorted(pp['Batch'].unique()):
        tmp=pp[pp['Batch']==batch]
        ax.scatter(tmp['y_pred'], tmp['Residual'], s=60, edgecolor='black', linewidth=0.6, alpha=0.85, label=batch)
    ax.axhline(0, linestyle='--', linewidth=1.5)
    ax.set_xlabel(f'Predicted {target}')
    ax.set_ylabel('Residual, measured - predicted')
    ax.set_title(f'SVR residual plot for {target}')
    ax.legend(frameon=False)
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f'SVR_residual_{target}.png'), dpi=600, bbox_inches='tight')
    fig.savefig(os.path.join(out_dir, f'SVR_residual_{target}.svg'), bbox_inches='tight')
    plt.close(fig)

# Final SVR fitted on full data for permutation importance with group-CV best params by global GroupKFold.
# This is supplementary interpretability only, not the validation metric.
importance_rows=[]
for _, row in best_table.iterrows():
    target=row['Target']; feat_name=row['Feature_Set']; feat_cols=feature_sets[feat_name]
    X=df[feat_cols].astype(float).values; y=df[target].astype(float).values
    pipe=Pipeline([('scaler',StandardScaler()),('svr',SVR())])
    cv=GroupKFold(n_splits=len(np.unique(groups)))
    gs=GridSearchCV(pipe,param_grid,scoring='neg_mean_absolute_error',cv=cv,n_jobs=1,refit=True)
    gs.fit(X,y,groups=groups)
    result=permutation_importance(gs.best_estimator_, X, y, n_repeats=20, random_state=42, scoring='neg_mean_absolute_error')
    imp=pd.DataFrame({'Target':target,'Feature_Set':feat_name,'Feature':feat_cols,
                      'Importance_MAE_increase':result.importances_mean,
                      'Importance_SD':result.importances_std})
    imp=imp.sort_values('Importance_MAE_increase', ascending=False)
    imp.to_csv(os.path.join(out_dir,f'SVR_permutation_importance_{target}.csv'), index=False)
    importance_rows.append(imp.assign(Rank=np.arange(1,len(imp)+1)))
    top=imp.head(15).iloc[::-1]
    fig,ax=plt.subplots(figsize=(8,6))
    ax.barh(top['Feature'], top['Importance_MAE_increase'])
    ax.set_xlabel('Permutation importance, MAE increase')
    ax.set_ylabel('Image-derived feature')
    ax.set_title(f'Top SVR permutation importance for {target}')
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir,f'SVR_permutation_importance_top15_{target}.png'), dpi=600, bbox_inches='tight')
    fig.savefig(os.path.join(out_dir,f'SVR_permutation_importance_top15_{target}.svg'), bbox_inches='tight')
    plt.close(fig)

if importance_rows:
    pd.concat(importance_rows, ignore_index=True).to_csv(os.path.join(out_dir,'SVR_permutation_importance_all_targets.csv'), index=False)

# Include readme
readme = f"""
SVR microalgae image-based OD analysis
======================================
Input file: {infile}
Observations: {df.shape[0]} image-level samples.
Validation: nested leave-one-batch-out cross-validation.
Groups: Batch parsed from image filename ({', '.join(sorted(df['Batch'].unique()))}).
Targets: OD750 and OD665.
Main input features: image-level features averaged from 4 ROI per image, plus optional 4-ROI SD features.
Excluded from input: Day, OD values, Specific GR, image size, ROI size, ROI_Selection_Score.

Important interpretation:
SVR is used as a non-linear benchmark model. It should be compared against PLSR, which remains more interpretable for small datasets with collinear color features.
Permutation importance is supplementary only and was calculated on the final fitted model, not as the main validation result.
"""
with open(os.path.join(out_dir,'analysis_readme.txt'),'w') as f:
    f.write(readme)

# zip results
zip_path = '/mnt/data/SVR_microalgae_analysis_results.zip'
if os.path.exists(zip_path): os.remove(zip_path)
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as z:
    for root, dirs, files in os.walk(out_dir):
        for file in files:
            p=os.path.join(root,file)
            z.write(p, arcname=os.path.relpath(p, out_dir))

print(summary.to_string(index=False))
print('\nBEST')
print(best_table.to_string(index=False))
print('\nZIP', zip_path)
