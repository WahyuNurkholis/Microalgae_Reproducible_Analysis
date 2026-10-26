import os, re, json, zipfile, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

INPUT='/mnt/data/microalgae_pbr_image_level_features.csv'
OUT='/mnt/data/PLSR_green_variability_results'
os.makedirs(OUT, exist_ok=True)

df=pd.read_csv(INPUT)
# Parse batch, PBR, environment, day from Image
def parse_batch(s):
    m=re.match(r'(B\d+)_', str(s))
    return m.group(1) if m else None
def parse_pbr(s):
    s=str(s).lower()
    if '_flat_' in s: return 'Flat'
    if '_tube_' in s: return 'Tube'
    return None
def parse_env(s):
    s=str(s).lower()
    if '_in_' in s: return 'Indoor'
    if '_out_' in s: return 'Outdoor'
    return None

df['Batch']=df['Image'].apply(parse_batch)
df['PBR_Type']=df['Image'].apply(parse_pbr)
df['Environment']=df['Image'].apply(parse_env)
if 'Day' not in df.columns:
    df['Day']=df.get('Detected_Day')

# sanitize column naming
for target in ['OD665','OD750']:
    df[target]=pd.to_numeric(df[target], errors='coerce')

# Feature sets
all_numeric = df.select_dtypes(include=[np.number]).columns.tolist()
exclude = set(['Detected_Day','Day','Image_Width','Image_Height','ROI_Size_px','OD665','OD750','Specific GR'])
# Exclude ROI selection score to avoid using algorithm confidence as predictor
exclude.update([c for c in all_numeric if 'ROI_Selection_Score' in c])
full_features = [c for c in all_numeric if c not in exclude]

# Green-only curated mean feature set
raw_green_only = [
    'G_mean_mean4ROI','g_norm_mean4ROI','G_R_ratio_mean4ROI','G_B_ratio_mean4ROI',
    'ExG_raw_mean4ROI','ExG_norm_mean4ROI','NGRDI_mean4ROI','VARI_mean4ROI','GLI_mean4ROI',
    'Green_fraction_mean4ROI','H_mean_deg_mean4ROI','S_mean_mean4ROI','a_star_mean_mean4ROI','b_star_mean_mean4ROI'
]
green_only = [c for c in raw_green_only if c in df.columns]

# Green + variability, compact and interpretable
raw_green_variability = green_only + [
    # pixel-level variability inside each ROI, averaged across 4 ROIs
    'G_std_mean4ROI','V_std_mean4ROI','H_std_deg_mean4ROI','S_std_mean4ROI',
    'L_star_std_mean4ROI','a_star_std_mean4ROI','b_star_std_mean4ROI',
    # spatial variability among 4 ROIs for green and optical/color-gradient features
    'G_mean_sd4ROI','g_norm_sd4ROI','G_R_ratio_sd4ROI','G_B_ratio_sd4ROI',
    'ExG_raw_sd4ROI','ExG_norm_sd4ROI','NGRDI_sd4ROI','VARI_sd4ROI','GLI_sd4ROI',
    'Green_fraction_sd4ROI','H_mean_deg_sd4ROI','S_mean_sd4ROI','V_mean_sd4ROI',
    'L_star_mean_sd4ROI','a_star_mean_sd4ROI','b_star_mean_sd4ROI'
]
green_variability = []
for c in raw_green_variability:
    if c in df.columns and c not in green_variability:
        green_variability.append(c)

# Mean-only color features without SD, for comparison
mean_only = [c for c in full_features if c.endswith('_mean4ROI') and not c.endswith('_std_mean4ROI')]
# Need include std features? This is color mean only; no pixel sd, no ROI sd
mean_only = [c for c in mean_only if '_std_' not in c]

feature_sets = {
    'Full_features': full_features,
    'Color_mean_only': mean_only,
    'Green_only': green_only,
    'Green_plus_variability': green_variability,
}

# remove constant or nonfinite columns per subset
def prepare_data(df, features, target):
    cols = [c for c in features if c in df.columns]
    temp = df[['Image','Batch','Environment','PBR_Type','Day',target] + cols].copy()
    temp = temp.dropna(subset=[target,'Batch']+cols).reset_index(drop=True)
    # remove nonfinite
    for c in cols:
        temp[c] = pd.to_numeric(temp[c], errors='coerce')
    temp = temp.replace([np.inf,-np.inf], np.nan).dropna(subset=cols+[target]).reset_index(drop=True)
    # remove near-zero variance
    valid=[]
    for c in cols:
        if temp[c].nunique(dropna=True)>1 and temp[c].std(skipna=True)>1e-12:
            valid.append(c)
    return temp, valid

def metrics(y, pred):
    return {
        'R2': r2_score(y, pred),
        'RMSE': float(np.sqrt(mean_squared_error(y, pred))),
        'MAE': float(mean_absolute_error(y, pred))
    }

def lo_bo_fixed(df_in, features, target, n_components=3):
    temp, cols = prepare_data(df_in, features, target)
    groups=temp['Batch'].values
    unique=np.unique(groups)
    preds=[]; rows=[]; fold_metrics=[]
    for test_batch in unique:
        train_idx=np.where(groups!=test_batch)[0]
        test_idx=np.where(groups==test_batch)[0]
        X_train=temp.loc[train_idx, cols].values; y_train=temp.loc[train_idx, target].values
        X_test=temp.loc[test_idx, cols].values; y_test=temp.loc[test_idx, target].values
        ncomp=min(n_components, X_train.shape[0]-1, X_train.shape[1])
        pipe=Pipeline([('scaler', StandardScaler()), ('pls', PLSRegression(n_components=ncomp, scale=False))])
        pipe.fit(X_train,y_train)
        pred=pipe.predict(X_test).ravel()
        for idx, p in zip(test_idx, pred):
            row=temp.loc[idx, ['Image','Batch','Environment','PBR_Type','Day',target]].to_dict()
            row.update({'Target':target,'Predicted':float(p),'Observed':float(temp.loc[idx,target]),'Model':'PLSR_fixed3','n_components':ncomp})
            rows.append(row)
        fm=metrics(y_test,pred); fm.update({'Target':target,'Test_Batch':test_batch,'n_components':ncomp,'n_test':len(y_test)})
        fold_metrics.append(fm)
    pred_df=pd.DataFrame(rows)
    overall=metrics(pred_df['Observed'], pred_df['Predicted'])
    overall.update({'Target':target,'Model':'PLSR_fixed3','n_components':'fixed_3','n_features':len(cols),'n_obs':len(temp)})
    return overall, pd.DataFrame(fold_metrics), pred_df, cols

def lo_bo_tuned(df_in, features, target, max_comp=10):
    temp, cols = prepare_data(df_in, features, target)
    groups=temp['Batch'].values
    unique=np.unique(groups)
    rows=[]; fold_metrics=[]; tuning_rows=[]
    for test_batch in unique:
        train_idx=np.where(groups!=test_batch)[0]
        test_idx=np.where(groups==test_batch)[0]
        train_groups=groups[train_idx]
        X_train_all=temp.loc[train_idx, cols].values; y_train_all=temp.loc[train_idx, target].values
        X_test=temp.loc[test_idx, cols].values; y_test=temp.loc[test_idx, target].values
        max_ncomp=min(max_comp, X_train_all.shape[0]-1, X_train_all.shape[1])
        # inner CV over 2 training batches
        candidate=list(range(1,max_ncomp+1))
        inner_results=[]
        for ncomp in candidate:
            inner_preds=[]; inner_obs=[]
            for inner_train_rel, inner_val_rel in GroupKFold(n_splits=len(np.unique(train_groups))).split(X_train_all,y_train_all,train_groups):
                pipe=Pipeline([('scaler', StandardScaler()), ('pls', PLSRegression(n_components=ncomp, scale=False))])
                pipe.fit(X_train_all[inner_train_rel], y_train_all[inner_train_rel])
                p=pipe.predict(X_train_all[inner_val_rel]).ravel()
                inner_preds.extend(p); inner_obs.extend(y_train_all[inner_val_rel])
            rmse=float(np.sqrt(mean_squared_error(inner_obs, inner_preds)))
            inner_results.append({'Target':target,'Outer_Test_Batch':test_batch,'n_components':ncomp,'inner_RMSE':rmse,'inner_R2':r2_score(inner_obs, inner_preds)})
        best=min(inner_results, key=lambda d: d['inner_RMSE'])
        best_n=best['n_components']
        tuning_rows.extend(inner_results)
        pipe=Pipeline([('scaler', StandardScaler()), ('pls', PLSRegression(n_components=best_n, scale=False))])
        pipe.fit(X_train_all,y_train_all)
        pred=pipe.predict(X_test).ravel()
        for idx, p in zip(test_idx, pred):
            row=temp.loc[idx, ['Image','Batch','Environment','PBR_Type','Day',target]].to_dict()
            row.update({'Target':target,'Predicted':float(p),'Observed':float(temp.loc[idx,target]),'Model':'PLSR_tuned','n_components':best_n})
            rows.append(row)
        fm=metrics(y_test,pred); fm.update({'Target':target,'Test_Batch':test_batch,'n_components':best_n,'n_test':len(y_test),'inner_best_RMSE':best['inner_RMSE']})
        fold_metrics.append(fm)
    pred_df=pd.DataFrame(rows)
    overall=metrics(pred_df['Observed'], pred_df['Predicted'])
    comps=','.join(str(x) for x in pd.DataFrame(fold_metrics)['n_components'].tolist())
    overall.update({'Target':target,'Model':'PLSR_tuned','n_components':comps,'n_features':len(cols),'n_obs':len(temp)})
    return overall, pd.DataFrame(fold_metrics), pred_df, cols, pd.DataFrame(tuning_rows)

# VIP function based on fitted pipeline
def calc_vip(pls, X_scaled, y):
    # X_scaled used for final model, y centered implicitly by PLSRegression
    T = pls.x_scores_
    W = pls.x_weights_
    Q = pls.y_loadings_.reshape(-1)
    p, h = W.shape
    # Sum of squares explained in y by each component
    ss = np.sum(T**2, axis=0) * (Q**2)
    total_ss = np.sum(ss)
    if total_ss == 0:
        return np.full(p, np.nan)
    vip = np.sqrt(p * (W**2 @ ss) / total_ss)
    return vip

summary=[]; all_fold_metrics=[]; all_preds=[]; all_tuning=[]; selected_cols_report=[]
# Run fixed3 for all sets + tuned for Green+variability
for fs_name, feats in feature_sets.items():
    for target in ['OD750','OD665']:
        try:
            overall, fmetrics, pred_df, cols = lo_bo_fixed(df, feats, target, n_components=3)
            overall['Feature_Set']=fs_name
            summary.append(overall)
            fmetrics['Feature_Set']=fs_name; fmetrics['Model']='PLSR_fixed3'; all_fold_metrics.append(fmetrics)
            pred_df['Feature_Set']=fs_name; all_preds.append(pred_df)
            selected_cols_report += [{'Feature_Set':fs_name,'Feature':c} for c in cols]
        except Exception as e:
            summary.append({'Feature_Set':fs_name,'Target':target,'Model':'PLSR_fixed3','Error':str(e)})
# Tuned only for curated green + variability
for target in ['OD750','OD665']:
    overall, fmetrics, pred_df, cols, tuning = lo_bo_tuned(df, green_variability, target, max_comp=10)
    overall['Feature_Set']='Green_plus_variability'
    summary.append(overall)
    fmetrics['Feature_Set']='Green_plus_variability'; fmetrics['Model']='PLSR_tuned'; all_fold_metrics.append(fmetrics)
    pred_df['Feature_Set']='Green_plus_variability'; all_preds.append(pred_df)
    tuning['Feature_Set']='Green_plus_variability'; all_tuning.append(tuning)

summary_df=pd.DataFrame(summary)
# reorder columns
summary_df=summary_df[[c for c in ['Feature_Set','Model','Target','R2','RMSE','MAE','n_components','n_features','n_obs','Error'] if c in summary_df.columns]]
summary_df.to_csv(os.path.join(OUT,'curated_feature_set_model_summary.csv'),index=False)
pd.concat(all_fold_metrics, ignore_index=True).to_csv(os.path.join(OUT,'curated_feature_set_fold_metrics.csv'),index=False)
pd.concat(all_preds, ignore_index=True).to_csv(os.path.join(OUT,'curated_feature_set_predictions.csv'),index=False)
if all_tuning:
    pd.concat(all_tuning, ignore_index=True).to_csv(os.path.join(OUT,'green_variability_inner_tuning_results.csv'),index=False)
pd.DataFrame(selected_cols_report).drop_duplicates().to_csv(os.path.join(OUT,'feature_sets_used.csv'),index=False)

# Final PLSR fit on all data for Green+variability with selected components based on average from tuned? Also fixed3 for VIP.
for target in ['OD750','OD665']:
    temp, cols = prepare_data(df, green_variability, target)
    X=temp[cols].values; y=temp[target].values
    ncomp=3
    pipe=Pipeline([('scaler', StandardScaler()), ('pls', PLSRegression(n_components=ncomp, scale=False))])
    pipe.fit(X,y)
    X_scaled=pipe.named_steps['scaler'].transform(X)
    pls=pipe.named_steps['pls']
    vip=calc_vip(pls, X_scaled, y)
    coef=pls.coef_.ravel()
    vip_df=pd.DataFrame({'Feature':cols,'VIP':vip,'Coefficient':coef})
    vip_df['abs_Coefficient']=vip_df['Coefficient'].abs()
    vip_df=vip_df.sort_values('VIP',ascending=False)
    vip_df.to_csv(os.path.join(OUT,f'PLSR_green_variability_VIP_coefficients_{target}.csv'),index=False)
    # Plot top 15 VIP
    top=vip_df.head(15).iloc[::-1]
    plt.figure(figsize=(9,6))
    plt.barh(top['Feature'], top['VIP'])
    plt.axvline(1.0, linestyle='--', linewidth=1.5)
    plt.xlabel('VIP score', fontsize=13)
    plt.ylabel('Feature', fontsize=13)
    plt.title(f'Top VIP features for {target}\nGreen + variability PLSR', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT,f'VIP_top15_green_variability_{target}.png'),dpi=600,bbox_inches='tight')
    plt.savefig(os.path.join(OUT,f'VIP_top15_green_variability_{target}.svg'),bbox_inches='tight')
    plt.close()

# True vs predicted for green+variability fixed3 and tuned
preds=pd.concat(all_preds, ignore_index=True)
for model in ['PLSR_fixed3','PLSR_tuned']:
    for target in ['OD750','OD665']:
        temp=preds[(preds['Feature_Set']=='Green_plus_variability') & (preds['Model']==model) & (preds['Target']==target)]
        if len(temp)==0: continue
        met=metrics(temp['Observed'],temp['Predicted'])
        plt.figure(figsize=(6.5,6))
        for b in sorted(temp['Batch'].dropna().unique()):
            sub=temp[temp['Batch']==b]
            plt.scatter(sub['Observed'], sub['Predicted'], s=70, edgecolor='black', linewidth=0.6, alpha=0.85, label=b)
        mn=min(temp['Observed'].min(), temp['Predicted'].min())
        mx=max(temp['Observed'].max(), temp['Predicted'].max())
        plt.plot([mn,mx],[mn,mx], linestyle='--', linewidth=2, label='Ideal fit')
        plt.xlabel(f'Observed {target}', fontsize=13)
        plt.ylabel(f'Predicted {target}', fontsize=13)
        plt.title(f'{target}: Green + variability {model}\nR²={met["R2"]:.3f}, RMSE={met["RMSE"]:.3f}, MAE={met["MAE"]:.3f}', fontsize=13)
        plt.legend(frameon=False, fontsize=10)
        plt.tight_layout()
        plt.savefig(os.path.join(OUT,f'true_vs_pred_green_variability_{model}_{target}.png'),dpi=600,bbox_inches='tight')
        plt.savefig(os.path.join(OUT,f'true_vs_pred_green_variability_{model}_{target}.svg'),bbox_inches='tight')
        plt.close()

# Residual plots
for target in ['OD750','OD665']:
    temp=preds[(preds['Feature_Set']=='Green_plus_variability') & (preds['Model']=='PLSR_fixed3') & (preds['Target']==target)].copy()
    if len(temp)==0: continue
    temp['Residual']=temp['Observed']-temp['Predicted']
    plt.figure(figsize=(7,5.5))
    plt.scatter(temp['Predicted'], temp['Residual'], s=70, edgecolor='black', linewidth=0.6, alpha=0.85)
    plt.axhline(0, linestyle='--', linewidth=1.5)
    plt.xlabel(f'Predicted {target}', fontsize=13)
    plt.ylabel('Residual', fontsize=13)
    plt.title(f'Residual plot: Green + variability PLSR {target}', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT,f'residual_green_variability_PLSR_fixed3_{target}.png'),dpi=600,bbox_inches='tight')
    plt.savefig(os.path.join(OUT,f'residual_green_variability_PLSR_fixed3_{target}.svg'),bbox_inches='tight')
    plt.close()

# Readme
readme=f"""PLSR Green + Variability curated feature-set analysis

Input file: {INPUT}
Main dataset: image-level average from 4 ROIs per image.
Number of observations: {len(df)} image-level rows.
Validation: leave-one-batch-out cross-validation using Batch parsed from image filenames.
Targets: OD750 and OD665.
Day, image size, OD values, Specific GR, and ROI_Selection_Score were not used as model inputs.

Feature sets:
- Full_features: all numeric image-derived features except metadata, targets, day, and ROI_Selection_Score.
- Color_mean_only: averaged color features only, excluding SD-based features.
- Green_only: green-focused averaged features.
- Green_plus_variability: green-focused averaged features plus within-ROI and among-ROI variability features.

The green + variability set was designed to test whether greenness plus optical/color heterogeneity can improve OD estimation.
Files included:
- curated_feature_set_model_summary.csv
- curated_feature_set_fold_metrics.csv
- curated_feature_set_predictions.csv
- green_variability_inner_tuning_results.csv
- feature_sets_used.csv
- PLSR_green_variability_VIP_coefficients_OD750.csv
- PLSR_green_variability_VIP_coefficients_OD665.csv
- true_vs_pred plots, residual plots, and VIP plots in PNG/SVG.
"""
with open(os.path.join(OUT,'analysis_readme.txt'),'w') as f:
    f.write(readme)

# Zip output
zip_path='/mnt/data/PLSR_green_variability_results.zip'
with zipfile.ZipFile(zip_path,'w',zipfile.ZIP_DEFLATED) as z:
    for root, dirs, files in os.walk(OUT):
        for file in files:
            fp=os.path.join(root,file)
            z.write(fp, arcname=os.path.relpath(fp, OUT))

print(summary_df.to_string(index=False))
print('ZIP:',zip_path)
