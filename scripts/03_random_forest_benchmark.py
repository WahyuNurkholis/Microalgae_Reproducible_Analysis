from pathlib import Path
import pandas as pd, matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneGroupOut
from analysis_utils import load_image_level_data, get_core_color_variability_features, regression_metrics
ROOT=Path(__file__).resolve().parents[1]; DATA=ROOT/'data'/'microalgae_pbr_image_level_features.csv'; OUT=ROOT/'outputs'/'03_RF_benchmark'; OUT.mkdir(parents=True, exist_ok=True)
df=load_image_level_data(DATA); features=get_core_color_variability_features(df); rows=[]; metrics=[]; imps=[]
for target in ['OD750','OD665']:
    use=df.dropna(subset=features+[target,'Batch']).copy(); X=use[features].to_numpy(float); y=use[target].to_numpy(float); groups=use.Batch.astype(str).to_numpy()
    target_rows=[]
    for fold,(tr,te) in enumerate(LeaveOneGroupOut().split(X,y,groups=groups),start=1):
        rf=RandomForestRegressor(n_estimators=500,max_depth=4,min_samples_leaf=2,random_state=42,n_jobs=-1)
        rf.fit(X[tr],y[tr]); pred=rf.predict(X[te])
        tmp=use.iloc[te][['Image','Batch','Environment','PBR_Type','Day']].copy(); tmp['Target']=target; tmp['Fold']=fold; tmp['Observed']=y[te]; tmp['Predicted']=pred; tmp['Residual']=tmp.Observed-tmp.Predicted; tmp['Error']=tmp.Predicted-tmp.Observed; tmp['Abs_Error']=tmp.Error.abs(); rows.append(tmp); target_rows.append(tmp)
        for f,v in zip(features,rf.feature_importances_): imps.append({'Target':target,'Fold':fold,'Feature':f,'Importance':v})
    pred_all=pd.concat(target_rows,ignore_index=True); m=regression_metrics(pred_all.Observed,pred_all.Predicted); m.update({'Target':target,'Model':'RF_fixed_LOBO','n':len(pred_all),'n_features':len(features)}); metrics.append(m)
    plt.figure(figsize=(6,5)); plt.scatter(pred_all.Observed,pred_all.Predicted,s=70,edgecolor='black',linewidth=0.5,alpha=0.85); mn=min(pred_all.Observed.min(),pred_all.Predicted.min()); mx=max(pred_all.Observed.max(),pred_all.Predicted.max()); plt.plot([mn,mx],[mn,mx],linestyle='--',linewidth=2); plt.xlabel(f'Observed {target}'); plt.ylabel(f'Predicted {target}'); plt.title(f'{target} RF, R²={m["R2"]:.3f}'); plt.tight_layout(); plt.savefig(OUT/f'RF_true_vs_predicted_{target}.png',dpi=600,bbox_inches='tight'); plt.close()
pd.concat(rows,ignore_index=True).to_csv(OUT/'RF_LOBO_predictions.csv',index=False); pd.DataFrame(metrics).to_csv(OUT/'RF_LOBO_overall_metrics.csv',index=False); imp=pd.DataFrame(imps); imp.to_csv(OUT/'RF_feature_importance_by_fold.csv',index=False); imp.groupby(['Target','Feature'],as_index=False).Importance.mean().sort_values(['Target','Importance'],ascending=[True,False]).to_csv(OUT/'RF_feature_importance_summary.csv',index=False)
print('RF benchmark completed:', OUT)
