from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneGroupOut, GroupKFold, GridSearchCV
from analysis_utils import load_image_level_data, get_core_color_variability_features, regression_metrics
ROOT=Path(__file__).resolve().parents[1]; DATA=ROOT/'data'/'microalgae_pbr_image_level_features.csv'; OUT=ROOT/'outputs'/'04_SVR_benchmark'; OUT.mkdir(parents=True, exist_ok=True)
df=load_image_level_data(DATA); features=get_core_color_variability_features(df); grid={'svr__kernel':['rbf','linear'],'svr__C':[0.1,1,10,100],'svr__epsilon':[0.01,0.05,0.1],'svr__gamma':['scale','auto']}; rows=[]; metrics=[]; best=[]
for target in ['OD750','OD665']:
    use=df.dropna(subset=features+[target,'Batch']).copy(); X=use[features].to_numpy(float); y=use[target].to_numpy(float); groups=use.Batch.astype(str).to_numpy(); target_rows=[]
    for fold,(tr,te) in enumerate(LeaveOneGroupOut().split(X,y,groups=groups),start=1):
        cv=GroupKFold(n_splits=len(np.unique(groups[tr]))); pipe=Pipeline([('scaler',StandardScaler()),('svr',SVR())]); gs=GridSearchCV(pipe,grid,cv=cv,scoring='neg_root_mean_squared_error',n_jobs=-1); gs.fit(X[tr],y[tr],groups=groups[tr]); pred=gs.predict(X[te])
        tmp=use.iloc[te][['Image','Batch','Environment','PBR_Type','Day']].copy(); tmp['Target']=target; tmp['Fold']=fold; tmp['Observed']=y[te]; tmp['Predicted']=pred; tmp['Residual']=tmp.Observed-tmp.Predicted; tmp['Error']=tmp.Predicted-tmp.Observed; tmp['Abs_Error']=tmp.Error.abs(); rows.append(tmp); target_rows.append(tmp); br={'Target':target,'Fold':fold}; br.update(gs.best_params_); best.append(br)
    pred_all=pd.concat(target_rows,ignore_index=True); m=regression_metrics(pred_all.Observed,pred_all.Predicted); m.update({'Target':target,'Model':'SVR_nested_LOBO','n':len(pred_all),'n_features':len(features)}); metrics.append(m)
    plt.figure(figsize=(6,5)); plt.scatter(pred_all.Observed,pred_all.Predicted,s=70,edgecolor='black',linewidth=0.5,alpha=0.85); mn=min(pred_all.Observed.min(),pred_all.Predicted.min()); mx=max(pred_all.Observed.max(),pred_all.Predicted.max()); plt.plot([mn,mx],[mn,mx],linestyle='--',linewidth=2); plt.xlabel(f'Observed {target}'); plt.ylabel(f'Predicted {target}'); plt.title(f'{target} SVR, R²={m["R2"]:.3f}'); plt.tight_layout(); plt.savefig(OUT/f'SVR_true_vs_predicted_{target}.png',dpi=600,bbox_inches='tight'); plt.close()
pd.concat(rows,ignore_index=True).to_csv(OUT/'SVR_nested_LOBO_predictions.csv',index=False); pd.DataFrame(metrics).to_csv(OUT/'SVR_nested_LOBO_summary.csv',index=False); pd.DataFrame(best).to_csv(OUT/'SVR_best_hyperparameters_by_fold.csv',index=False)
print('SVR benchmark completed:', OUT)
