from pathlib import Path
import pandas as pd, matplotlib.pyplot as plt
from analysis_utils import load_image_level_data, get_core_color_variability_features, plsr_lobo_predictions, regression_metrics, fit_final_plsr_and_vip
ROOT=Path(__file__).resolve().parents[1]; DATA=ROOT/'data'/'microalgae_pbr_image_level_features.csv'; OUT=ROOT/'outputs'/'02_PLSR_main_model'; OUT.mkdir(parents=True, exist_ok=True)
df=load_image_level_data(DATA); features=get_core_color_variability_features(df); preds=[]; metrics=[]; group_metrics=[]
for target in ['OD750','OD665']:
    pred=plsr_lobo_predictions(df,features,target,n_components=3); preds.append(pred); m=regression_metrics(pred.Observed,pred.Predicted); m.update({'Target':target,'Model':'PLSR_fixed3_LOBO','n':len(pred),'n_features':len(features)}); metrics.append(m)
    for b,g in pred.groupby('Batch'):
        gm=regression_metrics(g.Observed,g.Predicted); gm.update({'Target':target,'Batch':b,'n':len(g)}); group_metrics.append(gm)
    fit_final_plsr_and_vip(df,features,target,3).to_csv(OUT/f'PLSR_VIP_coefficients_{target}.csv',index=False)
    plt.figure(figsize=(6,5)); plt.scatter(pred.Observed,pred.Predicted,s=70,edgecolor='black',linewidth=0.5,alpha=0.85); mn=min(pred.Observed.min(),pred.Predicted.min()); mx=max(pred.Observed.max(),pred.Predicted.max()); plt.plot([mn,mx],[mn,mx],linestyle='--',linewidth=2); plt.xlabel(f'Observed {target}'); plt.ylabel(f'Predicted {target}'); plt.title(f'{target} PLSR, R²={m["R2"]:.3f}'); plt.tight_layout(); plt.savefig(OUT/f'PLSR_true_vs_predicted_{target}.png',dpi=600,bbox_inches='tight'); plt.close()
pd.concat(preds,ignore_index=True).to_csv(OUT/'PLSR_fixed3_LOBO_predictions.csv',index=False); pd.DataFrame(metrics).to_csv(OUT/'PLSR_fixed3_LOBO_metrics.csv',index=False); pd.DataFrame(group_metrics).to_csv(OUT/'PLSR_group_metrics.csv',index=False); pd.DataFrame({'Feature':features}).to_csv(OUT/'features_used_core_color_withinROI_variability.csv',index=False)
print('PLSR main model completed:', OUT)
