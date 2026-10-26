from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy.stats import spearmanr, kruskal, mannwhitneyu
from analysis_utils import regression_metrics
ROOT=Path(__file__).resolve().parents[1]; PRED=ROOT/'outputs'/'02_PLSR_main_model'/'PLSR_fixed3_LOBO_predictions.csv'; OUT=ROOT/'outputs'/'06_residual_and_subgroup_error_analysis'; OUT.mkdir(parents=True, exist_ok=True)
if not PRED.exists(): raise FileNotFoundError('Run 02_plsr_main_model.py first.')
pred=pd.read_csv(PRED); pred['Abs_Error']=np.abs(pred.Predicted-pred.Observed); pred['Residual']=pred.Observed-pred.Predicted; pred.to_csv(OUT/'PLSR_predictions_with_residuals.csv',index=False)
rows=[]
for target,g in pred.groupby('Target'):
    m=regression_metrics(g.Observed,g.Predicted); rho,p=spearmanr(g.Predicted,g.Residual); m.update({'Target':target,'n':len(g),'Spearman_residual_vs_predicted_r':rho,'Spearman_residual_vs_predicted_p':p}); rows.append(m)
pd.DataFrame(rows).to_csv(OUT/'PLSR_residual_overall_metrics.csv',index=False)
for gcol in ['Batch','Environment','PBR_Type']:
    g_rows=[]; t_rows=[]
    for target,sub in pred.groupby('Target'):
        for lab,g in sub.groupby(gcol):
            m=regression_metrics(g.Observed,g.Predicted); m.update({'Target':target,gcol:lab,'n':len(g),'Abs_Error_mean':g.Abs_Error.mean(),'Abs_Error_median':g.Abs_Error.median(),'NRMSE_by_mean_percent':m['RMSE']/g.Observed.mean()*100,'NMAE_by_mean_percent':m['MAE']/g.Observed.mean()*100}); g_rows.append(m)
        groups=[x.Abs_Error.to_numpy() for _,x in sub.groupby(gcol)]
        if len(groups)==2:
            labs=list(sub.groupby(gcol).groups.keys()); stat,p=mannwhitneyu(groups[0],groups[1],alternative='two-sided'); t_rows.append({'Target':target,'Group':gcol,'Test':'Mann-Whitney U','p':p,'Group1':labs[0],'Group2':labs[1]})
        elif len(groups)>2:
            stat,p=kruskal(*groups); t_rows.append({'Target':target,'Group':gcol,'Test':'Kruskal-Wallis','p':p})
    pd.DataFrame(g_rows).to_csv(OUT/f'PLSR_residual_metrics_by_{gcol}.csv',index=False); pd.DataFrame(t_rows).to_csv(OUT/f'PLSR_abs_error_tests_by_{gcol}.csv',index=False)
for target,sub in pred.groupby('Target'):
    plt.figure(figsize=(6,5)); plt.scatter(sub.Predicted,sub.Residual,s=70,edgecolor='black',linewidth=0.5,alpha=0.85); plt.axhline(0,linestyle='--',linewidth=1.5); plt.xlabel(f'Predicted {target}'); plt.ylabel('Residual, observed - predicted'); plt.tight_layout(); plt.savefig(OUT/f'PLSR_residual_vs_predicted_{target}.png',dpi=600,bbox_inches='tight'); plt.close()
    for gcol in ['Batch','Environment','PBR_Type']:
        labs=[]; data=[]
        for lab,g in sub.groupby(gcol): labs.append(str(lab)); data.append(g.Abs_Error)
        plt.figure(figsize=(6,5)); plt.boxplot(data,labels=labs,showmeans=True); plt.ylabel(f'Absolute error {target}'); plt.tight_layout(); plt.savefig(OUT/f'PLSR_absolute_error_by_{gcol}_{target}.png',dpi=600,bbox_inches='tight'); plt.close()
print('Residual and subgroup error analysis completed:', OUT)
