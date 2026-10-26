from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import pearsonr
from analysis_utils import load_image_level_data, get_core_color_variability_features
ROOT=Path(__file__).resolve().parents[1]; DATA=ROOT/'data'/'microalgae_pbr_image_level_features.csv'; OUT=ROOT/'outputs'/'01_PCA_results'; OUT.mkdir(parents=True, exist_ok=True)
df=load_image_level_data(DATA); features=get_core_color_variability_features(df); use=df.dropna(subset=features).copy()
X=StandardScaler().fit_transform(use[features].values); pca=PCA(n_components=min(5, X.shape[1], X.shape[0])); pcs=pca.fit_transform(X); expl=pca.explained_variance_ratio_*100
for i in range(min(3,pcs.shape[1])): use[f'PC{i+1}']=pcs[:,i]
score_cols=[c for c in ['Image','Batch','Environment','PBR_Type','Day','PC1','PC2','PC3','OD665','OD750','Specific GR'] if c in use.columns]
use[score_cols].to_csv(OUT/'PCA_scores.csv', index=False)
pd.DataFrame(pca.components_.T,index=features,columns=[f'PC{i+1}' for i in range(pca.components_.shape[0])]).to_csv(OUT/'PCA_loadings.csv')
corr=[]
for var in ['Day','OD665','OD750','Specific GR']:
    if var in use.columns:
        tmp=use.dropna(subset=['PC1','PC2',var])
        for pc in ['PC1','PC2']:
            r,p=pearsonr(tmp[pc], tmp[var]); corr.append({'Variable':var,'PC':pc,'r':r,'p':p,'n':len(tmp)})
pd.DataFrame(corr).to_csv(OUT/'PCA_correlations.csv', index=False)
def score_plot(gcol, fname):
    if gcol not in use.columns: return
    plt.figure(figsize=(7,6))
    for lab,g in use.groupby(gcol): plt.scatter(g.PC1,g.PC2,s=70,edgecolor='black',linewidth=0.5,alpha=0.85,label=str(lab))
    plt.xlabel(f'PC1 ({expl[0]:.1f}%)'); plt.ylabel(f'PC2 ({expl[1]:.1f}%)'); plt.legend(frameon=False); plt.tight_layout(); plt.savefig(OUT/fname,dpi=600,bbox_inches='tight'); plt.close()
for c,n in [('Day','PCA_score_by_day.png'),('Environment','PCA_score_by_environment.png'),('PBR_Type','PCA_score_by_PBR_type.png')]: score_plot(c,n)
for target in ['OD750','OD665']:
    tmp=use.dropna(subset=['PC1',target]); r,p=pearsonr(tmp.PC1,tmp[target]); plt.figure(figsize=(6,5)); plt.scatter(tmp.PC1,tmp[target],s=70,edgecolor='black',linewidth=0.5,alpha=0.85); z=np.polyfit(tmp.PC1,tmp[target],1); x=np.linspace(tmp.PC1.min(),tmp.PC1.max(),100); plt.plot(x,np.polyval(z,x),linestyle='--',linewidth=2); plt.xlabel(f'PC1 ({expl[0]:.1f}%)'); plt.ylabel(target); plt.title(f'PC1 vs {target}: r={r:.3f}, p={p:.4g}'); plt.tight_layout(); plt.savefig(OUT/f'PC1_vs_{target}.png',dpi=600,bbox_inches='tight'); plt.close()
print('PCA completed:', OUT)
