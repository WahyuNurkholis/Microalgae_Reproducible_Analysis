from pathlib import Path
import pandas as pd
from analysis_utils import load_image_level_data, get_green_feature_sets, plsr_lobo_predictions, regression_metrics
ROOT=Path(__file__).resolve().parents[1]; DATA=ROOT/'data'/'microalgae_pbr_image_level_features.csv'; OUT=ROOT/'outputs'/'05_feature_set_comparison'; OUT.mkdir(parents=True, exist_ok=True)
df=load_image_level_data(DATA); feature_sets=get_green_feature_sets(df); summary=[]; preds=[]
for name,features in feature_sets.items():
    if len(features)<2: continue
    for target in ['OD750','OD665']:
        pred=plsr_lobo_predictions(df,features,target,n_components=3); pred['Feature_Set']=name; preds.append(pred); m=regression_metrics(pred.Observed,pred.Predicted); m.update({'Feature_Set':name,'Target':target,'n_features':len(features),'n':len(pred)}); summary.append(m)
    pd.DataFrame({'Feature':features}).to_csv(OUT/f'features_{name}.csv',index=False)
pd.DataFrame(summary).to_csv(OUT/'feature_set_model_summary.csv',index=False); pd.concat(preds,ignore_index=True).to_csv(OUT/'feature_set_predictions.csv',index=False)
print('Feature set comparison completed:', OUT)
