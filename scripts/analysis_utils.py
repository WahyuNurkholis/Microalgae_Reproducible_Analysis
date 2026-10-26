import os, re, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression

def parse_metadata_from_image_name(name):
    s = str(name)
    stem = os.path.splitext(s)[0]
    m_batch = re.search(r'^(B\d+)', stem, flags=re.I)
    batch = m_batch.group(1).upper() if m_batch else None
    low = stem.lower()
    pbr = 'FPPBR' if '_flat_' in low else ('TPBR' if ('_tube_' in low or '_tubular_' in low) else None)
    env = 'Indoor' if ('_in_' in low or '_indoor_' in low) else ('Outdoor' if ('_out_' in low or '_outdoor_' in low) else None)
    m_day = re.search(r'day\s*([0-9]+)', stem, flags=re.I)
    day = int(m_day.group(1)) if m_day else np.nan
    return pd.Series({'Batch': batch, 'PBR_Type': pbr, 'Environment': env, 'Day': day})

def load_image_level_data(path):
    df = pd.read_csv(path)
    if 'Image' not in df.columns:
        raise ValueError('Input file must contain an Image column.')
    meta = df['Image'].apply(parse_metadata_from_image_name)
    for c in meta.columns:
        if c not in df.columns or df[c].isna().all():
            df[c] = meta[c]
    for c in ['OD665','OD750','Specific GR']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    return df

def get_core_color_variability_features(df):
    """Main curated feature set used in the reported PLSR model.
    It combines color means plus within-ROI pixel variability averaged across 4 ROIs.
    It intentionally excludes ROI-to-ROI spatial SD features, green fraction, day,
    image size, target OD values, and ROI selection score.
    """
    raw = [
        'R_mean_mean4ROI','G_mean_mean4ROI','B_mean_mean4ROI',
        'Mean_RGB_mean4ROI','Grayscale_mean4ROI',
        'H_mean_deg_mean4ROI','S_mean_mean4ROI','V_mean_mean4ROI',
        'L_star_mean_mean4ROI','a_star_mean_mean4ROI','b_star_mean_mean4ROI',
        'r_norm_mean4ROI','g_norm_mean4ROI','b_norm_mean4ROI',
        'G_R_ratio_mean4ROI','G_B_ratio_mean4ROI','R_B_ratio_mean4ROI',
        'ExG_raw_mean4ROI','ExG_norm_mean4ROI','NGRDI_mean4ROI','VARI_mean4ROI','GLI_mean4ROI','BGI_mean4ROI',
        # within-ROI pixel-level variability, averaged over four ROIs
        'R_std_mean4ROI','G_std_mean4ROI','B_std_mean4ROI',
        'H_std_deg_mean4ROI','S_std_mean4ROI','V_std_mean4ROI',
        'L_star_std_mean4ROI','a_star_std_mean4ROI','b_star_std_mean4ROI'
    ]
    return [c for c in raw if c in df.columns]

def get_green_feature_sets(df):
    full = get_core_color_variability_features(df)
    green_only_raw = [
        'G_mean_mean4ROI','g_norm_mean4ROI','G_R_ratio_mean4ROI','G_B_ratio_mean4ROI',
        'ExG_raw_mean4ROI','ExG_norm_mean4ROI','NGRDI_mean4ROI','VARI_mean4ROI','GLI_mean4ROI',
        'Green_fraction_mean4ROI','H_mean_deg_mean4ROI','S_mean_mean4ROI','a_star_mean_mean4ROI','b_star_mean_mean4ROI'
    ]
    green_only = [c for c in green_only_raw if c in df.columns]
    green_var_raw = green_only + [
        'G_std_mean4ROI','V_std_mean4ROI','H_std_deg_mean4ROI','S_std_mean4ROI',
        'L_star_std_mean4ROI','a_star_std_mean4ROI','b_star_std_mean4ROI',
        'G_mean_sd4ROI','g_norm_sd4ROI','G_R_ratio_sd4ROI','G_B_ratio_sd4ROI',
        'ExG_raw_sd4ROI','ExG_norm_sd4ROI','NGRDI_sd4ROI','VARI_sd4ROI','GLI_sd4ROI',
        'Green_fraction_sd4ROI','H_mean_deg_sd4ROI','S_mean_sd4ROI','V_mean_sd4ROI',
        'L_star_mean_sd4ROI','a_star_mean_sd4ROI','b_star_mean_sd4ROI'
    ]
    green_var=[]
    for c in green_var_raw:
        if c in df.columns and c not in green_var:
            green_var.append(c)
    mean_only = [c for c in full if c.endswith('_mean4ROI') and '_std_' not in c]
    return {'green_only': green_only, 'green_plus_variability': green_var, 'color_mean_only': mean_only, 'core_color_withinROI_variability': full}

def regression_metrics(y_true, y_pred):
    y_true=np.asarray(y_true); y_pred=np.asarray(y_pred)
    return {'R2': r2_score(y_true, y_pred), 'RMSE': float(np.sqrt(mean_squared_error(y_true, y_pred))), 'MAE': float(mean_absolute_error(y_true, y_pred)), 'Mean_residual_observed_minus_predicted': float(np.mean(y_true-y_pred))}

def plsr_lobo_predictions(df, feature_cols, target, n_components=3, group_col='Batch'):
    use = df.dropna(subset=feature_cols+[target, group_col]).copy()
    X = use[feature_cols].to_numpy(float)
    y = use[target].to_numpy(float)
    groups = use[group_col].astype(str).to_numpy()
    n_components = int(min(n_components, X.shape[1], X.shape[0]-1))
    rows=[]
    for fold, (train_idx, test_idx) in enumerate(LeaveOneGroupOut().split(X, y, groups=groups), start=1):
        model = Pipeline([('scaler', StandardScaler()), ('plsr', PLSRegression(n_components=n_components))])
        model.fit(X[train_idx], y[train_idx])
        pred = model.predict(X[test_idx]).ravel()
        fold_df = use.iloc[test_idx][['Image','Batch','Environment','PBR_Type','Day']].copy()
        fold_df['Target'] = target
        fold_df['Fold'] = fold
        fold_df['Observed'] = y[test_idx]
        fold_df['Predicted'] = pred
        fold_df['Residual'] = fold_df['Observed'] - fold_df['Predicted']
        fold_df['Error'] = fold_df['Predicted'] - fold_df['Observed']
        fold_df['Abs_Error'] = fold_df['Error'].abs()
        rows.append(fold_df)
    return pd.concat(rows, ignore_index=True)

def plsr_vip(pls, X_scaled, y):
    t = pls.x_scores_; w = pls.x_weights_; q = pls.y_loadings_
    p, h = w.shape
    s = np.diag(t.T @ t @ q.T @ q).reshape(h, -1)
    total_s = np.sum(s)
    vip = np.zeros((p,))
    for i in range(p):
        weight = np.array([(w[i, j] / np.linalg.norm(w[:, j]))**2 for j in range(h)])
        vip[i] = np.sqrt(p * (s.T @ weight) / total_s)
    return vip

def fit_final_plsr_and_vip(df, feature_cols, target, n_components=3):
    use = df.dropna(subset=feature_cols+[target]).copy()
    X = use[feature_cols].to_numpy(float); y = use[target].to_numpy(float)
    scaler = StandardScaler(); Xs = scaler.fit_transform(X)
    n_components = int(min(n_components, X.shape[1], X.shape[0]-1))
    pls = PLSRegression(n_components=n_components)
    pls.fit(Xs, y)
    vip = plsr_vip(pls, Xs, y)
    coef = pls.coef_.ravel()
    return pd.DataFrame({'Feature': feature_cols, 'VIP': vip, 'Coefficient_scaled_X': coef}).sort_values('VIP', ascending=False)
