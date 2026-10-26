import subprocess, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
for s in ['01_pca_analysis.py','02_plsr_main_model.py','03_random_forest_benchmark.py','04_svr_benchmark.py','05_feature_set_comparison_plsr.py','06_residual_and_subgroup_error_analysis.py']:
    print('\n=== Running',s,'===')
    subprocess.run([sys.executable,str(ROOT/s)],check=True)
print('\nAll analyses completed. Check outputs folder.')
