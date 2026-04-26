# =============================================================================
# 01_setup_environment.py
# Stage: Setup
# Description: Installs all dependencies and verifies environment
# =============================================================================

import subprocess
import sys

def install_dependencies():
    packages = [
        "pandas>=2.0",
        "numpy>=1.24",
        "scikit-learn>=1.3",
        "xgboost>=2.0",
        "imbalanced-learn>=0.11",
        "matplotlib>=3.7",
        "seaborn>=0.12",
        "scipy>=1.11",
        "jupyter>=1.0",
        "joblib>=1.3"
    ]
    print("Installing dependencies...")
    for pkg in packages:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])
    print("All dependencies installed successfully.")

def verify_environment():
    import pandas as pd
    import numpy as np
    import sklearn
    import xgboost as xgb
    import imblearn
    import matplotlib
    import scipy

    print("\nEnvironment Verification:")
    print(f"  pandas:        {pd.__version__}")
    print(f"  numpy:         {np.__version__}")
    print(f"  scikit-learn:  {sklearn.__version__}")
    print(f"  xgboost:       {xgb.__version__}")
    print(f"  imbalanced-learn: {imblearn.__version__}")
    print(f"  matplotlib:    {matplotlib.__version__}")
    print(f"  scipy:         {scipy.__version__}")
    print("\nEnvironment ready. No GPU required — CPU sufficient for all pipeline steps.")

if __name__ == "__main__":
    install_dependencies()
    verify_environment()
