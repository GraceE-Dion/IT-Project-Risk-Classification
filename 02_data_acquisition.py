# =============================================================================
# 02_data_acquisition.py
# Stage: Data
# Description: Downloads synthetic dataset from Kaggle and NASA MDP JM1.arff
#              directly from the NASADefectDataset GitHub repository
# =============================================================================

import urllib.request
import os

# -----------------------------------------------------------------------------
# Session 1 — Synthetic Dataset
# Source: Kaggle — Project Management Risk Raw (ka66ledata)
# Note: AI-generated dataset. Used for baseline methodology only.
#       Not suitable for real-world inference.
# -----------------------------------------------------------------------------
SYNTHETIC_KAGGLE_PATH = "/kaggle/input/datasets/ka66ledata/project-management-risk-raw/project_risk_raw_dataset.csv"

def check_synthetic_dataset():
    if os.path.exists(SYNTHETIC_KAGGLE_PATH):
        print(f"Synthetic dataset found at: {SYNTHETIC_KAGGLE_PATH}")
    else:
        print("Synthetic dataset not found.")
        print("To use: open Kaggle notebook from https://www.kaggle.com/datasets/ka66ledata/project-management-risk-raw")
        print("Dataset will be auto-attached at the path above.")

# -----------------------------------------------------------------------------
# Session 2 — NASA Raw MDP Data
# Source: NASADefectDataset GitHub Repository
# URL: https://github.com/klainfo/NASADefectDataset/tree/master/OriginalData/MDP
# File: JM1.arff — real software project metrics from NASA Metrics Data Program
# License: Public domain — peer-reviewed, widely cited in software engineering research
#
# IMPORTANT: Raw ARFF file used instead of Kaggle preprocessed version.
# The Kaggle version normalizes all features to [0,1] and reduces unique values
# (e.g. CYCLO: 24 unique values, INT_FAN_IN: 10 unique values), obscuring
# authentic data distributions. Raw source preserves data fidelity.
# -----------------------------------------------------------------------------
NASA_MDP_URL = "https://raw.githubusercontent.com/klainfo/NASADefectDataset/master/OriginalData/MDP/JM1.arff"
NASA_MDP_LOCAL = "JM1.arff"

def download_nasa_mdp():
    print("Downloading NASA MDP JM1.arff from GitHub...")
    urllib.request.urlretrieve(NASA_MDP_URL, NASA_MDP_LOCAL)
    print(f"Downloaded successfully: {NASA_MDP_LOCAL}")
    print("Source: NASA Metrics Data Program (MDP)")
    print("Cite: Menzies, T., et al. (2004). Data Mining Static Code Attributes to Learn Defect Predictors.")

if __name__ == "__main__":
    print("=" * 55)
    print("SESSION 1 — Synthetic Dataset")
    print("=" * 55)
    check_synthetic_dataset()

    print("\n" + "=" * 55)
    print("SESSION 2 — NASA Raw MDP Data")
    print("=" * 55)
    download_nasa_mdp()
