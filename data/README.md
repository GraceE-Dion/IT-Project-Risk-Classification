# Dataset

The datasets used in this project are not included in this repository due to file size.

## Session 1 — Synthetic Dataset
**Source:** Kaggle — Project Management Risk Raw (ka66ledata)
**Kaggle URL:** https://www.kaggle.com/datasets/ka66ledata/project-management-risk-raw
**Kaggle path:**
BASE = '/kaggle/input/datasets/ka66ledata/project-management-risk-raw/project_risk_raw_dataset.csv'

## Session 2 — NASA Raw MDP Data
**Source:** NASADefectDataset GitHub Repository
**GitHub URL:** https://github.com/klainfo/NASADefectDataset/tree/master/OriginalData/MDP
**File:** JM1.arff — downloaded directly in notebook via urllib, no manual download needed
**Download code:**
import urllib.request
url = "https://raw.githubusercontent.com/klainfo/NASADefectDataset/master/OriginalData/MDP/JM1.arff"
urllib.request.urlretrieve(url, "JM1.arff")
