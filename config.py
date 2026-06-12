"""
Central configuration for the multi-omics breast-cancer drug-response pipeline.

All file paths, output locations, and key hyperparameters live here so the
analysis can be reproduced on any machine by editing a single file (or by
setting the corresponding environment variables) rather than hunting for
hard-coded paths inside the scripts.
"""

import os
from pathlib import Path

# --------------------------------------------------------------------------- #
# Directories
# --------------------------------------------------------------------------- #
# Root of the repository (directory that contains this file).
ROOT_DIR = Path(__file__).resolve().parent

# Where the raw input data files live. Override with the DATA_DIR environment
# variable if your data is stored elsewhere, e.g.
#   export DATA_DIR=/path/to/datasets
DATA_DIR = Path(os.environ.get("DATA_DIR", ROOT_DIR / "data"))

# Where figures, result tables and model artefacts are written.
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", ROOT_DIR / "results"))
FIGURE_DIR = OUTPUT_DIR / "figures"
TABLE_DIR = OUTPUT_DIR / "tables"

# --------------------------------------------------------------------------- #
# Input data files
# --------------------------------------------------------------------------- #
# CCLE multi-omics + sample annotation and GDSC drug-response data.
# See data/README.md for download instructions and the expected schema.
METABOLOMICS_FILE = DATA_DIR / "Metabolomics_subsetted.csv"
METHYLATION_FILE = DATA_DIR / "Methylation_subsetted.csv"
SAMPLE_INFO_FILE = DATA_DIR / "Achilles_sample_info_19Q4.csv"
GDSC_RESPONSE_FILE = DATA_DIR / "GDSC2_fitted_dose_response_15Oct19.xlsx"
GDSC_DRUGS_FILE = DATA_DIR / "GDSC2_drugs.csv"

# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
RANDOM_SEED = 42

# --------------------------------------------------------------------------- #
# Pre-processing parameters
# --------------------------------------------------------------------------- #
# Drop feature columns whose share of missing values exceeds this percentage.
MISSING_THRESHOLD_PCT = 50
# Number of neighbours used by the KNN imputer.
KNN_NEIGHBORS = 5
# Drug-response target columns that must never be treated as input features.
TARGET_COLUMNS = ["LN_IC50", "AUC", "RMSE", "Z_SCORE"]

# --------------------------------------------------------------------------- #
# Feature-selection parameters
# --------------------------------------------------------------------------- #
VARIANCE_THRESHOLD = 0.01      # VarianceThreshold cut-off
K_BEST_MUTUAL_INFO = 200       # top-K features kept by mutual information
K_BEST_F_REGRESSION = 200      # top-K features kept by the F-test
PCA_COMPONENTS = 50            # components retained for the PCA diagnostic

# --------------------------------------------------------------------------- #
# Train / validation / test split
# --------------------------------------------------------------------------- #
TEST_SIZE = 0.15               # fraction held out as the test set
VAL_SIZE_OF_REMAINDER = 0.176  # ~0.15 of the full data once the test set is removed
STRATIFY_BINS = 5              # quantile bins used to stratify the regression split

# --------------------------------------------------------------------------- #
# Regression target
# --------------------------------------------------------------------------- #
REGRESSION_TARGET = "LN_IC50"

# --------------------------------------------------------------------------- #
# Classification target (drug-response Z-score thresholds)
# --------------------------------------------------------------------------- #
# Sensitive: Z-score < SENSITIVE_CUTOFF  (lower IC50 -> more sensitive)
# Resistant: Z-score > RESISTANT_CUTOFF  (higher IC50 -> more resistant)
# Samples in the intermediate band are excluded from the classification task.
SENSITIVE_CUTOFF = -0.5
RESISTANT_CUTOFF = 0.5


def ensure_output_dirs():
    """Create the output directory tree if it does not already exist."""
    for directory in (OUTPUT_DIR, FIGURE_DIR, TABLE_DIR):
        directory.mkdir(parents=True, exist_ok=True)
