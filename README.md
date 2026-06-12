# Multi-omics Machine Learning for Breast-Cancer Drug-Response Prediction

Code accompanying the MSc dissertation **"Employing machine learning
algorithms to predict the response of breast cancer cells to small molecule
inhibitors using DNA Methylation and Metabolomics"**
(Andiswa Majikijela, MJKAND006; Department of Integrative Biomedical Sciences,
University of Cape Town; supervisor: Dr. Musalula Sinkala, 2026).

This repository integrates DNA-methylation and metabolomics profiles of cancer
cell lines (CCLE/DepMap) with drug-response measurements (GDSC) and trains
machine-learning models to predict how breast-cancer cell lines respond to
small-molecule inhibitors, both as a **regression** task (continuous `LN_IC50`)
and a **classification** task (Sensitive vs Resistant).

## What the pipeline does

1. **Integrate** four data sources on a shared cell-line key, aggregating
   methylation CpG sites to gene level via a variance-weighted mean.
2. **Filter** to breast-cancer cell lines and **impute** missing values
   (KNN imputation after dropping high-missingness columns).
3. **Select features** using a low-variance filter combined with the union of
   mutual-information and F-test rankings.
4. **Model**
   - *Regression* (`LN_IC50`): Random Forest, XGBoost, LightGBM, CatBoost, a
     1D-CNN with an attention block, and a stacking ensemble.
   - *Classification* (Sensitive/Resistant from the `LN_IC50` Z-score):
     logistic regression, Random Forest, XGBoost, LightGBM, CatBoost and a
     1D-CNN.
5. **Interpret** the models with SHAP and link the most important methylation
   genes and metabolites back to drug response with statistical association
   tests (Spearman, Mann-Whitney U, Cohen's d; Benjamini-Hochberg FDR).

## Repository layout

```
.
├── config.py                 # central paths, parameters and thresholds
├── run_regression.py         # entry point: LN_IC50 regression pipeline
├── run_classification.py     # entry point: Sensitive/Resistant classification
├── src/
│   ├── data_pipeline.py      # load, clean, integrate, filter, impute (shared)
│   ├── features.py           # feature selection, scaling, train/val/test split
│   ├── regression.py         # regression models + CNN + evaluation
│   ├── classification.py     # binary target + classifiers + CNN + evaluation
│   └── interpretation.py     # SHAP + gene/metabolite-drug associations
├── data/                     # raw inputs (not tracked) + acquisition guide
├── requirements.txt
├── LICENSE
└── .gitignore
```

The two analyses share `src/data_pipeline.py` and `src/features.py` so the
data integration and feature selection are identical and the regression and
classification results are directly comparable.

## Installation

```bash
git clone <repository-url>
cd <repository>
python -m venv .venv && source .venv/bin/activate   # optional
pip install -r requirements.txt
```

Tested with Python 3.10. TensorFlow is required only for the CNN models; pass
`--no-cnn` to either script to run without it.

## Data

The raw data files are not redistributed here. Download the CCLE/DepMap and
GDSC files listed in [`data/README.md`](data/README.md) and place them in
`data/`, or point the `DATA_DIR` environment variable at their location.

## Usage

```bash
# Regression: predict continuous LN_IC50
python run_regression.py

# Classification: predict Sensitive vs Resistant
python run_classification.py

# Skip the CNN (no TensorFlow required)
python run_regression.py --no-cnn
```

Outputs (comparison tables, feature-importance rankings, association results
and predictions) are written to `results/tables/` by default. Override the
location with the `OUTPUT_DIR` environment variable.

## Configuration

All tunable settings live in `config.py`: input file names, the missing-value
threshold, KNN neighbours, feature-selection sizes, the train/validation/test
split fractions, the random seed (42), and the Z-score cut-offs that define the
Sensitive (`< -0.5`) and Resistant (`> 0.5`) classes.

## Reproducibility notes

- A fixed random seed (`config.RANDOM_SEED = 42`) is set for NumPy and the
  models. Grid-searched models and the CNN may still show minor run-to-run
  variation depending on the BLAS/TensorFlow backend and hardware.
- The grid searches over Random Forest, XGBoost and LightGBM are the most
  time-consuming step. Reduce the grids in `src/regression.py` for a faster
  smoke test.

## Citation

If you use this code, please cite the dissertation:

> Majikijela, A. (2026). *Employing machine learning algorithms to predict the
> response of breast cancer cells to small molecule inhibitors using DNA
> Methylation and Metabolomics.* MSc dissertation, University of Cape Town.

## License

Released under the MIT License — see [`LICENSE`](LICENSE).

