#!/usr/bin/env python
"""
End-to-end regression pipeline: predict continuous drug response (LN_IC50)
for breast-cancer cell lines from integrated DNA-methylation and metabolomics
features.

Steps
-----
1. Build the integrated, imputed breast-cancer dataset (shared front-end).
2. Select features (variance filter + mutual-information / F-test union).
3. Robustly scale and stratify-split into train / validation / test.
4. Train and evaluate six regressors (RF, XGBoost, LightGBM, CatBoost, CNN,
   stacking ensemble) and compare them.
5. Compute SHAP importance and gene/metabolite-drug associations.
6. Write the comparison table, feature-importance tables and association
   results to the output directory.

Usage
-----
    python run_regression.py [--no-cnn] [--quiet]

Set DATA_DIR / OUTPUT_DIR environment variables (or edit config.py) to point
at your data and desired output location.
"""

import argparse

import numpy as np
import pandas as pd

import config
from src import data_pipeline, features, regression, interpretation


def main(use_cnn=True, verbose=True):
    config.ensure_output_dirs()
    np.random.seed(config.RANDOM_SEED)

    # ------------------------------------------------------------------ #
    # 1. Shared data pipeline
    # ------------------------------------------------------------------ #
    print("=" * 80)
    print("STEP 1  Building integrated breast-cancer dataset")
    print("=" * 80)
    bundle = data_pipeline.build_breast_cancer_dataset(verbose=verbose)
    data = bundle["data"]
    methylation_features = bundle["methylation_features"]
    metabolomics_features = bundle["metabolomics_features"]

    # ------------------------------------------------------------------ #
    # 2. Feature engineering
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 80)
    print("STEP 2  Feature selection and scaling")
    print("=" * 80)
    X, y, _ = features.build_feature_matrix(
        data, methylation_features, metabolomics_features
    )
    selected, fs_diagnostics = features.select_features(X, y, verbose=verbose)
    X_selected = X[selected].copy()
    features.pca_diagnostic(X_selected, verbose=verbose)
    X_scaled, _ = features.scale_features(X_selected)
    splits = features.split_regression(X_scaled, y, verbose=verbose)
    X_train, X_val, X_test, y_train, y_val, y_test = splits

    # ------------------------------------------------------------------ #
    # 3. Model training
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 80)
    print("STEP 3  Training regression models")
    print("=" * 80)

    models, metrics, predictions = {}, {}, {}

    rf, metrics["Random Forest"], predictions["Random Forest"] = \
        regression.train_random_forest(splits, verbose=verbose)
    models["rf"] = rf

    xgb, metrics["XGBoost"], predictions["XGBoost"] = \
        regression.train_xgboost(splits, verbose=verbose)
    models["xgb"] = xgb

    lgbm, metrics["LightGBM"], predictions["LightGBM"] = \
        regression.train_lightgbm(splits, verbose=verbose)
    models["lgbm"] = lgbm

    cat, metrics["CatBoost"], predictions["CatBoost"] = \
        regression.train_catboost(splits, verbose=verbose)
    models["catboost"] = cat

    if use_cnn:
        try:
            cnn, metrics["CNN"], predictions["CNN"], _ = \
                regression.train_cnn(splits, verbose=verbose)
        except ImportError:
            print("TensorFlow not available - skipping CNN.")

    stack, metrics["Stacking Ensemble"], predictions["Stacking Ensemble"] = \
        regression.train_stacking(splits, models, verbose=verbose)

    # ------------------------------------------------------------------ #
    # 4. Comparison table
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 80)
    print("STEP 4  Model comparison")
    print("=" * 80)
    comparison = pd.DataFrame([
        {
            "Model": name,
            "Train_R2": m["train_r2"], "Val_R2": m["val_r2"], "Test_R2": m["test_r2"],
            "Train_RMSE": m["train_rmse"], "Val_RMSE": m["val_rmse"],
            "Test_RMSE": m["test_rmse"], "Test_MAE": m["test_mae"],
        }
        for name, m in metrics.items()
    ])
    comparison["Overfit_R2"] = comparison["Train_R2"] - comparison["Test_R2"]
    print(comparison.to_string(index=False))
    comparison.to_csv(config.TABLE_DIR / "regression_model_comparison.csv", index=False)

    # ------------------------------------------------------------------ #
    # 5. Interpretation
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 80)
    print("STEP 5  SHAP and biological association analysis")
    print("=" * 80)

    xgb_importance = pd.DataFrame({
        "feature": X_test.columns,
        "importance": xgb.feature_importances_,
    }).sort_values("importance", ascending=False)
    xgb_importance.to_csv(
        config.TABLE_DIR / "regression_feature_importance_xgboost.csv", index=False
    )

    try:
        shap_imp, _, _ = interpretation.shap_importance(xgb, X_test)
        shap_imp.to_csv(
            config.TABLE_DIR / "regression_feature_importance_shap.csv", index=False
        )
        print("\nTop 10 features by mean |SHAP|:")
        print(shap_imp.head(10).to_string(index=False))
    except Exception as exc:  # SHAP is optional
        print(f"SHAP analysis skipped: {exc}")

    meth_in_model = [f for f in methylation_features if f in X_test.columns]
    metab_in_model = [f for f in metabolomics_features if f in X_test.columns]
    top_meth = xgb_importance[
        xgb_importance["feature"].isin(meth_in_model)
    ].head(20)["feature"].tolist()
    top_metab = xgb_importance[
        xgb_importance["feature"].isin(metab_in_model)
    ].head(20)["feature"].tolist()

    gene_assoc = interpretation.gene_drug_associations(
        data, top_meth, target=config.REGRESSION_TARGET
    )
    if len(gene_assoc):
        gene_assoc.to_csv(
            config.TABLE_DIR / "gene_drug_associations.csv", index=False
        )
        print(f"\nSignificant genes (FDR): "
              f"{int(gene_assoc.get('significant', pd.Series(dtype=bool)).sum())}")

    metab_assoc = interpretation.metabolite_drug_associations(
        data, top_metab, target=config.REGRESSION_TARGET
    )
    if len(metab_assoc):
        metab_assoc.to_csv(
            config.TABLE_DIR / "metabolite_drug_associations.csv", index=False
        )

    # Persist mutual-information ranking for the record.
    fs_diagnostics["mutual_info"].to_csv(
        config.TABLE_DIR / "regression_feature_importance_mutual_info.csv", index=False
    )

    print("\n" + "=" * 80)
    print(f"Done. Result tables written to: {config.TABLE_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-cnn", action="store_true",
                        help="Skip the CNN model (avoids the TensorFlow dependency).")
    parser.add_argument("--quiet", action="store_true",
                        help="Reduce console output.")
    args = parser.parse_args()
    main(use_cnn=not args.no_cnn, verbose=not args.quiet)
