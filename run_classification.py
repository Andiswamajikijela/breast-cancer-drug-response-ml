#!/usr/bin/env python
"""
End-to-end classification pipeline: predict whether a breast-cancer cell line
is *Sensitive* or *Resistant* to a drug, from integrated DNA-methylation and
metabolomics features.

The binary label is derived from the LN_IC50 Z-score (Sensitive: z < -0.5,
Resistant: z > 0.5; the intermediate band is excluded). The same feature
selection used for the regression task is reused so the two analyses are
directly comparable.

Steps
-----
1. Build the integrated, imputed breast-cancer dataset (shared front-end).
2. Reuse the regression feature selection to define the feature set.
3. Construct the binary Sensitive/Resistant target and scale features.
4. Stratify-split into train / validation / test.
5. Train and evaluate six classifiers (logistic regression, RF, XGBoost,
   LightGBM, CatBoost, CNN) and compare them.
6. Write the comparison table and per-sample predictions to the output dir.

Usage
-----
    python run_classification.py [--no-cnn] [--quiet]
"""

import argparse

import numpy as np
import pandas as pd

import config
from src import data_pipeline, features, classification


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
    # 2. Feature selection (reused from the regression task)
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 80)
    print("STEP 2  Feature selection")
    print("=" * 80)
    X, y_reg, _ = features.build_feature_matrix(
        data, methylation_features, metabolomics_features
    )
    selected, _ = features.select_features(X, y_reg, verbose=verbose)

    # ------------------------------------------------------------------ #
    # 3. Binary target + feature scaling
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 80)
    print("STEP 3  Building binary target and scaling features")
    print("=" * 80)
    classification_data = classification.build_classification_target(
        data, verbose=verbose
    )
    X_scaled, y = classification.prepare_classification_features(
        classification_data, selected, verbose=verbose
    )

    # ------------------------------------------------------------------ #
    # 4. Split
    # ------------------------------------------------------------------ #
    splits = classification.split_classification(X_scaled, y, verbose=verbose)
    _, _, X_test, _, _, y_test = splits

    # ------------------------------------------------------------------ #
    # 5. Train classifiers
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 80)
    print("STEP 4  Training classifiers")
    print("=" * 80)

    results = {}
    _, results["Logistic Regression"] = \
        classification.train_logistic_regression(splits, verbose=verbose)
    _, results["Random Forest"] = \
        classification.train_random_forest_clf(splits, verbose=verbose)
    _, results["XGBoost"] = \
        classification.train_xgboost_clf(splits, verbose=verbose)
    _, results["LightGBM"] = \
        classification.train_lightgbm_clf(splits, verbose=verbose)
    _, results["CatBoost"] = \
        classification.train_catboost_clf(splits, verbose=verbose)

    if use_cnn:
        try:
            _, results["CNN"], _ = \
                classification.train_cnn_classifier(splits, verbose=verbose)
        except ImportError:
            print("TensorFlow not available - skipping CNN.")

    # ------------------------------------------------------------------ #
    # 6. Comparison table
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 80)
    print("STEP 5  Classifier comparison")
    print("=" * 80)
    comparison = pd.DataFrame([
        {
            "Model": name,
            "Test_Accuracy": m["test_acc"],
            "Test_Precision": m["test_precision"],
            "Test_Recall": m["test_recall"],
            "Test_F1": m["test_f1"],
            "Test_ROC_AUC": m["test_roc_auc"],
        }
        for name, m in results.items()
    ]).sort_values("Test_F1", ascending=False)
    print(comparison.to_string(index=False))
    comparison.to_csv(
        config.TABLE_DIR / "classification_model_comparison.csv", index=False
    )

    best_name = comparison.iloc[0]["Model"]
    predictions = pd.DataFrame({
        "y_true": y_test.values,
        "y_proba": results[best_name]["probabilities"],
        "y_pred": (results[best_name]["probabilities"] > 0.5).astype(int),
    }, index=y_test.index)
    predictions.to_csv(config.TABLE_DIR / "classification_predictions.csv")

    print("\n" + "=" * 80)
    print(f"Done. Best classifier: {best_name}")
    print(f"Result tables written to: {config.TABLE_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-cnn", action="store_true",
                        help="Skip the CNN model (avoids the TensorFlow dependency).")
    parser.add_argument("--quiet", action="store_true",
                        help="Reduce console output.")
    args = parser.parse_args()
    main(use_cnn=not args.no_cnn, verbose=not args.quiet)
