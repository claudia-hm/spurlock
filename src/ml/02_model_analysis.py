"""Script to analyse model predictions.

Author: Claudia Herron Mulet (clherrom12@alumnes.ub.edu)
Date: 27/03/2023
"""
import logging
import os
import warnings
from datetime import datetime

import joblib
from numba import NumbaDeprecationWarning

warnings.filterwarnings("ignore", category=NumbaDeprecationWarning)
import numpy as np
from imblearn.metrics import classification_report_imbalanced
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from src.config import paths
from src.data.loading import getDisease, getMetadataWithCategories, getProcessedFields
from src.data.utils import getSamePatients
from src.ml.utils import (
    calculatePrecision,
    calculateSensitivity,
    calculateSpecificity,
    findOptimalThreshold,
    getMLConfiguration,
    getTopFeatures,
)
from src.visualization.ml import (
    plot_confusion_matrix,
    plot_metrics_vs_threshold,
    plot_precision_recall,
    plot_roc_curve,
    plot_shap,
    plotProbs,
)

if __name__ == "__main__":
    # 1. Configure model parameters
    logging.basicConfig(
        filename=os.path.join(paths.LOGS_DIR, "model_analysis.log"),
        filemode="w",
        encoding="utf-8",
        format="%(asctime)s %(levelname)s: %(message)s",
        level=logging.INFO,
    )
    logging.info("Starting model analysis script")
    # set parameters
    config = getMLConfiguration()
    seed = config["SEED"]
    icd10 = config["ICD10"]
    opt_score = config["OPT_SCORE"]
    model_filename = config["CURRENT_MODEL"]
    model_name = config["CURRENT_MODEL_NAME"]

    # 2. Load data
    logging.info("Load data")
    X = getProcessedFields()
    y = getDisease(icd10)
    X, y = getSamePatients(X, y)
    metadata = getMetadataWithCategories()
    nfeatures = X.shape[1]
    # get unbalanced ratio
    unbalanced_ratio = y.sum() / len(y)
    logging.info(f"The unbalance rate is {unbalanced_ratio * 100:.4f}%")

    # 3. Load model
    logging.info("Load model")
    model = joblib.load(os.path.join(paths.MODELS_DIR, model_filename))
    if config["IS_GRID"]:
        model = model.best_estimator_
    positive_class = np.where(model.classes_ == 1)[0][0]

    # 4. Train, test, validation split
    logging.info("Train test split")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, random_state=seed
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_test, y_test, stratify=y_test, random_state=seed, test_size=2 / 3
    )

    # 5. Get model predictions
    logging.info("Compute predictions")
    y_probs_val = model.predict_proba(X_val)[:, positive_class]
    y_probs_train = model.predict_proba(X_train)[:, positive_class]
    y_probs_test = model.predict_proba(X_test)[:, positive_class]

    # 6. Find decision threshold and apply it
    th, max_score = findOptimalThreshold(y_probs_val, y_val, opt_score)
    y_pred_train = y_probs_train > th
    y_pred_test = y_probs_test > th
    y_pred_val = y_probs_val > th

    # 7. Report results
    logging.info("Report results")
    feature_names = model.named_steps.preprocessor.get_feature_names_out()
    top10_features = getTopFeatures(model.named_steps.classifier, feature_names, k=10)
    with open(
        os.path.join(
            paths.REPORTS_ML_DIR,
            f"model_analysis_report_" f"{datetime.now()}_{model_name}.txt",
        ),
        "w",
    ) as f:
        f.write(f"ICD10: {icd10}\n")
        f.write(f"Optimal threshold for {opt_score}: {th}\n")
        f.write("\n========TRAIN========\n")
        print(classification_report_imbalanced(y_train.values, y_pred_train), file=f)
        roc_auc = roc_auc_score(y_train.values, y_probs_train)
        f1 = f1_score(y_train.values, y_pred_train)
        balanced_accuracy = balanced_accuracy_score(y_train.values, y_pred_train)
        mc = matthews_corrcoef(y_train.values, y_pred_train)
        sensitivity = calculateSensitivity(y_train.values, y_pred_train)
        specificity = calculateSpecificity(y_train.values, y_pred_train)
        precision = calculatePrecision(y_train.values, y_pred_train)
        f.write(f"ROC AUC score: {roc_auc:.4f}\n")
        f.write(f"F1 score: {f1:.4f}\n")
        f.write(f"Balanced accuracy: {balanced_accuracy:.4f}\n")
        f.write(f"Matthews correlation: {mc:.4f}\n")
        f.write(f"Sensitivity/Recall: {sensitivity:.4f}\n")
        f.write(f"Specificity: {specificity:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")

        f.write("\n========VAL========\n")
        print(classification_report_imbalanced(y_val.values, y_pred_val), file=f)
        roc_auc = roc_auc_score(y_val.values, y_probs_val)
        f1 = f1_score(y_val.values, y_pred_val)
        balanced_accuracy = balanced_accuracy_score(y_val.values, y_pred_val)
        mc = matthews_corrcoef(y_val.values, y_pred_val)
        sensitivity = calculateSensitivity(y_val.values, y_pred_val)
        specificity = calculateSpecificity(y_val.values, y_pred_val)
        precision = calculatePrecision(y_val.values, y_pred_val)
        f.write(f"ROC AUC score: {roc_auc:.4f}\n")
        f.write(f"F1 score: {f1:.4f}\n")
        f.write(f"Balanced accuracy: {balanced_accuracy:.4f}\n")
        f.write(f"Matthews correlation: {mc:.4f}\n")
        f.write(f"Sensitivity/Recall: {sensitivity:.4f}\n")
        f.write(f"Specificity: {specificity:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")

        f.write("\n========TEST========\n")
        print(classification_report_imbalanced(y_test.values, y_pred_test), file=f)
        roc_auc = roc_auc_score(y_test.values, y_probs_test)
        f1 = f1_score(y_test.values, y_pred_test)
        balanced_accuracy = balanced_accuracy_score(y_test.values, y_pred_test)
        mc = matthews_corrcoef(y_test.values, y_pred_test)
        sensitivity = calculateSensitivity(y_test.values, y_pred_test)
        specificity = calculateSpecificity(y_test.values, y_pred_test)
        precision = calculatePrecision(y_test.values, y_pred_test)
        f.write(f"ROC AUC score: {roc_auc:.4f}\n")
        f.write(f"F1 score: {f1:.4f}\n")
        f.write(f"Balanced accuracy: {balanced_accuracy:.4f}\n")
        f.write(f"Matthews correlation: {mc:.4f}\n")
        f.write(f"Sensitivity/Recall: {sensitivity:.4f}\n")
        f.write(f"Specificity: {specificity:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")

        f.write("\nTop-10 most important features:\n")
        for i, feature in enumerate(top10_features):
            if "_" in feature:
                feature = feature.split("_")[0]
            feature_name = metadata.loc[feature, "Name"]
            feature_category = metadata.loc[feature, "cat_name"]
            if len(feature.split("_")) > 1:
                category = feature.split("_")[1]
                f.write(
                    f"\t {i}. {feature}-{feature_name} ({category}). "
                    f"{feature_category}. \n"
                )
            else:
                f.write(f"\t {i}. {feature}-{feature_name}. {feature_category}\n")

    # 8. Figures
    logging.info("Plotting")
    logging.info("Explainability")
    try:
        plot_shap(model, X_train, "Train", metadata, icd10=icd10)
        plot_shap(model, X_test, "Test", metadata, icd10=icd10)

    except KeyError as e:
        print(
            "Unable to do shap plots because metadata does not contain one hot "
            "encoded features"
        )
        print(e)
    logging.info("Confusion matrices")
    plot_confusion_matrix(
        y_train,
        y_pred_train,
        [True, False],
        dataset="Train",
        model=model_name,
        icd10=icd10,
    )
    plot_confusion_matrix(
        y_test,
        y_pred_test,
        [True, False],
        dataset="Test",
        model=model_name,
        icd10=icd10,
    )
    logging.info("Metrics vs threshold")
    plot_metrics_vs_threshold(
        y_train, y_probs_train, th=th, dataset="Train", model=model_name, icd10=icd10
    )
    plot_metrics_vs_threshold(
        y_test, y_probs_test, th=th, dataset="Test", model=model_name, icd10=icd10
    )

    logging.info("Precision recall curves")
    plot_precision_recall(
        y_train.values, y_probs_train, dataset="Train", model=model_name, icd10=icd10
    )
    plot_precision_recall(
        y_test.values, y_probs_test, dataset="Test", model=model_name, icd10=icd10
    )

    logging.info("ROC curves")
    plot_roc_curve(
        y_train.values, y_probs_train, dataset="Train", model=model_name, icd10=icd10
    )
    plot_roc_curve(
        y_test.values, y_probs_test, dataset="Test", model=model_name, icd10=icd10
    )

    logging.info("Probability distributions")
    plotProbs(
        y_probs_train[y_train.values == 1],
        y_probs_train[y_train.values == 0],
        dataset="Train",
        model=model_name,
        icd10=icd10,
    )
    plotProbs(
        y_probs_test[y_test.values == 1],
        y_probs_test[y_test.values == 0],
        dataset="Test",
        model=model_name,
        icd10=icd10,
    )
