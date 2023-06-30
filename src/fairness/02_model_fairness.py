"""Script to identify biases in training data.

Author: Claudia Herron Mulet (clherrom12@alumnes.ub.edu)
Date: 01/03/2023
"""
import logging
import os

import joblib
import numpy as np
import pandas as pd
from aif360.sklearn.metrics import (
    average_odds_error,
    disparate_impact_ratio,
    equal_opportunity_difference,
)
from sklearn.model_selection import train_test_split

from src.config import paths
from src.data.loading import getDisease, getMetadataWithCategories, getProcessedFields
from src.data.utils import getFeaturesByDatatype, getSamePatients
from src.fairness.utils import FPR_difference, createBinaryFairnessFeatures
from src.ml.utils import findOptimalThreshold, getMLConfiguration
from src.visualization.fairness import disagregatedROC, generatePalette, plotBars

if __name__ == "__main__":
    # 1. Configuration
    logging.info("Starting model bias evaluation script")
    logging.basicConfig(
        filename=os.path.join(paths.LOGS_DIR, "model_fairness.log"),
        filemode="w",
        encoding="utf-8",
        format="%(asctime)s %(levelname)s: %(message)s",
        level=logging.INFO,
    )
    logging.info("Starting data bias script")
    config = getMLConfiguration()
    seed = config["SEED"]
    icd10 = config["ICD10"]
    opt_score = config["OPT_SCORE"]
    model_filename = config["CURRENT_MODEL"]
    model_name = config["CURRENT_MODEL_NAME"]
    model_bias_report_filename = os.path.join(
        paths.REPORTS_FAIRNESS, f"{icd10}_model_bias.csv"
    )

    # 2. Load data
    metadata = getMetadataWithCategories()
    X = getProcessedFields()
    y = getDisease(icd10)
    X, y = getSamePatients(X, y)
    cat_features, num_features, num_features_special = getFeaturesByDatatype(
        X, cat_idx=False
    )

    # 3. Create fairness multiindex
    X = createBinaryFairnessFeatures(X)
    PA_features = [col for col in X.columns if col.startswith("PA")]
    PA_names = [
        metadata.loc[col.split("_")[1]].Name
        if metadata.loc[col.split("_")[1]].Name
        != "Townsend deprivation index at recruitment"
        else "TDI"
        for col in PA_features
    ]
    palette = generatePalette(PA_names)
    multi_index = pd.MultiIndex.from_frame(
        X[PA_features].reset_index(), names=["patient"] + PA_features
    )
    X.set_index(multi_index, inplace=True)
    X.drop(columns=PA_features, inplace=True)
    y.index = multi_index

    # 3. Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, random_state=seed
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_test, y_test, stratify=y_test, random_state=seed, test_size=2 / 3
    )

    # 3. Load model
    model = joblib.load(os.path.join(paths.MODELS_DIR, model_filename))
    if config["IS_GRID"]:
        model = model.best_estimator_
    positive_class = np.where(model.classes_ == 1)[0][0]

    # 4. Get model predictions
    logging.info("Compute predictions")
    y_probs_val = model.predict_proba(X_val)[:, positive_class]
    y_probs_train = model.predict_proba(X_train)[:, positive_class]
    y_probs_test = model.predict_proba(X_test)[:, positive_class]

    # 5. Find decision threshold and apply it
    logging.info("Find decision threshold")
    th, max_score = findOptimalThreshold(y_probs_val, y_val, opt_score)
    y_pred_train = y_probs_train > th
    y_pred_test = y_probs_test > th
    y_pred_val = y_probs_val > th

    # 6. Measure disparate impact
    logging.info("Measure Disparate Impact Ration")
    disparate_impact_dict = {}
    for feature, name in zip(PA_features, PA_names):
        di = disparate_impact_ratio(
            y_test, y_pred_test, prot_attr=feature, priv_group=1
        )
        disparate_impact_dict[name] = di
    plotBars(
        disparate_impact_dict,
        title="DIR per protected attribute in test predictions",
        ref="Equal base rates",
        filename=f"{icd10}/model_fairness_disparate_impact_test_pred.png",
        palette=palette,
    )

    # 7. Measure Average odds difference
    logging.info("Measure Average Odds Difference")
    aod = {}
    for feature, name in zip(PA_features, PA_names):
        aod[name] = average_odds_error(
            y_test, y_pred_test, prot_attr=feature, priv_group=1
        )

    plotBars(
        aod,
        title="AOE per " "protected attribute in " "test predictions",
        ref=None,
        filename=f"{icd10}/model_fairness_average_odds_error_test_pred.png",
        palette=palette,
        hideticks=True,
    )

    # 8. Equal opportunity difference (difference in TPR)
    logging.info("Measure Equal Opportunity Difference")
    eod = {}
    for feature, name in zip(PA_features, PA_names):
        eod[name] = equal_opportunity_difference(
            y_test, y_pred_test, prot_attr=feature, priv_group=1
        )
    plotBars(
        eod,
        title="EOD per " "protected attribute in " "test predictions",
        ref=None,
        filename=f"{icd10}/model_fairness_equal_opportunity_difference_test_pred.png",
        palette=palette,
        hideticks=True,
    )

    # 9. FPR difference
    logging.info("Measure FPR Difference")
    fprd = {}
    for feature, name in zip(PA_features, PA_names):
        fprd[name] = FPR_difference(
            y_test, y_pred_test, prot_attr=feature, priv_group=1
        )
    plotBars(
        fprd,
        title="FPR difference per " "protected attribute in " "test predictions",
        ref=None,
        filename=f"{icd10}/model_fairness_fpr_difference_test_pred.png",
        palette=palette,
        hideticks=False,
    )

    # 10, Disagregated ROC
    logging.info("Disagregated ROC")
    for feature, name in zip(PA_features, PA_names):
        disagregatedROC(
            y_test,
            y_probs_test,
            name,
            filename=f"{icd10}/model_fairness_rocauc_curve_Test_{feature}.png",
            prot_attr=feature,
            priv_group=1,
            dataset="Test",
        )
