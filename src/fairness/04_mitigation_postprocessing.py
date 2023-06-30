"""Script to train model for bias post-processing techniques.

Author: Claudia Herron Mulet (clherrom12@alumnes.ub.edu)
Date: 01/03/2023
"""
import logging
import os

import joblib
import numpy as np
import pandas as pd
from aif360.sklearn.postprocessing import (
    CalibratedEqualizedOdds,
    PostProcessingMeta,
    RejectOptionClassifier,
    RejectOptionClassifierCV,
)
from sklearn.model_selection import train_test_split

from src.config import paths
from src.data.loading import getDisease, getMetadataWithCategories, getProcessedFields
from src.data.utils import getFeaturesByDatatype, getSamePatients
from src.fairness.utils import (
    SeparationPostprocessor,
    createBinaryFairnessFeatures,
    getEvalMetrics,
    getFPR,
)
from src.ml.utils import findOptimalThreshold, getMLConfiguration
from src.visualization.fairness import generatePalette

if __name__ == "__main__":
    # 1. Configuration
    logging.basicConfig(
        filename=os.path.join(paths.LOGS_DIR, "fairness_postprocessing.log"),
        filemode="w",
        encoding="utf-8",
        format="%(asctime)s %(levelname)s: %(message)s",
        level=logging.INFO,
    )
    logging.info("Starting Catboost training script")
    config = getMLConfiguration()
    seed = config["SEED"]
    icd10 = config["ICD10"]
    opt_score = config["OPT_SCORE"]
    model_name = config["CURRENT_MODEL_NAME"]
    model_filename = config["CURRENT_MODEL"]
    metrics = []
    model_bias_report_filename = os.path.join(
        paths.REPORTS_FAIRNESS, f"{icd10}_model_bias_postproc.csv"
    )
    PA_features_mitigation = ["PA_F31", "PA_F34", "PA_F21001"]

    # 2. Load data
    logging.info("Load data")
    X = getProcessedFields()
    y = getDisease(icd10)
    X, y = getSamePatients(X, y)
    metadata = getMetadataWithCategories()
    nfeatures = X.shape[1]
    cat_features, num_features, num_features_special = getFeaturesByDatatype(
        X, cat_idx=False
    )
    # 3. Create fairness multiindex
    X = createBinaryFairnessFeatures(X)
    PA_features = [col for col in X.columns if col.startswith("PA")]
    PA_names = [metadata.loc[col.split("_")[1]].Name for col in PA_features]
    palette = generatePalette(PA_names)
    multi_index = pd.MultiIndex.from_frame(
        X[PA_features].reset_index(), names=["patient"] + PA_features
    )
    X.set_index(multi_index, inplace=True)
    y.index = multi_index

    # 4. Split data
    logging.info("Split train and test")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, random_state=seed
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_test, y_test, stratify=y_test, random_state=seed, test_size=2 / 3
    )

    # 5. Load model
    logging.info("Load model")
    model = joblib.load(os.path.join(paths.MODELS_DIR, model_filename))
    positive_class = np.where(model.classes_ == 1)[0][0]

    # 6. Compute probabilities
    y_probs_train = model.predict_proba(X_train)[:, positive_class]
    y_probs_val = model.predict_proba(X_val)[:, positive_class]
    y_probs_test = model.predict_proba(X_test)[:, positive_class]

    # 7. Baseline
    th, max_score = findOptimalThreshold(y_probs_val, y_val, opt_score)
    # apply threshold to get labels
    y_pred_train = y_probs_train > th
    y_pred_val = y_probs_val > th
    y_pred_test = y_probs_test > th
    metrics.append(getEvalMetrics(y_pred_val, y_val, "Catboost Baseline"))

    # 8. Postprocessing methods
    # SeparationPostprocessor for PA_F21001
    fpr_goal = getFPR(y_val, y_pred_val)
    sp = SeparationPostprocessor(model, prot_attr="PA_F21001")
    sp.fit(X_val, y_val, desired_fpr_per_group=fpr_goal)
    y_pred_fair = sp.predict(X_val)
    metrics.append(getEvalMetrics(y_pred_fair, y_val, "Separation Postprocessor BMI"))

    # SeparationPostprocessor for PA_F34
    sp = SeparationPostprocessor(model, prot_attr="PA_F34")
    sp.fit(X_val, y_val, desired_fpr_per_group=fpr_goal)
    y_pred_fair = sp.predict(X_val)
    metrics.append(
        getEvalMetrics(y_pred_fair, y_val, "Separation Postprocessor Year " "of Birth")
    )
    # RejectOptionClassifier for PA_F31
    roc = RejectOptionClassifier(prot_attr="PA_F31", threshold=th)
    roc.fit(X_val, y_val)
    y_pred_fair = roc.predict(
        pd.DataFrame(model.predict_proba(X_val), index=y_val.index)
    )
    metrics.append(
        getEvalMetrics(y_pred_fair, y_val, "RejectOptionClassifier F1-threshold Sex")
    )

    # RejectOptionClassifier for PA_F21001
    roc = RejectOptionClassifier(prot_attr="PA_F21001", threshold=th)
    roc.fit(X_val, y_val)
    y_pred_fair = roc.predict(
        pd.DataFrame(model.predict_proba(X_val), index=y_val.index)
    )
    metrics.append(
        getEvalMetrics(y_pred_fair, y_val, "RejectOptionClassifier F1-threshold BMI")
    )

    # RejectOptionClassifier for PA_F34
    roc = RejectOptionClassifier(prot_attr="PA_F34", threshold=th)
    roc.fit(X_val, y_val)
    y_pred_fair = roc.predict(
        pd.DataFrame(model.predict_proba(X_val), index=y_val.index)
    )
    metrics.append(
        getEvalMetrics(
            y_pred_fair, y_val, "RejectOptionClassifier F1-threshold Year of birth"
        )
    )

    # RejectOptionClassifier for PA_F31 with threshold search
    metric = "equal_opportunity"
    roc_cb = PostProcessingMeta(
        model,
        RejectOptionClassifierCV(
            "PA_F31", scoring=metric, n_jobs=-1, error_score="raise"
        ),
        prefit=True,
    )
    roc_cb.fit(X_val, y_val)
    y_pred_fair = roc_cb.predict(X_val)
    metrics.append(getEvalMetrics(y_pred_fair, y_val, "RejectOptionClassifier EO Sex"))

    # RejectOptionClassifier for PA_F34 with threshold search
    metric = "average_odds"
    roc_cb = PostProcessingMeta(
        model,
        RejectOptionClassifierCV(
            "PA_F34", scoring=metric, n_jobs=-1, error_score="raise"
        ),
        prefit=True,
    )
    roc_cb.fit(X_val, y_val)
    y_pred_fair = roc_cb.predict(X_val)
    metrics.append(
        getEvalMetrics(y_pred_fair, y_val, "RejectOptionClassifier AO Year of birth")
    )

    # RejectOptionClassifier for PA_F21001 with threshold search
    metric = "average_odds"
    roc_cb = PostProcessingMeta(
        model,
        RejectOptionClassifierCV(
            "PA_F21001", scoring=metric, n_jobs=-1, error_score="raise"
        ),
        prefit=True,
    )
    roc_cb.fit(X_val, y_val)
    y_pred_fair = roc_cb.predict(X_val)
    metrics.append(getEvalMetrics(y_pred_fair, y_val, "RejectOptionClassifier AO BMI"))

    # CalibratedEqualizedOdds for PA_F31
    ceo_cb = PostProcessingMeta(
        model,
        CalibratedEqualizedOdds("PA_F31"),
        prefit=True,
    )
    ceo_cb.fit(X_val, y_val)
    y_pred_fair = ceo_cb.predict(X_val)
    metrics.append(getEvalMetrics(y_pred_fair, y_val, "CalibratedEqualizedOdds Sex"))

    # CalibratedEqualizedOdds for PA_F21001
    ceo_cb = PostProcessingMeta(
        model,
        CalibratedEqualizedOdds("PA_F21001"),
        prefit=True,
    )
    ceo_cb.fit(X_val, y_val)
    y_pred_fair = ceo_cb.predict(X_val)
    metrics.append(getEvalMetrics(y_pred_fair, y_val, "CalibratedEqualizedOdds BMI"))

    # CalibratedEqualizedOdds for PA_F34
    ceo_cb = PostProcessingMeta(
        model,
        CalibratedEqualizedOdds("PA_F34"),
        prefit=True,
    )
    ceo_cb.fit(X_val, y_val)
    y_pred_fair = ceo_cb.predict(X_val)
    metrics.append(
        getEvalMetrics(y_pred_fair, y_val, "CalibratedEqualizedOdds Year of birth")
    )

    print(metrics)
    pd.DataFrame(metrics, index=None).round(3).astype(str).replace(
        "\.", ",", regex=True
    ).to_csv(model_bias_report_filename, index=False)
