"""Script to train catboost for \
disease classification.

Author: Claudia Herron Mulet (clherrom12@alumnes.ub.edu)
Date: 01/03/2023
"""
import logging
import os

import numpy as np
import pandas as pd
from aif360.sklearn.inprocessing import (
    ExponentiatedGradientReduction,
    GridSearchReduction,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from src.config import paths
from src.data.loading import getDisease, getMetadataWithCategories, getProcessedFields
from src.data.utils import getFeaturesByDatatype, getSamePatients
from src.fairness.utils import createBinaryFairnessFeatures, getEvalMetrics
from src.ml.ml_models import tree_models
from src.ml.utils import (
    findOptimalThreshold,
    getMLConfiguration,
    getPreprocessingPipelines,
)

if __name__ == "__main__":
    # 1. Configuration
    logging.basicConfig(
        filename=os.path.join(paths.LOGS_DIR, "fairness_inprocessing.log"),
        filemode="w",
        encoding="utf-8",
        format="%(asctime)s %(levelname)s: %(message)s",
        level=logging.INFO,
    )
    logging.info("Starting Inprocessing training script")
    config = getMLConfiguration()
    seed = config["SEED"]
    icd10 = config["ICD10"]
    opt_score = config["OPT_SCORE"]
    model_name = config["CURRENT_MODEL_NAME"]
    model_filename = config["CURRENT_MODEL"]
    metrics = []
    model_bias_report_filename = os.path.join(
        paths.REPORTS_FAIRNESS, f"{icd10}_model_bias_inproc.csv"
    )
    N = 30000

    # 2. Load data
    logging.info("Load data")
    X = getProcessedFields()
    y = getDisease(icd10)
    metadata = getMetadataWithCategories()
    X, y = getSamePatients(X, y)
    nfeatures = X.shape[1]
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
    multi_index = pd.MultiIndex.from_frame(
        X[PA_features].reset_index(), names=["patient"] + PA_features
    )
    X.set_index(multi_index, inplace=True)
    y.index = multi_index

    # 4. Split data
    logging.info("Split train, test and val")
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, stratify=y, random_state=seed
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_test, y_test, stratify=y_test, random_state=seed, test_size=2 / 3
    )

    # 5. Resample training
    logging.info("Resample training")
    X_train, X_disc, y_train, y_disc = train_test_split(
        X_temp, y_temp, stratify=y_temp, random_state=seed, train_size=N
    )

    # 6. Prepare baseline model
    logging.info("Prepare Pipeline")
    logging.info("Load model")
    prep = getPreprocessingPipelines(
        cat_features,
        num_features,
        num_features_special,
        tree=model_name in tree_models,
    )
    prep.fit(X_train.drop(columns=PA_features))
    X_train_baseline = pd.DataFrame(
        prep.transform(X_train.drop(columns=PA_features)),
        columns=prep.get_feature_names_out(),
        index=X_train.index,
    )
    X_val_baseline = pd.DataFrame(
        prep.transform(X_val.drop(columns=PA_features)),
        columns=prep.get_feature_names_out(),
        index=X_val.index,
    )
    X_test_baseline = pd.DataFrame(
        prep.transform(X_test.drop(columns=PA_features)),
        columns=prep.get_feature_names_out(),
        index=X_test.index,
    )
    model = LogisticRegression(solver="liblinear", random_state=seed)
    model.fit(X_train_baseline, y_train)
    positive_class = np.where(model.classes_ == 1)[0][0]
    # predictions
    y_probs_train = model.predict_proba(X_train_baseline)[:, positive_class]
    y_probs_val = model.predict_proba(X_val_baseline)[:, positive_class]
    y_probs_test = model.predict_proba(X_test_baseline)[:, positive_class]
    th, max_score = findOptimalThreshold(y_probs_val, y_val, opt_score)
    y_pred_train = y_probs_train > th
    y_pred_val = y_probs_val > th
    y_pred_test = y_probs_test > th

    metrics.append(getEvalMetrics(y_pred_val, y_val, "LogisticRegression Baseline"))

    # 7. Inprocessing models
    # apply preprocessing, but conserve fairness features
    X_train = pd.DataFrame(
        prep.fit_transform(X_train),
        columns=prep.get_feature_names_out(),
        index=X_train.index,
    )
    X_val = pd.DataFrame(
        prep.transform(X_val), columns=prep.get_feature_names_out(), index=X_val.index
    )
    X_test = pd.DataFrame(
        prep.transform(X_test), columns=prep.get_feature_names_out(), index=X_test.index
    )
    estimator = LogisticRegression(solver="liblinear", random_state=seed)

    logging.info("ExponentiatedGradientReduction FalsePositiveRateParity")
    # option 1: ExponentiatedGradientReduction FalsePositiveRateParity
    exp_grad_red = ExponentiatedGradientReduction(
        prot_attr=PA_features,
        estimator=estimator,
        constraints="FalsePositiveRateParity",
        drop_prot_attr=True,
    )
    exp_grad_red.fit(X_train, y_train)
    y_pred_fair = exp_grad_red.predict(X_val)
    metrics.append(
        getEvalMetrics(
            y_pred_fair,
            y_val,
            "ExponentiatedGradientReduction FalsePositiveRateParity",
        )
    )

    logging.info("ExponentiatedGradientReduction TruePositiveRateParity")
    # option 2: ExponentiatedGradientReduction TruePositiveRateParity
    exp_grad_red = ExponentiatedGradientReduction(
        prot_attr=PA_features,
        estimator=estimator,
        constraints="TruePositiveRateParity",
        drop_prot_attr=True,
    )
    exp_grad_red.fit(X_train, y_train)
    y_pred_fair = exp_grad_red.predict(X_val)
    metrics.append(
        getEvalMetrics(
            y_pred_fair, y_val, "ExponentiatedGradientReduction TruePositiveRateParity"
        )
    )

    logging.info("GridSearchReduction with TruePositiveRateParity")
    # option 3: GridSearchReduction with TruePositiveRateParity
    cv_red = GridSearchReduction(
        prot_attr=PA_features,
        estimator=estimator,
        constraints="TruePositiveRateParity",
        drop_prot_attr=True,
    )
    cv_red.fit(X_train, y_train)
    y_pred_fair = cv_red.predict(X_val)
    metrics.append(
        getEvalMetrics(y_pred_fair, y_val, "GridSearchReduction TruePositiveRateParity")
    )

    logging.info("GridSearchReduction with TruePositiveRateParity")
    # option 4: GridSearchReduction with TruePositiveRateParity
    cv_red_red = GridSearchReduction(
        prot_attr=PA_features,
        estimator=estimator,
        constraints="FalsePositiveRateParity",
        drop_prot_attr=True,
    )
    cv_red_red.fit(X_train, y_train)
    y_pred_fair = cv_red_red.predict(X_val)
    metrics.append(
        getEvalMetrics(
            y_pred_fair, y_val, "GridSearchReduction FalsePositiveRateParity"
        )
    )

    # 8. Format results
    logging.info("Format results")
    print(metrics)
    pd.DataFrame(metrics, index=None).round(3).astype(str).replace(
        "\.", ",", regex=True
    ).to_csv(model_bias_report_filename, index=False)
    pd.DataFrame(metrics, index=None).round(3).astype(str).replace(
        "\.", ",", regex=True
    ).to_clipboard(index=False)
