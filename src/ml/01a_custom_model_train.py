"""Script to train custom models.

Author: Claudia Herron Mulet (clherrom12@alumnes.ub.edu)
Date: 01/03/2023
"""
import logging
import multiprocessing
import os
import warnings

from numba import NumbaDeprecationWarning

warnings.filterwarnings("ignore", category=NumbaDeprecationWarning)
import joblib
from catboost import CatBoostClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.over_sampling import SMOTENC, RandomOverSampler
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from ml_models import tree_models
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier

from src.config import paths
from src.data.loading import getDisease, getMetadataWithCategories, getProcessedFields
from src.data.utils import getFeaturesByDatatype, getSamePatients
from src.ml.utils import (
    generateFilenameCustomModel,
    getMLConfiguration,
    getPreprocessingPipelines,
)

if __name__ == "__main__":
    # 1. Configure model parameters
    logging.basicConfig(
        filename=os.path.join(paths.LOGS_DIR, "custom_train.log"),
        filemode="w",
        encoding="utf-8",
        format="%(asctime)s %(levelname)s: %(message)s",
        level=logging.INFO,
    )
    logging.info("Starting Custom training script")
    config = getMLConfiguration()
    seed = config["SEED"]
    icd10 = config["ICD10"]
    model_name = config["CURRENT_MODEL_NAME"]

    # 2. Load data
    logging.info("Load data")
    X = getProcessedFields()
    y = getDisease(icd10)
    X, y = getSamePatients(X, y)
    metadata = getMetadataWithCategories()
    cat_features, num_features, num_features_special, cat_idx = getFeaturesByDatatype(
        X, cat_idx=True
    )
    nfeatures = X.shape[1]
    # get imbalance ratio
    logging.info(f"Imbalance ratio: {sum(y) / len(y):.4f}")

    # 3. Divide in train, test and validation sets
    logging.info("Split train and test")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, random_state=seed
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_test, y_test, stratify=y_test, random_state=seed, test_size=2 / 3
    )

    # 4. Prepare model
    logging.info("Prepare Pipeline")
    is_tree_model = model_name in tree_models
    preprocessor = getPreprocessingPipelines(
        cat_features, num_features, num_features_special, tree=is_tree_model
    )

    if model_name == "LogisticRegression":
        model = LogisticRegression(
            max_iter=1000, n_jobs=-1, random_state=seed, class_weight="balanced"
        )
    elif model_name == "RandomForestClassifier":
        model = RandomForestClassifier(
            n_jobs=multiprocessing.cpu_count() - 1,
            random_state=seed,
            verbose=10,
            class_weight="balanced",
        )
    elif model_name == "SVM":
        svc = LinearSVC(
            C=0.1, dual=False, verbose=10, random_state=seed, class_weight="balanced"
        )
        model = CalibratedClassifierCV(svc, method="sigmoid", n_jobs=-1)
    elif model_name == "XGBClassifier":
        model = XGBClassifier(
            n_jobs=multiprocessing.cpu_count() - 1,
            random_state=seed,
            class_weight="balanced",
        )
    elif model_name == "CatBoostClassifier":
        model = CatBoostClassifier(
            train_dir=paths.REPORTS_CATBOOST,
            silent=True,
            random_state=seed,
        )
    elif model_name == "MLPClassifier":
        model = MLPClassifier(random_state=seed, verbose=True)
    elif model_name == "BalancedRandomForestClassifier":
        model = BalancedRandomForestClassifier(random_state=seed, verbose=10)
    else:
        logging.error("Invalid model name")
        exit(1)

    if config["RESAMPLER"] == "None":
        clf_pipeline = Pipeline([("preprocessor", preprocessor), ("classifier", model)])
    else:
        if config["RESAMPLER"] == "RUS":
            clf_pipeline = Pipeline(
                [
                    ("preprocessor", preprocessor),
                    ("unders", RandomUnderSampler(random_state=seed)),
                    ("classifier", model),
                ]
            )
            model_name = model_name + "_" + "RUS"
        elif config["RESAMPLER"] == "ROS":
            clf_pipeline = Pipeline(
                [
                    ("preprocessor", preprocessor),
                    ("overs", RandomOverSampler(random_state=seed)),
                    ("classifier", model),
                ]
            )
            model_name = model_name + "_" + "ROS"
        elif config["RESAMPLER"] == "RUSROS":
            clf_pipeline = Pipeline(
                [
                    ("preprocessor", preprocessor),
                    (
                        "unders",
                        RandomUnderSampler(random_state=seed, sampling_strategy=1 / 2),
                    ),
                    (
                        "overs",
                        RandomOverSampler(random_state=seed, sampling_strategy=1),
                    ),
                    ("classifier", model),
                ]
            )
            model_name = model_name + "_" + "RUSROS"
        elif config["RESAMPLER"] == "RUSSMOTENC":
            clf_pipeline = Pipeline(
                [
                    ("preprocessor", preprocessor),
                    (
                        "unders",
                        RandomUnderSampler(random_state=seed, sampling_strategy=1 / 2),
                    ),
                    (
                        "overs",
                        SMOTENC(
                            categorical_features=cat_idx,
                            random_state=seed,
                            sampling_strategy=1,
                        ),
                    ),
                    ("classifier", model),
                ]
            )
            model_name = model_name + "_" + "RUSSMOTENC"

    # 5. Fit model
    logging.info(f"Fit pipeline with {model_name}")
    clf_pipeline.fit(X_train, y_train)

    logging.info("Save grid search")
    filename = generateFilenameCustomModel(icd10, nfeatures, model_name, seed)
    joblib.dump(clf_pipeline, os.path.join(paths.MODELS_DIR, filename))
