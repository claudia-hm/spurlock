"""Script to generate ML predictions for diseases.

Author: Claudia Herron Mulet (clherrom12@alumnes.ub.edu)
Date: 01/03/2023
"""
import logging
import os

import joblib
from imblearn.pipeline import Pipeline
from ml_models import getParamGrid, models, tree_models
from sklearn.model_selection import GridSearchCV, train_test_split
from tqdm import tqdm

from src.config import paths
from src.data.loading import getDisease, getMetadataWithCategories, getProcessedFields
from src.data.utils import getFeaturesByDatatype, getSamePatients
from src.ml.utils import (
    generateFilenameGrid,
    getMLConfiguration,
    getPreprocessingPipelines,
    getScorer,
    saveGridSearchResults,
)

if __name__ == "__main__":
    # 1. Configure model selection parameters
    logging.basicConfig(
        filename=os.path.join(paths.LOGS_DIR, "model_selection.log"),
        filemode="w",
        encoding="utf-8",
        format="%(asctime)s %(levelname)s: %(message)s",
        level=logging.INFO,
    )
    logging.info("Starting ML classification script")
    config = getMLConfiguration()
    seed = config["SEED"]
    icd10 = config["ICD10"]
    opt_score = config["OPT_SCORE"]
    LOAD_STORED_MODELS = config["LOAD_STORED_MODELS"]
    scorer = getScorer(opt_score)

    # 2. Load data
    logging.info("Load data")
    X = getProcessedFields()
    y = getDisease(icd10)
    X, y = getSamePatients(X, y)
    metadata = getMetadataWithCategories()
    nfeatures = X.shape[1]
    logging.info("Get features by datatype")
    (
        cat_features,
        num_features,
        num_features_special,
        cat_indices,
    ) = getFeaturesByDatatype(X)

    # 3. Divide in train, test and validation sets
    logging.info("Train test splitting")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, random_state=seed
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_test, y_test, stratify=y_test, random_state=seed, test_size=2 / 3
    )

    # 4. Model selection loop
    logging.info("Start model selection")
    scores, all_grid_search = {}, {}
    for name, model in tqdm(models.items()):
        # Test all models
        try:
            logging.info(f"Building {name} grid search")
            grid = getParamGrid(name, seed)
            filename = generateFilenameGrid(
                grid, icd10, nfeatures, name, seed, opt_score
            )
            # Load saved models
            if LOAD_STORED_MODELS and filename in os.listdir(
                os.path.join(paths.MODELS_DIR)
            ):
                logging.info(f"Loading {name} grid search from disk")
                grid_search = joblib.load(os.path.join(paths.MODELS_DIR, filename))
            # Refit model
            else:
                logging.info("Defining preprocessing steps")
                # define pipeline
                preprocessor = getPreprocessingPipelines(
                    cat_features,
                    num_features,
                    num_features_special,
                    tree=name in tree_models,
                )
                clf_pipeline = Pipeline(
                    [
                        ("preprocessor", preprocessor),
                        ("classifier", model),
                    ]
                )

                # perform grid search
                logging.info("Starting Grid Search")
                grid_search = GridSearchCV(
                    clf_pipeline,
                    param_grid=grid,
                    cv=5,
                    scoring=scorer,
                    verbose=10,
                    n_jobs=-1,
                    error_score="raise",
                )
                grid_search.fit(X_train, y_train)
                logging.info(f"Saving model: {name}")
                # save model
                joblib.dump(
                    grid_search,
                    os.path.join(os.path.join(paths.MODELS_DIR, filename)),
                )

            # Print best hyperparameters and corresponding scores
            logging.info(f"Best parameters found: {grid_search.best_params_}")
            scores[name] = grid_search.score(X_val, y_val)
            all_grid_search[name] = grid_search
        except Exception as e:
            logging.info(f"Model {name} failed")
            logging.error(f"Error message: {e}")
            continue

    # save results of grid search to file
    logging.info("Saving model selection and grid search results to file")
    saveGridSearchResults(
        all_grid_search,
        opt_score,
        X_val,
        y_val,
        metadata,
        icd10,
    )
