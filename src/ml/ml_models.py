"""Script where to specify all ML models and\
 parameters for classification pipeline.

Author: Claudia Herron Mulet (clherrom12@alumnes.ub.edu)
Date: 16/03/2023
"""
import multiprocessing

from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from src.config import paths

DEBUG = False

models = {
    "RandomForestClassifier": RandomForestClassifier(
        n_jobs=multiprocessing.cpu_count() - 1, verbose=10, class_weight="balanced"
    ),
    "LogisticRegression": LogisticRegression(
        max_iter=1000, n_jobs=-1, class_weight="balanced", solver="newton-cholesky"
    ),
    "XGBClassifier": XGBClassifier(
        n_jobs=multiprocessing.cpu_count() - 1, class_weight="balanced"
    ),
    "CatBoostClassifier": CatBoostClassifier(
        train_dir=paths.REPORTS_CATBOOST,
        silent=True,
    ),
}

tree_models = [
    "CatBoostClassifier",
    "XGBClassifier",
    "RandomForestClassifier",
    "AdaBoostClassifier",
    "DecisionTreeClassifier",
    "GradientBoostingClassifier",
    "BalancedRandomForestClassifier",
]

if DEBUG:

    def getParamGrid(name: str, seed: int):
        """Returns param grid depending on model name.

        Args:
            name: model name.
            seed: random seed.
        """
        param_grid = {
            "LogisticRegression": {
                "classifier__random_state": [seed],
            },
            "RandomForestClassifier": {"classifier__random_state": [seed]},
            "SVC": {
                "classifier__random_state": [seed],
            },
            "KNeighborsClassifier": {},
            "DecisionTreeClassifier": {
                "classifier__random_state": [seed],
            },
            "GradientBoostingClassifier": {
                "classifier__random_state": [seed],
            },
            "AdaBoostClassifier": {
                "classifier__random_state": [seed],
            },
            "XGBClassifier": {
                "classifier__random_state": [seed],
            },
            "GaussianNB": {},
            "BernoulliNB": {},
            "LinearDiscriminantAnalysis": {},
            "QuadraticDiscriminantAnalysis": {},
            "CatBoostClassifier": {
                "classifier__random_state": [seed],
            },
        }
        return param_grid[name]

else:

    def getParamGrid(name: str, seed: int):
        """Returns param grid depending on model name.

        Args:
            name: model name.
            seed: random seed.
        """
        param_grid = {
            "LogisticRegression": {
                "classifier__random_state": [seed],
                "classifier__penalty": ["l2"],
                "classifier__C": [0.01, 0.1, 1],  # , 10, 100]
            },
            "RandomForestClassifier": {
                "classifier__random_state": [seed],
                "classifier__n_estimators": [10, 50, 100],
                "classifier__max_depth": [5, 10, None],
                "classifier__min_samples_split": [2, 5, 10],
                "classifier__min_samples_leaf": [1, 2, 4],
                "classifier__bootstrap": [True, False],
            },
            "SVC": {
                "classifier__random_state": [seed],
                "classifier__kernel": ["linear", "poly", "rbf", "sigmoid"],
                "classifier__C": [0.01, 0.1, 1, 2],
                "classifier__gamma": ["scale", "auto"],
            },
            "KNeighborsClassifier": {
                "classifier__n_neighbors": [3, 5, 7],
                "classifier__weights": ["uniform", "distance"],
            },
            "DecisionTreeClassifier": {
                "classifier__random_state": [seed],
                "classifier__criterion": ["gini", "entropy"],
                "classifier__max_depth": [5, 10, None],
                "classifier__min_samples_split": [2, 5, 10],
                "classifier__min_samples_leaf": [1, 2, 4],
                "classifier__max_features": ["sqrt", "log2", None],
            },
            "GradientBoostingClassifier": {
                "classifier__random_state": [seed],
                "classifier__n_estimators": [50, 100, 200],
                "classifier__max_depth": [3, 5, None],
                "classifier__learning_rate": [0.01, 0.1, 1],
                "classifier__subsample": [0.5, 0.8, 1.0],
            },
            "AdaBoostClassifier": {
                "classifier__random_state": [seed],
                "classifier__n_estimators": [50, 100, 200],
                "classifier__learning_rate": [0.01, 0.1, 1],
                "classifier__algorithm": ["SAMME", "SAMME.R"],
            },
            "XGBClassifier": {
                "classifier__random_state": [seed],
                "classifier__n_estimators": [50, 100, 200],
                "classifier__max_depth": [3, 5, 10],
                "classifier__learning_rate": [0.01, 0.1, 1],
                "classifier__subsample": [0.5, 0.8, 1.0],
            },
            "GaussianNB": {},
            "BernoulliNB": {},
            "LinearDiscriminantAnalysis": {
                "classifier__solver": ["svd", "lsqr", "eigen"],
                "classifier__shrinkage": [None, "auto", 0.1, 0.5, 0.9],
                "classifier__n_components": [None, 1],
                "classifier__store_covariance": [True, False],
                "classifier__tol": [1e-4, 1e-3, 1e-2],
            },
            "QuadraticDiscriminantAnalysis": {
                "classifier__reg_param": [0.0, 0.1, 0.5, 1.0],
                "classifier__store_covariance": [True, False],
                "classifier__tol": [1e-4, 1e-3, 1e-2],
            },
            "CatBoostClassifier": {
                "classifier__random_state": [seed],
                "classifier__iterations": [500],
                "classifier__learning_rate": [0.01, 0.05, 0.1],
                "classifier__depth": [4, 8],
                "classifier__l2_leaf_reg": [1, 5],
                "classifier__border_count": [32, 64],
            },
        }
        return param_grid[name]
