"""Configuration file for defining paths."""

import os

# Set the base directory of your project
BASE_DIR = os.getcwd()

# Set the path for your data directories
DATA_DIR = os.path.join(BASE_DIR, "data")
DATA_DIR_RAW = os.path.join(DATA_DIR, "raw")
DATA_DIR_INTERIM = os.path.join(DATA_DIR, "interim")
DATA_DIR_EXTERNAL = os.path.join(DATA_DIR, "external")
DATA_DIR_FIELDS = os.path.join(DATA_DIR_RAW, "fields")
DATA_DIR_DIAGNOSIS = os.path.join(DATA_DIR_RAW, "diagnosis")
DATA_DIR_PROCESSED = os.path.join(DATA_DIR, "processed")
RESULTS_DIR = os.path.join(DATA_DIR, "results")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
FIGURES_DIR = os.path.join(REPORTS_DIR, "figures")
MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_ML_DIR = os.path.join(REPORTS_DIR, "ml")
REPORTS_CATBOOST = os.path.join(REPORTS_ML_DIR, "catboost")
REPORTS_FAIRNESS = os.path.join(REPORTS_DIR, "fairness")

# Set the path for your logs directory
LOGS_DIR = os.path.join(BASE_DIR, "logs")
