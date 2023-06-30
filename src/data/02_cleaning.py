"""Script to perform basic data cleaning."""
import logging
import os

from src.config import paths
from src.data.loading import getAllFields
from src.data.utils import dropAxisWithManyNaN, setDataTypes

if __name__ == "__main__":
    logging.basicConfig(
        filename=os.path.join(paths.LOGS_DIR, "cleaning.log"),
        filemode="w",
        encoding="utf-8",
        format="%(asctime)s %(levelname)s: %(message)s",
        level=logging.INFO,
    )
    logging.info("Data loading...")

    # 1. Load features
    X = getAllFields()
    all_fields = X.columns.tolist()

    # 2. Basic data cleaning
    logging.info("Starting data cleaning")
    logging.info("Removing columns with more than 90% of null values")
    X = dropAxisWithManyNaN(X, axis=1)
    dropped_fields = set(all_fields) - set(X.columns)
    for col in dropped_fields:
        logging.info(f"Removed {col}")
    logging.info("Removing rows with more than 90% of null values")
    X = dropAxisWithManyNaN(X, axis=0)

    # 3. Set correct data types
    logging.info("Setting correct data types")
    setDataTypes(X)

    # 4. Save final feature file
    X.to_parquet(
        os.path.join(paths.DATA_DIR_PROCESSED, "fields.parquet.gzip"),
        compression="gzip",
    )
