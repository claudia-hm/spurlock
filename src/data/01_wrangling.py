"""Script to perform data wrangling for UK Biobank files.

In particular, this script does the following tasks:
- Construct a dataset from a set of field files
- Label patients indicating future presence or not of icd10.

Author: Claudia Herron Mulet (clherrom12@alumnes.ub.edu)
Date: 20/02/2023
"""
import argparse
import logging
import os

from src.config import paths
from src.data.utils import createDatasetFromFields, createDiseaseLabel

if __name__ == "__main__":
    # 1. Read arguments
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--selected_fields", action="store_true", help="Read selected fields from file"
    )

    args = parser.parse_args()
    logging.basicConfig(
        filename=os.path.join(paths.LOGS_DIR, "wrangling.log"),
        filemode="w",
        encoding="utf-8",
        format="%(asctime)s %(levelname)s: %(message)s",
        level=logging.INFO,
    )

    logging.info("Starting data wrangling")
    logging.info("Creating dataset")

    # 2. Create feature dataset from fields
    fields_from_file = args.selected_fields
    createDatasetFromFields(from_file=fields_from_file)

    # 3. Create disease labels
    logging.info("Creating disease labels")
    createDiseaseLabel()

    logging.info("Data wrangling completed")
