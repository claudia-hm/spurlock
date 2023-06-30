"""Script to perform data loading from files.

Author: Claudia Herron Mulet (clherrom12@alumnes.ub.edu)
Date: 20/02/2023
"""
import os
from typing import List

import numpy as np
import pandas as pd

from src.config import paths


def getDiseases() -> pd.DataFrame:
    """Read file with disease codes and return dataframe."""
    return pd.read_csv(
        os.path.join(paths.DATA_DIR_DIAGNOSIS, "f.41270.tab"),
        sep="\t",
        index_col=0,
        low_memory=False,
    ).set_index("f.eid")


def getDiagnosisDates() -> pd.DataFrame:
    """Read file with diagnosis dates and return dataframe.\
    Basic cleaning is performed."""
    dates = (
        pd.read_csv(
            os.path.join(paths.DATA_DIR_DIAGNOSIS, "f.41280.tab"),
            sep="\t",
            low_memory=False,
        )
        .iloc[:, 1:]
        .set_index("f.eid")
    )
    dates.replace("+", np.nan, inplace=True)
    colnames = dates.columns
    for col in colnames:
        dates[col] = pd.to_datetime(dates[col])
    return dates


def getAssessmentDate() -> pd.DataFrame:
    """Read file with date of visit to assessment center\
    and return dataframe. Basic cleaning is performed."""
    assessment_date = pd.read_csv(
        os.path.join(paths.DATA_DIR_DIAGNOSIS, "f.53.tab"),
        sep="\t",
        usecols=[0, 1],
        low_memory=False,
    ).set_index("f.eid")
    assessment_date["f.53.0.0"] = pd.to_datetime(assessment_date["f.53.0.0"])
    return assessment_date.loc[:, "f.53.0.0"]


def getCodings() -> pd.DataFrame:
    """Load file for data coding, categories, provided by UKBiobank."""
    return pd.read_csv(
        os.path.join(paths.DATA_DIR_EXTERNAL, "Codings.csv"),
        on_bad_lines="skip",
        encoding="ISO-8859-1",
    ).set_index("Coding")


def getDataDictShowcase() -> pd.DataFrame:
    """Load data dictionary file provided by UKBiobank."""
    return pd.read_csv(
        os.path.join(paths.DATA_DIR_EXTERNAL, "Data_Dictionary_Showcase.csv"),
        on_bad_lines="skip",
    )[["FieldID", "Field", "ValueType", "Coding"]].set_index("FieldID")


def getField(field_file: str) -> pd.DataFrame:
    """Load a specific field file.

    Args:
        field_file: file to retrieve.
    """
    df = pd.read_csv(
        os.path.join(paths.DATA_DIR_FIELDS, field_file),
        sep="\t",
        index_col="f.eid",
        encoding="ISO-8859-1",
        on_bad_lines="skip",
    )
    if "Unnamed: 0" in list(df.columns):
        return df.drop(columns=["Unnamed: 0"])
    elif "Unnamed:" in list(df.columns):
        return df.drop(columns=["Unnamed:"])
    elif "Identifier" in list(df.columns):
        return df.drop(columns=["Identifier"])
    else:
        return df


def getAllFields(debug: bool = False) -> pd.DataFrame:
    """Load file containing all fields.

    Args:
        debug: boolean to retrieve only a subset of\
         patients, for debugging purposes.
    """
    if debug:
        return pd.read_csv(
            os.path.join(paths.DATA_DIR_INTERIM, "fields.csv"),
            low_memory=False,
            index_col="f.eid",
            nrows=100,
        )
    else:
        return pd.read_csv(
            os.path.join(paths.DATA_DIR_INTERIM, "fields.csv"),
            low_memory=False,
            index_col="f.eid",
        )


def getDisease(icd10: str) -> pd.DataFrame:
    """Load disease file, indicating for every patient\
    if it suffered the disease posterior to first visit\
    to assessment center."""
    return pd.read_csv(
        os.path.join(paths.DATA_DIR_INTERIM, icd10.upper() + ".csv"), index_col="f.eid"
    ).squeeze()


def getMetadata() -> pd.DataFrame:
    """Load metadata file."""
    return pd.read_csv(
        os.path.join(paths.DATA_DIR_INTERIM, "metadata_fields.csv"), index_col="Field"
    ).drop_duplicates()


def getMetadataWithCategories() -> pd.DataFrame:
    """Load metadata file."""
    return pd.read_csv(
        os.path.join(paths.DATA_DIR_INTERIM, "metadata_fields_with_category.csv"),
        index_col="Field",
    ).drop_duplicates()


def getProcessedFields() -> pd.DataFrame:
    """Load processed file containing all fields."""
    return pd.read_parquet(
        os.path.join(paths.DATA_DIR_PROCESSED, "fields.parquet.gzip")
    )


def getSpecialNaNFeatures() -> List:
    """Get a list of features with nan features marked as -1."""
    return (
        pd.read_csv(os.path.join(paths.DATA_DIR_INTERIM, "replaced_nan.csv"))
        .values.flatten()
        .tolist()
    )
