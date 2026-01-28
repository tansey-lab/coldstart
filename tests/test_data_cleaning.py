import os
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from tabulate import tabulate
from pipeline.data_cleaning import (
    split_dose_drug,
    fit_isotonic,
    clean_data,
    remove_noisy_data,
)

# ----------------------
# Fixtures and Mock Data
# ----------------------

@pytest.fixture
def mock_df():
    df_path = os.path.join(os.path.dirname(__file__), "test_raw_df.feather")
    return pd.read_feather(df_path)

@pytest.fixture
def split_df(mock_df):
    return split_dose_drug(mock_df)[0]

# ----------------------
# Unit Tests
# ----------------------

def test_split_dose_drug(mock_df):
    mock_df = mock_df.reset_index(drop=True)
    df, max_drug = split_dose_drug(mock_df)

    assert max_drug == 2
    assert "dose1" in df.columns
    assert "drug1" in df.columns
    assert "dose2" in df.columns
    assert "drug2" in df.columns
    assert df.shape[0] == mock_df.shape[0]

    for i, row in mock_df.iterrows():
        if len(row["drug_ids"]) == 1:
            assert df.loc[i, "drug2"] is None, f"Expected empty drug2 at index {i}"


def test_fit_isotonic(split_df):
    # Mimic what isotonic expects: consistent doses per drug
    split_df["id"] = split_df["id"].astype(str)
    split_df["dose1"] = [0.032, 0.016, 0.25, 2.0, 0.05, 0.13, 10.0, 2.5, 2.5, 0.01]
    split_df["float_value"] = [0.93, 0.97, 0.92, 0.80, 1.21, 0.95, 0.006, 0.56, 2.30, 0.57]

    result = fit_isotonic(split_df, result_col="isotonic_pred", max_num_drugs=2)

    assert "isotonic_pred" in result.columns
    assert len(result) == len(split_df)
    assert not result["isotonic_pred"].isnull().any()


def test_remove_noisy_data(split_df):
    split_df["float_value"] = np.array([0.93, 0.97, 0.92, 0.80, 1.21, 0.95, 0.006, 0.56, 2.30, 0.57])
    split_df["isotonic_pred"] = split_df["float_value"] * 0.98  # Simulate slight prediction errors

    # Threshold too high to remove anything
    cleaned = remove_noisy_data(split_df, threshold=0.1)
    assert len(cleaned) == len(split_df)

    # Make all predictions bad to trigger removals
    split_df["isotonic_pred"] = 0.0
    cleaned = remove_noisy_data(split_df, threshold=0.01)
    assert len(cleaned) < len(split_df)

    # Now test remove_study_ids logic
    cleaned = remove_noisy_data(split_df, threshold=0.5, remove_study_ids=["golub_2020", "ctrp2"])
    assert all(~cleaned["study_id"].isin(["golub_2020", "ctrp2"]))


def test_clean_data_pipeline(mock_df):
    df_to_show = mock_df.drop(["drug_treatment_id", "replicate_number",
                               "assay_measurement_ids", "source_file_id", 
                               "study_viability_measurement_id"], axis=1)
    df_to_show["id"] = df_to_show["id"].str[:10]
    df_to_show["sample_id"] = df_to_show["sample_id"].str[:10]
    print("\nCleaned dataframe sample:\n",
          tabulate(df_to_show, headers='keys', tablefmt='psql'))

    df = clean_data(
        mock_df,
        result_col="isotonic_pred",
        remove_study_ids=["benes_2023"],
        max_isotonic_error=0.5,
        use_cache_clean=False,
        use_cache_isotonic=False,
    )

    df_to_show = df.drop(["drug_treatment_id", "replicate_number"], axis=1)
    df_to_show["id"] = df_to_show["id"].str[:10]
    df_to_show["sample_id"] = df_to_show["sample_id"].str[:10]
    df_to_show["drug1"] = df_to_show["drug1"].str[:10]
    df_to_show["drug2"] = df_to_show["drug2"].str[:10]
    print("\nCleaned dataframe sample:\n",
          tabulate(df_to_show, headers='keys', tablefmt='psql'))

    assert isinstance(df, pd.DataFrame)
    assert "isotonic_pred" in df.columns
    assert len(df) > 0
    assert not any(df["study_id"] == "benes_2023")