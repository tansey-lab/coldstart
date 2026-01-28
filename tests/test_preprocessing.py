import pytest
import pandas as pd
import numpy as np
from tempfile import TemporaryDirectory
from pathlib import Path
from pipeline.preprocessing import preprocessing


@pytest.fixture
def sample_data():
    df = pd.DataFrame({
        "study_id": ["S1", "S1", "S2"],
        "sample_id": ["P1", "P2", "P3"],
        "float_value": [0.5, -0.1, 1.2],
        "drug1": ["D1", "D2", np.nan],
        "drug2": ["D2", "D3", "D1"],
        "dose1": [1e-6, 1e-3, np.nan],
        "dose2": [1e-4, np.nan, 1e-5],
    })

    drug_df = pd.DataFrame({
        "id": ["D1", "D2", "D3"],
        "smiles": ["CCO", "CCN", "CCC"]
    })

    return df, drug_df

def test_preprocessing_basic(sample_data):
    df, drug_df = sample_data
    processed_df, enc_df = preprocessing(df.copy(), drug_df)

    # Check clipping
    assert (processed_df["float_value"] >= 0).all() and (processed_df["float_value"] <= 1).all()

    # Check dose discretization
    for col in ["dose1", "dose2"]:
        assert pd.api.types.is_integer_dtype(processed_df[col])
        assert processed_df[col].min() >= 0
        assert processed_df[col].max() < 100

    # Check encoding columns exist
    assert "study_id" in processed_df.columns
    assert "sample_id" in processed_df.columns
    assert "drug1_smile" in processed_df.columns
    assert "drug2_no_smile" in processed_df.columns

    # Check enc_df has label mappings
    assert "study_id" in enc_df.columns
    assert "smiles" in enc_df.columns
    assert not enc_df.empty

def test_preprocessing_cache(sample_data):
    df, drug_df = sample_data
    with TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        # First run: creates cache
        processed_df_1, enc_df_1 = preprocessing(
            df.copy(), drug_df, use_cache=True, cache_path=str(tmpdir)
        )
        # Second run: loads from cache
        processed_df_2, enc_df_2 = preprocessing(
            df.copy(), drug_df, use_cache=True, cache_path=str(tmpdir)
        )

        # Check cache consistency
        pd.testing.assert_frame_equal(processed_df_1, processed_df_2)
        pd.testing.assert_frame_equal(enc_df_1, enc_df_2)

def test_missing_columns_handled_gracefully(sample_data):
    df, drug_df = sample_data
    # Drop drug2 and dose2
    df = df.drop(columns=["drug2", "dose2"])
    processed_df, _ = preprocessing(df.copy(), drug_df, n_drugs=2)

    # Should fill missing columns
    assert "drug2_smile" in processed_df.columns
    assert "dose2" in processed_df.columns