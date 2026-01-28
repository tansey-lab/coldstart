import pandas as pd
import pytest
import tempfile
import os
from unittest.mock import patch, MagicMock

from pipeline.drug_library import extract_drug_info


@pytest.fixture
def sample_csv_file():
    df = pd.DataFrame({
        "Drug Name": ["DrugA", "DrugB", "DrugA", None]
    })

    with tempfile.NamedTemporaryFile(mode='w', suffix=".csv", delete=False) as f:
        df.to_csv(f.name, index=False)
        yield f.name
    os.remove(f.name)


@pytest.fixture
def mock_drug_object():
    def mock(name, study, source_file):
        mock_obj = MagicMock()
        mock_obj.id = f"id_{name}"
        mock_obj.study_id = "study_123"
        mock_obj.name = name
        mock_obj.source_file_id = "source_file_123"
        mock_obj.pubchem_name = f"Pubchem_{name}"
        mock_obj.pubchem_substance_id = f"Substance_{name}"
        mock_obj.pubchem_compound_id = f"Compound_{name}"
        mock_obj.pubchem_parent_compound_id = f"Parent_{name}"
        mock_obj.smiles = f"SMILES_{name}"
        mock_obj.fda_approval.value = True
        return mock_obj
    return mock


def test_extract_drug_info_csv(sample_csv_file, mock_drug_object):
    with tempfile.NamedTemporaryFile(mode='w', suffix=".csv", delete=False) as cache_file:
        cache_path = cache_file.name

    with patch("pipeline.drug_library.drug_object_from_name", side_effect=mock_drug_object):
        df = extract_drug_info(
            raw_path=sample_csv_file,
            column_name="Drug Name",
            cache_path=cache_path,
            use_cache=False
        )

    assert df.shape[0] == 2  # Unique drugs: DrugA, DrugB
    assert "smiles" in df.columns
    assert df["name"].tolist() == ["DrugA", "DrugB"]

    # Re-run with use_cache=True to test loading
    df_cached = extract_drug_info(
        raw_path=sample_csv_file,
        column_name="Drug Name",
        cache_path=cache_path,
        use_cache=True
    )
    assert df_cached.equals(df)

    os.remove(cache_path)