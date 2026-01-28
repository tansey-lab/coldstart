import pytest
import pandas as pd
import numpy as np
import tensorflow as tf
import tempfile
import os

from pipeline.training import train_pipeline, stratified_split, extract_held_out_drug_split
from pipeline.model import create_deep_model


@pytest.fixture
def mock_dataframe():
    # Simulate dataset with 3 drug features, 10 samples
    n_samples = 100
    df = pd.DataFrame({
        "study_id": np.random.randint(0, 5, size=n_samples),
        "sample_id": np.random.randint(0, 10, size=n_samples),
        "drug1_smile": np.random.randint(0, 20, size=n_samples),
        "drug1_no_smile": np.zeros(n_samples, dtype=int),
        "drug2_smile": np.random.randint(0, 2, size=n_samples),
        "drug2_no_smile": np.zeros(n_samples, dtype=int),
        "dose1": np.random.randint(0, 5, size=n_samples),
        "dose2": np.random.randint(0, 5, size=n_samples),
        "float_value": np.random.rand(n_samples)
    })
    return df


def test_stratified_split_no_leakage(mock_dataframe):
    train_idx, test_idx = stratified_split(mock_dataframe, test_size=0.2)

    train_df = mock_dataframe.loc[train_idx]
    test_df = mock_dataframe.loc[test_idx]

    # Check no overlapping sample_id or drug1_smile values
    train_sample_ids = set(train_df["sample_id"])
    train_drugs = set(train_df["drug1_smile"])

    assert all(sid in train_sample_ids for sid in test_df["sample_id"])
    assert all(drug in train_drugs for drug in test_df["drug1_smile"])


def test_extract_held_out_drug_split(mock_dataframe):
    drugs_to_keep = {0, 1, 2}

    train_df, held_out_df = extract_held_out_drug_split(
        mock_dataframe,
        drugs_to_keep=drugs_to_keep,
        n_held_out_drugs=5,
        seed=42
    )

    held_out_drugs = set(held_out_df["drug1_smile"])
    train_drugs = set(train_df["drug1_smile"])

    # Held-out drugs should not be in drugs_to_keep
    assert held_out_drugs.isdisjoint(drugs_to_keep)

    # Held-out drugs should not be used as drug2
    for i in range(2, 10):
        col = f"drug{i}_smile"
        if col in mock_dataframe.columns:
            assert held_out_drugs.isdisjoint(set(mock_dataframe[col]))

    # Ensure drug1_no_smile == 0 in held-out
    assert (held_out_df["drug1_no_smile"] == 0).all()


def test_train_pipeline_runs(mock_dataframe, tmp_path):
    model, _ = create_deep_model(
    study_vocab_size=30,
    sample_vocab_size=30,
    drug_vocab_size=30,
    n_doses= 10,
    n_drugs= 2,
    emb_dim= 6,
    drug_emb_dim=12,
    n_layers=3,
    units=128,
    rank=3,
    semi_window_size= 5,
    gamma=0.2,
    drug_embeddings_weights=None,
    activation="relu")

    history_path = os.path.join(tmp_path, "history.csv")
    weights_path = os.path.join(tmp_path, "model.weights.h5")

    result = train_pipeline(
        df=mock_dataframe,
        model=model,
        n_drugs=2,
        batch_size=16,
        eval_batch_size=16,
        epochs=2,
        save_path=weights_path,
        drugs_to_keep={0, 1},
        n_held_out_drugs=5,
        history_path=history_path,
        optimizer="adam",
        optimizer_params={"learning_rate": 0.001},
        use_cache=False
    )

    assert result is True
    assert os.path.exists(history_path), "History file not saved"
    assert os.path.exists(weights_path), "Model weights not saved"

    df_history = pd.read_csv(history_path)
    assert not df_history.empty
    assert "held_out_drugs_loss" in df_history.columns