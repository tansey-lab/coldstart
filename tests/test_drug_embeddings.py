import pytest
import numpy as np
import pandas as pd
from tempfile import TemporaryDirectory

from pipeline.drug_embeddings import (
    morgan, chembert, drug_embeddings,
    drug_response_auc_representation,
    compute_drug_importance_scores
)

# Dummy model
import tensorflow as tf

class DummyModel(tf.keras.Model):
    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        return tf.ones((batch_size, 3, 4, 10), dtype=tf.float32)

@pytest.fixture
def dummy_data():
    smiles_list = ["CCO", "CCN", None]
    vocab_size = 3
    study_sample_pairs = [(0, 0), (1, 1)]
    drug_smile_labels = [0, 1, 2]
    drug_no_smile_labels = [0, 1, 2]
    combi_drugs = pd.DataFrame({
        "drug1_smile": [0, 1],
        "drug2_smile": [1, 2],
        "drug1_no_smile": [0, 1],
        "drug2_no_smile": [1, 2]
    })
    return smiles_list, vocab_size, study_sample_pairs, drug_smile_labels, drug_no_smile_labels, combi_drugs

@pytest.fixture
def model():
    return DummyModel()

def test_morgan(dummy_data):
    smiles_list, vocab_size, *_ = dummy_data
    fp = morgan(smiles_list, vocab_size)
    assert fp.shape == (vocab_size, 1024)
    assert np.all(fp >= 0)

@pytest.mark.skip(reason="ChemBERTa requires GPU and internet access to load the model.")
def test_chembert(dummy_data):
    smiles_list, vocab_size, *_ = dummy_data
    emb = chembert(smiles_list, vocab_size)
    assert emb.shape[0] == vocab_size

def test_drug_embeddings(dummy_data):
    smiles_list, vocab_size, *_ = dummy_data
    with TemporaryDirectory() as tmpdir:
        emb = drug_embeddings(smiles_list, vocab_size, "morgan", use_cache=True, cache_path=f"{tmpdir}/embeddings.npz")
        assert isinstance(emb, np.ndarray)
        assert emb.shape[0] == vocab_size

def test_auc_projection(dummy_data, model):
    _, _, study_sample_pairs, _, _, combi_drugs = dummy_data
    with TemporaryDirectory() as tmpdir:
        df = drug_response_auc_representation(
            combi_drugs,
            study_sample_pairs,
            model,
            use_cache=True,
            cache_path=f"{tmpdir}/auc.feather"
        )
        assert "auc" in df.columns
        assert len(df) > 0

def test_importance_scores(dummy_data, model):
    _, _, study_sample_pairs, smile_labels, no_smile_labels, _ = dummy_data
    with TemporaryDirectory() as tmpdir:
        scores, avg = compute_drug_importance_scores(
            model,
            study_sample_pairs,
            smile_labels,
            no_smile_labels,
            num_drugs=3,
            use_cache=True,
            cache_path=f"{tmpdir}/imp.npz"
        )
        assert scores.shape == avg.shape
        assert scores.shape[0] >= len(smile_labels)