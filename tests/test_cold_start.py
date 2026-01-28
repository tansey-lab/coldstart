import numpy as np
import pandas as pd
import pytest
from sklearn.metrics.pairwise import euclidean_distances

from pipeline.cold_start import (
    objective,
    get_best_doses,
    select_diverse_drug_combinations,
    assign_doses_to_combinations,
    cold_start_kmedoids,
)

# -------------------------
# Fixtures
# -------------------------

@pytest.fixture
def mock_importance_scores():
    np.random.seed(123)
    return np.random.random((3, 10))

@pytest.fixture
def target_doses():
    return np.arange(10)  # Assume dose levels are 0, 1, 2

@pytest.fixture
def mock_drug_df():
    return pd.DataFrame({
        "study_id": [1, 1, 2, 2],
        "sample_id": [1, 2, 1, 2],
        "drug1": [0, 1, 0, 2],
        "drug2": [1, 2, 2, 0],
        "auc": [0.5, 0.6, 0.55, 0.65]
    })


# -------------------------
# Unit Tests
# -------------------------

def test_objective_basic():
    doses = [0, 2]
    weights = np.array([0.1, 0.5, 0.4])
    val = objective(doses, weights)
    assert 0 <= val <= 2, "Objective value out of expected range"

def test_get_best_doses_returns_valid_doses(target_doses):
    weights = np.array([0.3, 0.4, 0.3, 0.1, 0.6]*2)
    doses = get_best_doses(n=3, weights=weights, target_doses=target_doses, n_iter=5)
    assert len(doses) == 3
    assert set(doses).issubset(set(target_doses))

def test_select_diverse_drug_combinations(mock_drug_df):
    selected_combos = select_diverse_drug_combinations(mock_drug_df, n_combinations=2)
    assert selected_combos.shape == (2, 2)
    assert selected_combos.dtype.kind in {'i', 'u'}  # integers

def test_assign_doses_to_combinations(mock_importance_scores, target_doses):
    selected_combos = np.array([
        [0, 1],
        [0, 2],
        [1, 2]
    ])
    doses = assign_doses_to_combinations(selected_combos, mock_importance_scores, target_doses)
    assert doses.shape == selected_combos.shape
    assert np.all(np.isin(doses, target_doses))

def test_cold_start_kmedoids_pipeline(mock_drug_df, mock_importance_scores, target_doses, tmp_path):
    save_path = tmp_path / "results.csv"
    combos, doses = cold_start_kmedoids(
        drug_rpz_df=mock_drug_df,
        importance_scores=mock_importance_scores,
        unique_dose=target_doses,
        n_combinations=2,
        save_path=str(save_path)
    )
    assert combos.shape == doses.shape
    assert save_path.exists()