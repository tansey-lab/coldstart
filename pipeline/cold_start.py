from kmedoids import KMedoids
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import pairwise_distances
import numpy as np
import pandas as pd
from tqdm import tqdm


def objective(doses, weights):
    doses_pdf = np.zeros(len(weights))
    doses_pdf[doses] = 1.
    doses_pdf /= doses_pdf.sum()
    
    weights_pdf = weights / weights.sum()

    cdf1 = np.cumsum(doses_pdf)
    cdf2 = np.cumsum(weights_pdf)
    
    obj = np.abs(cdf1 - cdf2).sum()
    return obj


def get_best_doses(n, weights, target_doses, n_iter=10):
    doses = np.random.choice(len(target_doses), n, replace=False)
    for k in range(n_iter):
        for d1 in range(len(target_doses)):
            changes = []
            old_obj = objective(doses, weights)
            for i in range(len(doses)):
                doses_copy = doses.copy()
                doses_copy[i] = d1
                obj = objective(doses_copy, weights) - old_obj
                changes.append(obj)
            if np.min(changes) < 0.:
                arg = np.argmin(changes)
                doses[arg] = d1
    return target_doses[doses]


def select_diverse_drug_combinations(
    drug_rpz_df: pd.DataFrame,
    n_combinations: int = 100,
    distance_metric: str = "euclidean"
):
    """
    Select diverse drug combinations using k-medoids.

    Args:
        drug_rpz_df: DataFrame with drug columns (e.g., 'drug1', 'drug2', ...) and 'study_id', 'sample_id', 'auc'.
        n_combinations: Number of diverse combinations to select.
        distance_metric: Distance metric used in k-medoids.

    Returns:
        selected_combinations: np.ndarray of shape (n_combinations, n_drugs)
        drug_cols: list of column names used to identify drugs
    """
    np.random.seed(123)
    
    drug_cols = [col for col in drug_rpz_df.columns if col.startswith("drug")]
    n_drugs = len(drug_cols)

    n_samples = drug_rpz_df[["study_id", "sample_id"]].drop_duplicates().shape[0]
    n_drug_combos = drug_rpz_df.shape[0] // n_samples
    
    assert n_drug_combos * n_samples == drug_rpz_df.shape[0]
    
    matrix = np.empty((n_drug_combos, n_samples), dtype=np.float64)

    # Group by study_id and sample_id
    grouped = drug_rpz_df.groupby(['study_id', 'sample_id'])
    
    drug_combinations = next(iter(grouped))[1][drug_cols].values
    
    # Fill matrix
    for j, group in tqdm(enumerate(grouped), total=n_samples, desc="Pivot"):
        index = group[1][drug_cols].values
        assert np.all(index == drug_combinations)
        values = group[1]["auc"].values
        matrix[:, j] = values

    # Compute distances between drug combinations
    print("Computing distance matrix...")
    distance_matrix = pairwise_distances(matrix, Y=None, metric='euclidean', n_jobs=-1)

    print("KMedoids...")
    # Perform k-medoids clustering
    km = KMedoids(n_clusters=n_combinations, metric="precomputed", method='fasterpam',
                  random_state=42)
    km.fit(distance_matrix)

    selected_combinations = drug_combinations[km.medoid_indices_]

    return selected_combinations


def assign_doses_to_combinations(
    selected_combinations: np.ndarray,
    importance_scores,
    target_doses: np.ndarray,
):
    """
    Assigns optimal doses to a set of selected drug combinations.

    Args:
        selected_combinations: Array of shape (n_combinations, n_drugs) with drug IDs.
        drug_cols: List of column names (e.g., ['drug1', 'drug2', ...]) for drug positions.
        get_drug_weight: Function to return per-drug weights.
        unique_dose: Array of possible dose levels.
        get_best_doses: Function that takes (n, weight, unique_dose) and returns n best doses.

    Returns:
        dose_combinations: Array of shape (n_combinations, n_drugs) with dose assignments.
    """
    np.random.seed(123)
    
    n_combinations, n_drugs = selected_combinations.shape
    unique_drugs = np.unique(selected_combinations).astype(int)

    # Get weights for all unique drugs
    drug_weights_map = importance_scores

    # Map drug → which combination indices it appears in
    drug_to_indices = {
        drug: np.argwhere((selected_combinations == drug))
        for drug in unique_drugs
    }

    # Preallocate dose assignment matrix
    dose_combinations = np.zeros_like(selected_combinations, dtype=np.int32)

    for drug in unique_drugs:
        indices = drug_to_indices[drug]
        weight = drug_weights_map[drug]
        doses = get_best_doses(len(indices), weight, target_doses)

        for k, idx in enumerate(drug_to_indices[drug]):
            dose_combinations[tuple(idx)] = doses[k]

    return dose_combinations


def cold_start_kmedoids(
    drug_rpz_df: pd.DataFrame,
    importance_scores,
    unique_dose: np.ndarray,
    n_combinations: int = 100,
    distance_metric: str = "euclidean",
    save_path: str = None
):
    """
    Complete cold start pipeline: select diverse drug combos + assign optimal doses.
    """
    print("Drug Combos Selection...")
    selected_combos = select_diverse_drug_combinations(
        drug_rpz_df, n_combinations=n_combinations, distance_metric=distance_metric
    )
    print("Dose Combos Selection...")
    dose_combos = assign_doses_to_combinations(
        selected_combos, importance_scores[:, unique_dose], unique_dose
    )
    if save_path:
        pd.DataFrame(np.concatenate((selected_combos, dose_combos), axis=1)).to_csv(save_path)
    
    return selected_combos, dose_combos