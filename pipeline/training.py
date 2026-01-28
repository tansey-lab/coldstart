import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from typing import Optional, Tuple, List, Dict

SEED = 42

OPTIMIZERS_DICT = {
    "adam": tf.keras.optimizers.Adam
}

def stratified_split(df: pd.DataFrame, test_size: float = 0.01) -> Tuple[np.ndarray, np.ndarray]:
    """
    Efficient stratified split on (sample_id, drug1) combinations, returning only row indices.

    Ensures all test rows have at least one shared sample_id and drug1 in training.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    test_size : float
        Proportion of data to assign to the test set.

    Returns
    -------
    train_index : np.ndarray
        Indices for training data.
    test_index : np.ndarray
        Indices for test data (no sample_id or drug1 leakage).
    """
    drug_columns = [c for c in df.columns if c.endswith("_smile") and not c.endswith("_no_smile")]
    
    # Create group IDs for (sample_id, drug1) combinations
    group_series = df.groupby(["sample_id"] + drug_columns, sort=False).ngroup()
    group_ids = group_series.to_numpy()
    
    # Mapping group -> list of row indices
    group_to_indices = df.groupby(group_series, sort=False).indices

    # Stratified split on unique group IDs
    unique_groups = np.unique(group_ids)
    train_groups, test_groups = train_test_split(unique_groups, test_size=test_size, random_state=SEED)

    # Get row indices for train/test groups
    train_idx = np.concatenate([group_to_indices[g] for g in train_groups])
    test_idx = np.concatenate([group_to_indices[g] for g in test_groups])

    # Ensure no leakage on sample_id or drug1
    train_sample_ids = set(df.loc[train_idx, "sample_id"])
    train_drugs = set(df.loc[train_idx][drug_columns].dropna().values.ravel())

    leak_mask = ~df.loc[test_idx, "sample_id"].isin(train_sample_ids)
    for col in drug_columns:
        leak_mask = leak_mask | ~df.loc[test_idx, col].isin(train_drugs)
    reassign_idx = test_idx[leak_mask.to_numpy()]

    # Reassign leaked samples to train
    train_idx = np.concatenate([train_idx, reassign_idx])
    test_idx = np.setdiff1d(test_idx, reassign_idx, assume_unique=True)

    # Final leakage check
    test_values = [df.loc[test_idx, "sample_id"]] + [df.loc[test_idx, col] for col in drug_columns]
    train_values = [df.loc[train_idx, "sample_id"]] + [df.loc[train_idx, col] for col in drug_columns]
    
    test_pairs = set(zip(*test_values))
    train_pairs = set(zip(*train_values))
    
    assert not test_pairs & train_pairs, "Train and test leakage detected!"

    return train_idx, test_idx


def extract_held_out_drug_split(
    df: pd.DataFrame,
    drugs_to_keep: set = set(),
    n_held_out_drugs: int = 20,
    seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the dataset into training and held-out sets based on unseen drug1 compounds.

    The held-out set contains rows where drug1:
    - Is not in drugs_to_keep
    - Is not used as drug2
    - Has a valid SMILES (i.e., drug1_no_smile == 0)

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset with 'drug1', 'drug2', and 'drug1_no_smile' columns.
    drugs_to_keep : set
        Set of drug1 IDs that must remain in the training set.
    n_held_out_drugs : int
        Number of unique drug1 compounds to hold out.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        (train_df, held_out_df)
    """
    candidates = set(df["drug1_smile"])
    for i in range(2, 10):
        if "drug%i_smile"%i in df.columns:
            candidates = candidates - set(df["drug%i_smile"%i])
    candidates = candidates - set(drugs_to_keep)

    valid_rows = df["drug1_no_smile"] == 0
    valid_drugs = set(df.loc[valid_rows, "drug1_smile"])
    valid_drugs = list(candidates & valid_drugs)

    np.random.seed(seed)
    selected_drugs = np.random.choice(valid_drugs, size=min(n_held_out_drugs, len(valid_drugs)), replace=False)

    held_out_df = df[df["drug1_smile"].isin(selected_drugs)]
    train_df = df[~df["drug1_smile"].isin(selected_drugs)]

    return train_df, held_out_df


class EvaluateLossCallback(tf.keras.callbacks.Callback):
    """
    Keras callback to evaluate and log loss on held-out data at the end of each epoch.

    Parameters
    ----------
    X : tf.Tensor or np.ndarray
        Input features for the held-out dataset.
    y : tf.Tensor or np.ndarray
        True targets for the held-out dataset.
    file_history_path : str
        Path to save the CSV file tracking evaluation history.
    """
    def __init__(self, X, y, file_history_path: str):
        super().__init__()
        self.X = tf.convert_to_tensor(X)
        self.y = tf.convert_to_tensor(y)
        self.history = []
        self.file_history_path = file_history_path

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        y_pred = self.model(self.X, training=False)
        loss = self.model.loss(self.y, y_pred)
        avg_loss = tf.reduce_mean(loss).numpy()

        logs["held_out_drugs_loss"] = avg_loss
        self.history.append({"epoch": epoch + 1, **logs})

        pd.DataFrame(self.history).to_csv(self.file_history_path, index=False)


def mae_loss_1(y_true, y_pred):
    indexes1 = tf.cast(tf.reshape(y_true[:, 1], (-1,)), tf.int32)
    
    yp1 = tf.gather(y_pred[:, 0], indexes1, axis=-1, batch_dims=1)

    mu = tf.reduce_mean(yp1, axis=-1)
    
    mu_errors = tf.abs(tf.cast(y_true[:, 0], mu.dtype) - mu)
    loss = tf.reduce_mean(mu_errors)
    return loss


def mae_loss_2(y_true, y_pred):
    indexes1 = tf.cast(tf.reshape(y_true[:, 1], (-1,)), tf.int32)
    indexes2 = tf.cast(tf.reshape(y_true[:, 2], (-1,)), tf.int32)
    
    yp1 = tf.gather(y_pred[:, 0], indexes1, axis=-1, batch_dims=1)
    yp2 = tf.gather(y_pred[:, 1], indexes2, axis=-1, batch_dims=1)

    mu = tf.reduce_mean(yp1*yp2, axis=-1)
    
    mu_errors = tf.abs(tf.cast(y_true[:, 0], mu.dtype) - mu)
    loss = tf.reduce_mean(mu_errors)
    return loss


def mae_loss_3(y_true, y_pred):
    indexes1 = tf.cast(tf.reshape(y_true[:, 1], (-1,)), tf.int32)
    indexes2 = tf.cast(tf.reshape(y_true[:, 2], (-1,)), tf.int32)
    indexes3 = tf.cast(tf.reshape(y_true[:, 3], (-1,)), tf.int32)
    
    yp1 = tf.gather(y_pred[:, 0], indexes1, axis=-1, batch_dims=1)
    yp2 = tf.gather(y_pred[:, 1], indexes2, axis=-1, batch_dims=1)
    yp3 = tf.gather(y_pred[:, 2], indexes3, axis=-1, batch_dims=1)

    mu = tf.reduce_mean(yp1*yp2*yp3, axis=-1)
    
    mu_errors = tf.abs(tf.cast(y_true[:, 0], mu.dtype) - mu)
    loss = tf.reduce_mean(mu_errors)
    return loss


def train_pipeline(
    df: pd.DataFrame,
    model: tf.keras.Model,
    n_drugs: int,
    batch_size: int,
    eval_batch_size: int,
    epochs: int,
    save_path: str = None,
    drugs_to_keep: Optional[set] = set(),
    steps_per_epoch: int = None,
    n_held_out_drugs: int = 20,
    history_path: Optional[str] = "./history.csv",
    seed: int = 42,
    optimizer: str = "adam",
    optimizer_params: dict = {},
    use_cache: bool = False
):
    if use_cache and save_path and os.path.exists(save_path):
        model.load_weights(save_path)
        return True
    # Dynamically find all drug smile/no_smile feature columns
    cols = df.columns
    smile_cols = []
    no_smile_cols = []
    dose_cols = []
    for i in range(1, n_drugs+1):
        smile_cols.append("drug%i_smile"%i)
        no_smile_cols.append("drug%i_no_smile"%i)
        dose_cols.append("dose%i"%i)
    feature_cols = ["study_id", "sample_id"] + smile_cols + no_smile_cols
    target_cols = ["float_value"] + dose_cols

    # Extract held-out drugs (unseen during training)
    df, held_out_df = extract_held_out_drug_split(
        df, drugs_to_keep=drugs_to_keep, n_held_out_drugs=n_held_out_drugs, seed=seed
    )
    df.reset_index(drop=True, inplace=True)
    held_out_df.reset_index(drop=True, inplace=True)

    print(f"Held-out data selected! Held-out {held_out_df.shape} - Remaining {df.shape}")
    
    # Stratified split of train/validation indices
    train_idx, val_idx = stratified_split(df)

    print(f"Train/Val split completed! Train {len(train_idx)} - Val {len(val_idx)}")

    X = df[feature_cols]
    y = df[target_cols]

    # Prepare tf.data datasets
    train_ds = tf.data.Dataset.from_tensor_slices((X.loc[train_idx].values.astype(np.float32),
                                                   y.loc[train_idx].values.astype(np.float32))).shuffle(1024).batch(batch_size)
    val_ds = tf.data.Dataset.from_tensor_slices((X.loc[val_idx].values.astype(np.float32),
                                                 y.loc[val_idx].values.astype(np.float32))).batch(eval_batch_size)

    # Callbacks
    if save_path:
        checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
            filepath=save_path, save_weights_only=True, monitor="val_loss", mode="min", save_best_only=True, verbose=1
        )
        evaluation_cb = EvaluateLossCallback(
            held_out_df[feature_cols].values.astype(np.float32),
            held_out_df[target_cols].values.astype(np.float32),
            file_history_path=history_path
        )
        callbacks = [checkpoint_cb, evaluation_cb]
    else:
        callbacks = []

    if n_drugs == 1:
        loss = mae_loss_1
    elif n_drugs == 2:
        loss = mae_loss_2
    elif n_drugs == 3:
        loss = mae_loss_3

    # Compile and train
    model.compile(optimizer=OPTIMIZERS_DICT[optimizer](**optimizer_params), loss=loss)
    model.fit(train_ds, validation_data=val_ds, steps_per_epoch=steps_per_epoch, epochs=epochs, callbacks=callbacks)
    return True


def train_fm(
    df: pd.DataFrame,
    model: tf.keras.Model,
    drug_embs: None,
    n_drugs: int,
    batch_size: int,
    eval_batch_size: int,
    epochs: int,
    save_path: str = None,
    drugs_to_keep: Optional[set] = set(),
    steps_per_epoch: int = None,
    n_held_out_drugs: int = 20,
    history_path: Optional[str] = "./history.csv",
    seed: int = 42,
    optimizer: str = "adam",
    optimizer_params: dict = {},
    use_cache: bool = False
):
    # Morgan Tokens
    max_length = int(drug_embs.sum(1).max() + 1)

    tokens_ids = np.full((drug_embs.shape[0], max_length), 1025, dtype=np.int32)
    tokens_ids[:, 0] = 1024
    attention_mask = np.zeros((drug_embs.shape[0], max_length), dtype=np.int32)
    
    for i in range(emb.shape[0]):
        args = np.argwhere(drug_embs[i]).ravel()
        tokens_ids[i, 1:len(args)+1] = args
        attention_mask[i, :len(args)+1] = 1

    
    