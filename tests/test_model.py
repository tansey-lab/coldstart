import pytest
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from pipeline.model import create_deep_model  # adjust to your actual import path

# ----------------------
# Fixtures and Dummy Data
# ----------------------

@pytest.fixture
def dummy_input():
    # (batch_size, 2 + 2 * n_drugs)
    return np.array([
        [0, 1, 3, 4, 5, 6],
        [1, 2, 4, 5, 6, 7],
        [2, 0, 2, 3, 8, 9],
        [3, 1, 1, 2, 7, 6],
    ], dtype=np.float32)

@pytest.fixture
def model_args():
    return {
        "study_vocab_size": 5,
        "sample_vocab_size": 10,
        "drug_vocab_size": 20,
        "n_doses": 100,
        "n_drugs": 2,
        "emb_dim": 64,
        "drug_emb_dim": 256,
        "n_layers": 2,
        "units": 64,
        "rank": 10,
        "semi_window_size": 3,
        "gamma": 0.2,
        "drug_embeddings_weights": None,
        "activation": "relu"
    }

# ----------------------
# Model Tests
# ----------------------

def test_create_deep_model_structure(model_args):
    model_train, model_predict = create_deep_model(**model_args)

    assert isinstance(model_train, tf.keras.Model)
    if model_args["n_drugs"] <= 3:
        assert isinstance(model_predict, tf.keras.Model)
    else:
        assert model_predict is None

def test_forward_pass_train_model(dummy_input, model_args):
    model_train, _ = create_deep_model(**model_args)
    output = model_train(dummy_input)
    assert isinstance(output, tf.Tensor)
    assert output.shape[0] == dummy_input.shape[0]
    assert output.shape[1] == model_args["n_drugs"]
    assert output.shape[2] == model_args["rank"]
    assert output.shape[3] == model_args["n_doses"]

def test_forward_pass_predict_model(dummy_input, model_args):
    if model_args["n_drugs"] > 3:
        pytest.skip("Prediction model not supported for n_drugs > 3")
    _, model_predict = create_deep_model(**model_args)
    output = model_predict(dummy_input)
    assert isinstance(output, tf.Tensor)
    assert output.shape[0] == dummy_input.shape[0]
    assert output.shape[1] == model_args["n_doses"]

def test_model_with_drug_embeddings(dummy_input, model_args):
    weights = np.random.rand(model_args["drug_vocab_size"], model_args["drug_emb_dim"]).astype(np.float32)
    model_args["drug_embeddings_weights"] = weights

    model_train, _ = create_deep_model(**model_args)
    output = model_train(dummy_input)

    assert output.shape == (dummy_input.shape[0], model_args["n_drugs"],
                            model_args["rank"], model_args["n_doses"])

def test_run_model():
    import numpy as np
    from pipeline.model import create_deep_model  # adjust import if needed

    study_vocab_size = 5
    sample_vocab_size = 10
    drug_vocab_size = 20
    n_doses = 100
    n_drugs = 2  # set to >= 4 to test fallback behavior
    dummy_input = np.array([
        [0, 1, 3, 4, 5, 6],
        [1, 2, 4, 5, 6, 7],
        [2, 0, 2, 3, 8, 9],
        [3, 1, 1, 2, 7, 6],
    ], dtype=np.float32)

    model_train, model_predict = create_deep_model(
        study_vocab_size=study_vocab_size,
        sample_vocab_size=sample_vocab_size,
        drug_vocab_size=drug_vocab_size,
        n_doses=n_doses,
        n_drugs=n_drugs,
        emb_dim=64,
        drug_emb_dim=256,
        n_layers=2,
        units=64,
        rank=10,
        semi_window_size=3,
        gamma=0.2,
        drug_embeddings_weights=None,
        activation='relu'
    )

    print("\nModel (train) Summary:")
    model_train.summary()

    print("\nRunning training model...")
    output_train = model_train(dummy_input)
    print("Train output shape:", output_train.shape)

    if model_predict:
        print("\nRunning prediction model...")
        output_predict = model_predict(dummy_input)
        print("Predict output shape:", output_predict.shape)
    else:
        print("\nPrediction model not built (n_drugs >= 4)")


@pytest.mark.parametrize("n_drugs", [1, 2, 3])
def test_model_output_monotonicity(n_drugs):
    batch_size = 8
    n_doses = 20
    study_vocab_size = 5
    sample_vocab_size = 10
    drug_vocab_size = 50

    # Create dummy inputs:
    dummy_input = np.zeros((batch_size, 2 + 2 * n_drugs), dtype=np.float32)
    dummy_input[:, 0] = np.random.randint(0, study_vocab_size, size=batch_size)
    dummy_input[:, 1] = np.random.randint(0, sample_vocab_size, size=batch_size)
    for i in range(2 * n_drugs):
        dummy_input[:, 2 + i] = np.random.randint(0, drug_vocab_size, size=batch_size)

    # Build model
    model_train, model_predict = create_deep_model(
        study_vocab_size=study_vocab_size,
        sample_vocab_size=sample_vocab_size,
        drug_vocab_size=drug_vocab_size,
        n_doses=n_doses,
        n_drugs=n_drugs,
        emb_dim=32,
        drug_emb_dim=64,
        n_layers=1,
        units=32,
        rank=10,
        semi_window_size=3,
        gamma=0.2,
        drug_embeddings_weights=None,
        activation='relu'
    )

    # Run model
    output = model_predict(dummy_input).numpy()

    # Check output is within [0, 1]
    assert np.all((output >= 0.0) & (output <= 1.0)), f"Output out of [0, 1] range for n_drugs={n_drugs}"

    # Ensure output is decreasing with increasing doses
    if n_drugs == 1:
        # Shape: (batch_size, n_doses)
        assert output.shape == (batch_size, n_doses)
        for i in range(batch_size):
            diffs = np.diff(output[i], axis=0)
            assert np.all(diffs <= 1e-5), f"Output not monotonic for n_drugs=1\n{diffs}"
    elif n_drugs == 2:
        assert output.shape == (batch_size, n_doses, n_doses)
        # Shape: (batch_size, n_doses, n_doses)
        for i in range(batch_size):
            for axis in [0, 1]:
                diffs = np.diff(output[i], axis=axis)
                assert np.all(diffs <= 1e-5), f"Output not monotonic at batch {i} axis {axis}\n{diffs}"
    elif n_drugs == 3:
        assert output.shape == (batch_size, n_doses, n_doses, n_doses)
        # Shape: (batch_size, n_doses, n_doses, n_doses)
        for i in range(batch_size):
            for axis in [0, 1, 2]:
                diffs = np.diff(output[i], axis=axis)
                assert np.all(diffs <= 1e-5), f"Output not monotonic at batch {i} axis {axis}\n{diffs}"


@pytest.mark.parametrize("n_drugs", [2, 3])
def test_model_permutation_invariance(n_drugs):
    batch_size = 4
    n_doses = 20
    study_vocab_size = 5
    sample_vocab_size = 10
    drug_vocab_size = 50

    # Create dummy input
    dummy_input = np.zeros((batch_size, 2 + 2 * n_drugs), dtype=np.float32)
    dummy_input[:, 0] = np.random.randint(0, study_vocab_size, size=batch_size)
    dummy_input[:, 1] = np.random.randint(0, sample_vocab_size, size=batch_size)

    # Fill in drug IDs and doses
    for i in range(n_drugs):
        dummy_input[:, 2 + 2 * i] = np.random.randint(0, drug_vocab_size, size=batch_size)  # drug_id

    # Create permuted version (swap drugs 0 and 1, or rotate if 3)
    if n_drugs == 2:
        permuted_input = dummy_input.copy()
        # Swap drug1 <-> drug2 
        permuted_input[:, [2, 3, 4, 5]] = dummy_input[:, [3, 2, 5, 4]]
        axis_permutation = (0, 2, 1)
    elif n_drugs == 3:
        permuted_input = dummy_input.copy()
        # Rotate: drug1 → drug2, drug2 → drug3, drug3 → drug1
        permuted_input[:, [2, 3, 4, 5, 6, 7]] = dummy_input[:, [3, 4, 2, 6, 7, 5]]
        axis_permutation = (0, 3, 1, 2)

    # Build model
    model_train, model_predict = create_deep_model(
        study_vocab_size=study_vocab_size,
        sample_vocab_size=sample_vocab_size,
        drug_vocab_size=drug_vocab_size,
        n_doses=n_doses,
        n_drugs=n_drugs,
        emb_dim=32,
        drug_emb_dim=64,
        n_layers=1,
        units=32,
        rank=10,
        semi_window_size=3,
        gamma=0.2,
        drug_embeddings_weights=None,
        activation='relu'
    )

    # Run predictions
    out1 = model_predict(dummy_input).numpy()
    out2 = model_predict(permuted_input).numpy()

    # Check shape
    assert out1.shape == out2.shape

    # Check permutation of axes gives matching output
    out2_perm = np.transpose(out2, axes=axis_permutation)
    assert np.allclose(out1, out2_perm, atol=1e-5), f"Permutation invariance failed for n_drugs={n_drugs}"


@pytest.mark.parametrize("n_drugs", [1, 2, 3])
def test_vizu_model_pred(n_drugs):
    study_vocab_size = 5
    sample_vocab_size = 10
    drug_vocab_size = 20
    n_doses = 30
    batch_size = 8

    dummy_input = np.zeros((batch_size, 2 + 2 * n_drugs), dtype=np.float32)
    dummy_input[:, 0] = np.random.randint(0, study_vocab_size, size=batch_size)
    dummy_input[:, 1] = np.random.randint(0, sample_vocab_size, size=batch_size)
    for i in range(2 * n_drugs):
        dummy_input[:, 2 + i] = np.random.randint(0, drug_vocab_size, size=batch_size)

    model_train, model_predict = create_deep_model(
        study_vocab_size=study_vocab_size,
        sample_vocab_size=sample_vocab_size,
        drug_vocab_size=drug_vocab_size,
        n_doses=n_doses,
        n_drugs=n_drugs,
        emb_dim=64,
        drug_emb_dim=256,
        n_layers=2,
        units=64,
        rank=10,
        semi_window_size=3,
        gamma=0.2,
        drug_embeddings_weights=None,
        activation='relu'
    )

    if model_predict:
        print("\nRunning prediction model...")
        output_predict = model_predict(dummy_input).numpy()
        print("Predict output shape:", output_predict.shape)

        # Visualize prediction for the first sample
        if n_drugs == 1:
            plt.plot(output_predict[0])
            plt.title("Prediction (n_drugs=1)")
            plt.xlabel("Dose index")
            plt.savefig("tests/imgs/preds_ndrug1.png")
            plt.clf()

        elif n_drugs == 2:
            plt.imshow(output_predict[0], cmap='viridis')
            plt.title("Prediction Heatmap (n_drugs=2)")
            plt.xlabel("Dose 2 index")
            plt.ylabel("Dose 1 index")
            plt.colorbar()
            plt.savefig("tests/imgs/preds_ndrug2.png")
            plt.clf()

        elif n_drugs == 3:
            plt.imshow(output_predict[0][0], cmap='viridis')
            plt.title("Prediction Heatmap (n_drugs=3)")
            plt.xlabel("Dose 2 index")
            plt.ylabel("Dose 1 index")
            plt.colorbar()
            plt.savefig("tests/imgs/preds_ndrug3_0.png")
            plt.clf()

            plt.imshow(output_predict[0][n_doses // 2], cmap='viridis')
            plt.title("Prediction Heatmap (n_drugs=3)")
            plt.xlabel("Dose 2 index")
            plt.ylabel("Dose 1 index")
            plt.colorbar()
            plt.savefig("tests/imgs/preds_ndrug3_50.png")
            plt.clf()
    else:
        print("\nPrediction model not built (n_drugs >= 4)")