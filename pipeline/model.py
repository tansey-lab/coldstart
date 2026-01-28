import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Embedding, Lambda, Concatenate, Dense, Activation,
    LayerNormalization, Layer, Reshape
)
from tensorflow.keras.models import Model
from tensorflow.keras import Sequential


# ----------- Custom Layers -----------

class Integration(Layer):
    """Layer that computes the normalized reverse cumulative sum along the last axis."""
    def call(self, inputs):
        preds = tf.cumsum(inputs, axis=-1, reverse=True)
        preds /= tf.reduce_sum(inputs, axis=-1, keepdims=True)
        return preds

    def get_config(self):
        return super().get_config()


class Padding(Layer):
    """
    Layer for symmetric padding along the last axis.

    Args:
        semi_window_size (int): Number of padding elements on each side.
        left_value (float or None): Constant value or None to replicate edge for left pad.
        right_value (float or None): Constant value or None to replicate edge for right pad.
    """
    def __init__(self, semi_window_size=10, left_value=0., right_value=0., **kwargs):
        super().__init__(**kwargs)
        self.semi_window_size = semi_window_size
        self.left_value = left_value
        self.right_value = right_value

    def call(self, inputs):
        left_pad = tf.ones_like(inputs[..., :self.semi_window_size]) * (
            inputs[..., :1] if self.left_value is None else self.left_value
        )
        right_pad = tf.ones_like(inputs[..., :self.semi_window_size]) * (
            inputs[..., -1:] if self.right_value is None else self.right_value
        )
        return tf.concat([left_pad, inputs, right_pad], axis=-1)

    def get_config(self):
        config = super().get_config()
        config.update({
            "semi_window_size": self.semi_window_size,
            "left_value": self.left_value,
            "right_value": self.right_value
        })
        return config


class ColumnSplitter(Layer):
    """
    Splits input tensor into study, sample, SMILES drugs, and non-SMILES drugs columns.

    Args:
        n_drugs (int): Number of drugs per sample.
    """
    def __init__(self, n_drugs=3, **kwargs):
        super().__init__(**kwargs)
        self.n_drugs = n_drugs

    def call(self, inputs):
        part1 = tf.cast(inputs[:, :1], tf.int32)
        part2 = tf.cast(inputs[:, 1:2], tf.int32)
        part3 = tf.cast(inputs[:, 2:2 + self.n_drugs], tf.int32)
        part4 = tf.cast(inputs[:, 2 + self.n_drugs:2 + 2 * self.n_drugs], tf.int32)
        return part1, part2, part3, part4

    def get_config(self):
        config = super().get_config()
        config.update({"n_drugs": self.n_drugs})
        return config


# ----------- Utility Functions -----------

def get_gaussian_kernel(size=100, semi_window_size: int = 5, gamma: float = 0.2) -> tf.Tensor:
    """
    Generates a Gaussian smoothing kernel matrix.

    Args:
        size (int): Output size of the kernel.
        semi_window_size (int): Number of elements to each side for smoothing.
        gamma (float): Controls the spread of the Gaussian kernel.

    Returns:
        tf.Tensor: Gaussian kernel matrix of shape [size + 2 * semi_window_size, size]
    """
    window = np.arange(-semi_window_size, semi_window_size + 1).astype(np.float32)
    kernel = np.exp(-gamma * np.square(window))
    kernel /= np.sum(kernel)

    W = np.zeros((size, size + 2 * semi_window_size))
    for i in range(size):
        W[i, i:i + 2 * semi_window_size + 1] = kernel

    return tf.constant(W.T, dtype=tf.float32)


def dense_block(inputs, units=128, activation="relu", name_prefix="block"):
    """
    Standard dense block with normalization and activation.

    Args:
        inputs: Input tensor.
        units (int): Number of units in Dense layer.
        activation (str): Activation function.
        name_prefix (str): Prefix for layer names.

    Returns:
        Output tensor after dense block.
    """
    x = Dense(units, name=f"{name_prefix}_dense")(inputs)
    x = LayerNormalization(name=f"{name_prefix}_norm")(x)
    x = Activation(activation, name=f"{name_prefix}_act")(x)
    return x


def create_filter(size=100, semi_window_size=5, gamma=0.2) -> tf.keras.Sequential:
    """
    Creates a non-trainable Gaussian filter as a sequential model.

    Args:
        size (int): Target size of the output.
        semi_window_size (int): Number of smoothing elements on each side.
        gamma (float): Spread parameter for Gaussian kernel.

    Returns:
        tf.keras.Sequential: A Gaussian filter layer.
    """
    gaussian_conv = Sequential(name="gaussian_filter")
    gaussian_kernel = Dense(size, use_bias=False, name="gaussian_kernel")
    gaussian_kernel.build((None, size + 2 * semi_window_size))
    gaussian_kernel.kernel.assign(get_gaussian_kernel(size, semi_window_size, gamma))
    gaussian_kernel.trainable = False
    gaussian_conv.add(gaussian_kernel)
    return gaussian_conv


# ----------- Drug Processing -----------

def single_drug(embeddings, semi_window_size=5, gamma=0.2):
    """
    Processes a single drug embedding through smoothing and integration.

    Args:
        embeddings: Input tensor representing per-dose drug responses.
        semi_window_size (int): Number of elements on either side for smoothing.
        gamma (float): Spread parameter for Gaussian kernel.

    Returns:
        Tuple[tf.Tensor, tf.Tensor]: (Smoothed integral, smoothed derivative)
    """
    gaussian_kernel = create_filter(100, semi_window_size, gamma)

    derivative = Lambda(tf.abs, name="abs_derivative")(embeddings)
    derivative = Padding(semi_window_size, 0., 0., name="pad_derivative")(derivative)
    derivative = gaussian_kernel(derivative)

    integral = Integration(name="integrate_derivative")(derivative)
    integral = Padding(semi_window_size, 1., 0., name="pad_integral")(integral)
    integral = gaussian_kernel(integral)

    return integral, derivative


def combi_drug(embeddings, units=128, rank=20, semi_window_size=5, gamma=0.2, n_drugs=2, n_doses=100):
    """
    Applies a drug combination processing pipeline using kernel smoothing and integration.

    Args:
        embeddings: Input tensor representing drug embeddings per sample.
        units: Number of units in projection Dense layer.
        rank: Rank for factorization.
        semi_window_size: Size of the smoothing window.
        gamma: Spread parameter for Gaussian kernel.
        n_drugs: Number of drugs in each combination.
        n_doses: Number of dose points to model.

    Returns:
        A tensor of shape [batch_size, n_drugs, rank, n_doses] representing processed embeddings.
    """
    gaussian_kernel = create_filter(n_doses, semi_window_size, gamma)

    x = Dense(n_doses * rank, name="combi_proj")(embeddings)
    x = Reshape((rank, n_doses), name="combi_reshape")(x)
    x = Lambda(tf.abs, name="combi_abs")(x)

    x = Lambda(lambda x: tf.concat(tf.unstack(x, axis=1), axis=0), name="combi_unstack_rank")(x)
    x = Padding(semi_window_size, 0., 0., name="combi_pad_1")(x)
    x = gaussian_kernel(x)

    x = Integration(name="combi_integrate")(x)
    x = Padding(semi_window_size, 1., 0., name="combi_pad_2")(x)
    x = gaussian_kernel(x)

    x = Lambda(lambda x: tf.stack(tf.split(x, rank, axis=0), axis=1), name="combi_stack_rank")(x)
    x = Lambda(lambda x: tf.stack(tf.split(x, n_drugs, axis=0), axis=1), name="combi_stack_drugs")(x)

    return x


# ----------- Model Creation -----------

def create_deep_model(
    study_vocab_size: int,
    sample_vocab_size: int,
    drug_vocab_size: int,
    n_doses: int = 100,
    n_drugs: int = 1,
    emb_dim: int = 128,
    drug_emb_dim: int = 1024,
    n_layers: int = 3,
    units: int = 128,
    rank: int = 30,
    semi_window_size: int = 5,
    gamma: float = 0.2,
    drug_embeddings_weights=None,
    activation: str = "relu"
) -> tuple[tf.keras.Model, tf.keras.Model | None]:
    """
    Creates a deep learning model for multi-drug interaction prediction.

    Args:
        study_vocab_size: Vocabulary size for study embedding.
        sample_vocab_size: Vocabulary size for sample embedding.
        drug_vocab_size: Vocabulary size for drugs.
        n_doses: Number of possible dose levels (default: 100).
        n_drugs: Number of drugs per input sample.
        emb_dim: Dimensionality of learnable embeddings.
        drug_emb_dim: Dimensionality of fixed drug embeddings.
        n_layers: Number of dense layers after concatenation.
        units: Number of units per dense layer.
        rank: Latent rank for drug interaction modeling.
        semi_window_size: Window size for the smoothing kernel.
        gamma: Spread parameter for the Gaussian smoothing kernel.
        drug_embeddings_weights: Pretrained weights for SMILES embedding (optional).
        activation: Activation function to use in dense layers.

    Returns:
        model_train: Model that outputs intermediate drug interaction tensors.
        model_predict: Model that outputs final predictions (or None if n_drugs > 3).
    """
    input_layer = Input(shape=(2 + 2 * n_drugs,), dtype=tf.float32, name="input")
    input_study, input_sample, input_drug_smile, input_drug_no_smile = ColumnSplitter(n_drugs=n_drugs)(input_layer)

    # Embeddings
    embedding_sample = Embedding(sample_vocab_size, emb_dim, name="sample_embedding")
    embedding_study = Embedding(study_vocab_size, emb_dim, name="study_embedding")

    if drug_embeddings_weights is not None:
        embedding_smile = Embedding(
            drug_vocab_size, drug_emb_dim, name="drug_embedding_smile",
            weights=[tf.identity(drug_embeddings_weights)], trainable=False
        )
    else:
        embedding_smile = Embedding(drug_vocab_size, drug_emb_dim, name="drug_embedding_smile", trainable=False)

    embedding_no_smile = Embedding(drug_vocab_size, emb_dim, name="drug_embedding_no_smile")

    emb_sample = embedding_sample(input_sample)
    emb_study = embedding_study(input_study)
    emb_no_smile = embedding_no_smile(input_drug_no_smile)
    emb_smile = embedding_smile(input_drug_smile)

    emb_smile = Lambda(lambda x: tf.stop_gradient(x), name="stop_gradient")(emb_smile)
    # emb_smile = dense_block(emb_smile, drug_emb_dim, activation, name_prefix="smile_block1")
    emb_smile = dense_block(emb_smile, emb_dim, activation, name_prefix="smile_block2")

    emb_drug = Lambda(lambda x: x[0] + x[1], name="drug_combination")([emb_smile, emb_no_smile])

    emb_sample = Lambda(lambda x: tf.tile(x, [1, n_drugs, 1]), name="tile_sample")(emb_sample)
    emb_study = Lambda(lambda x: tf.tile(x, [1, n_drugs, 1]), name="tile_study")(emb_study)

    emb_interact = Lambda(lambda x: tf.tile(tf.reduce_prod(x, axis=1, keepdims=True), [1, n_drugs, 1]), name="drug_interact")(emb_drug)

    full_embedding = Concatenate(axis=-1, name="full_embedding")([emb_study, emb_sample, emb_drug, emb_interact])
    per_drug_embedding = Lambda(lambda x: tf.concat(tf.unstack(x, axis=1), axis=0), name="unstack_drugs")(full_embedding)
    per_drug_embedding = dense_block(per_drug_embedding, units, activation, name_prefix="embed_input")

    for i in range(n_layers):
        per_drug_embedding = dense_block(per_drug_embedding, units, activation, name_prefix=f"layer_{i}")

    combi_embedding = dense_block(per_drug_embedding, units, activation, name_prefix="combi_embed")

    integral_combi = combi_drug(
        combi_embedding, units=units, rank=rank,
        semi_window_size=semi_window_size, gamma=gamma, n_drugs=n_drugs, n_doses=n_doses
    )

    model_train = Model(input_layer, outputs=integral_combi, name="DeepDrugTrainModel")

    if n_drugs < 4:
        if n_drugs == 1:
            predictions = Lambda(
                lambda x: (1. / rank) * tf.reduce_sum(x[:, 0], axis=1),
                name="final_prediction"
            )(integral_combi)
        elif n_drugs == 2:
            predictions = Lambda(
                lambda x: (1. / rank) * tf.matmul(tf.transpose(x[:, 0], (0, 2, 1)), x[:, 1]),
                name="final_prediction"
            )(integral_combi)
        elif n_drugs == 3:
            predictions = Lambda(
                lambda x: (1. / rank) * tf.einsum('bti, btj, btk -> bijk', x[:, 0], x[:, 1], x[:, 2]),
                name="final_prediction"
            )(integral_combi)

        model_predict = Model(input_layer, outputs=predictions, name="DeepDrugPredictModel")
    else:
        model_predict = None

    return model_train, model_predict