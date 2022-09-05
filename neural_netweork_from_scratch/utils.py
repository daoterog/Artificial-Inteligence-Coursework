"""
Utils module.
"""

import os
import typing as t
from collections import OrderedDict
from csv import reader

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from scipy.stats import norm, nbinom

from neural_network import NeuralNetwork


def get_batch(
    feature_matrix: np.ndarray, targets: np.ndarray, batch_size: int
) -> t.Tuple[np.ndarray, np.ndarray]:
    """Returns a batch of the feature matrix and targets.
    Args:
        feature_matrix (np.ndarray): Transposed feature matrix.
        targets (np.ndarray): Transposed targets.
    Returns:
        t.Tuple[np.ndarray, np.ndarray]: Batch of the feature matrix and targets.
    """
    for i in range(0, feature_matrix.shape[1], batch_size):
        yield feature_matrix[:, i : i + batch_size], targets[:, i : i + batch_size]


def train_model(
    model: NeuralNetwork,
    feature_matrix: np.ndarray,
    targets: np.ndarray,
    n_iter: int,
    batch_size: int,
    verbose: bool = False,
) -> t.Tuple[NeuralNetwork, OrderedDict]:
    """Trains the model for a given number of epochs and returns the training history.
    Args:
        model (NeuralNetwork): Neural network model.
        feature_matrix (np.ndarray): Feature matrix.
        targets (np.ndarray): Targets.
        n_iter (int): Number of iterations.
        batch_size (int): Batch size.
        verbose (bool, optional): Whether to print the loss. Defaults to False.
    Returns:
        t.Tuple[NeuralNetwork, OrderedDict]: Trained model and training history.
    """
    num_epochs = int(np.ceil(n_iter / (feature_matrix.shape[0] / batch_size)))
    train_history = OrderedDict()
    iter_num = 0
    for epoch in range(num_epochs):
        loss_list = []
        for batch_inputs, batch_targets in get_batch(
            feature_matrix.T, targets.T, batch_size
        ):
            # Forward pass
            _ = model.forward(batch_inputs)
            # Compute loss
            loss = model.get_loss(batch_targets)
            loss_list.append(loss)
            # Backward pass
            model.backward(batch_targets)
            # Update parameters
            model.step()
            # Get grads
            weight_grads, bias_grads = model.get_grads()
            # Store state
            train_history[iter_num] = {
                "loss": loss,
                "weight_grads": weight_grads,
                "bias_grads": bias_grads,
            }
            iter_num += 1
        if verbose:
            print(f"Epoch {epoch + 1}/{num_epochs} | Loss: {np.mean(loss_list)}")
    return model, train_history


def unstack_grads(
    grads: t.Tuple[t.Dict[int, np.ndarray]]
) -> t.Tuple[t.Dict[int, t.List[np.ndarray]], t.Dict[int, t.List[float]]]:
    """Unstacks the gradients.
    Args:
        grads (t.Tuple[t.Dict[int, np.ndarray]]): Gradients.
    Returns:
        t.Tuple[t.Dict[int, t.List[np.ndarray]], t.Dict[int, t.List[float]]]: Tuple of unstacked
            gradients, the first one containing the whole gradient matrices and the second one the
            mean of the gradients.
    """
    layer_grads = OrderedDict()
    layer_mean_grads = OrderedDict()
    for iter_grads in grads:
        for layer, grad in iter_grads.items():
            if layer not in layer_grads:
                layer_grads[layer] = []
                layer_mean_grads[layer] = []
            layer_grads[layer].append(grad)
            layer_mean_grads[layer].append(np.mean(grad))
    return layer_grads, layer_mean_grads


def stack_mean_gradients(
    mean_grads: t.Dict[int, t.List[float]]
) -> t.Dict[int, np.ndarray]:
    """Stacks the mean gradients.
    Args:
        mean_grads (t.Dict[int, t.List[float]]): Mean gradients.
    Returns:
        t.Dict[int, np.ndarray]: Stacked mean gradients.
    """
    return np.stack(list(mean_grads.values())).T


def plot_training_history(train_history: OrderedDict) -> None:
    """Plots the training history.
    Args:
        train_history (OrderedDict): Training history.
    """

    # Unzip items
    iters, loss, weight_grads, bias_grads = zip(
        *[
            (k, v["loss"], v["weight_grads"], v["bias_grads"])
            for k, v in train_history.items()
        ]
    )

    # Get mean loss
    mean_loss = [np.mean(loss_i) for loss_i in loss]

    # Unstack gradients
    _, weight_layer_mean_grads = unstack_grads(weight_grads)
    _, bias_layer_mean_grads = unstack_grads(bias_grads)

    # Stack mean gradients
    weight_stacked_mean_grads = stack_mean_gradients(weight_layer_mean_grads)
    bias_stacked_mean_grads = stack_mean_gradients(bias_layer_mean_grads)

    _, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(15, 3))

    ax0.plot(iters, mean_loss)
    ax0.set_title("Loss")
    ax0.set_xlabel("Iteration")
    ax0.set_ylabel("Loss")

    ax1.plot(iters, weight_stacked_mean_grads)
    ax1.set_title("Mean weight gradients")

    ax2.plot(iters, bias_stacked_mean_grads)
    ax2.set_title("Mean bias gradients")

    for ax in (ax1, ax2):
        ax.legend([f"Layer {i}" for i in range(len(bias_stacked_mean_grads[0]))])
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Mean gradient")

    plt.tight_layout()
    plt.show()

def readmatfile(filepath: str) -> np.ndarray:
    """Reads a .mat file.
    Args:
        filepath (str): filepath.
    Returns:
        np.ndarray: Data.
    """
    # Read matlab file
    mat = sio.loadmat(filepath)
    # Define column names
    columns = list(mat.keys() - ['__header__', '__version__', '__globals__'])
    # Iterate over columns length and extract the most repeated length
    all_n_rows = []
    for column in columns:
        n_rows = mat[column].shape[0]
        if n_rows != 1:
            all_n_rows.append(n_rows)
    row_len = np.unique(all_n_rows, return_counts=True)[0][0]
    # Store columns that have the most repeated length in a list and stack them into a ndarray
    col_list = []
    for column in columns:
        if mat[column].shape[0] == row_len:
            col_list.append(mat[column])
    return np.hstack(col_list)


def readfile(filename: str) -> np.ndarray:
    """Reads a csv file and returns a numpy array.
    Args:
        filename (str): Name of the file.
    Returns:
        np.ndarray: data.
    """
    filepath = os.path.join(os.getcwd(), "data", filename)
    if filename[-3:] == 'mat':
        return readmatfile(filepath)

    with open(filepath, "r", encoding="UTF-8") as file:
        csv_reader = reader(file)
        data = np.vstack(list(csv_reader)).astype(np.float)
    return data


def binary_search_percentile(
    random_number: float,
    distribution: str,
    lower_bound: float,
    upper_bound: float,
    tol: float,
) -> float:

    """Recieves a random number generated from a specific distribution and uses binary search to
    find the percentile to which it corresponds.

    Args:
        random_number (float): random number generated from a specific distribution.
        distribution (str): name of the used distribution.
        lower_bound (float): lower bound of the percentile search.
        upper_bound (float): upper bound of the percentile search.
        tol (float): tol of the percentile search.

    Returns:
        float: percentile of the random number."""

    middle_percentile = (lower_bound + upper_bound) / 2

    if distribution == "normal":
        percentile_number = norm.ppf(middle_percentile, loc=0, scale=1)
    else:
        percentile_number = nbinom.ppf(middle_percentile, n=1, p=0.1)

    if abs(random_number - percentile_number) < tol:
        return middle_percentile
    elif random_number > percentile_number:
        return binary_search_percentile(
            random_number, distribution, middle_percentile, upper_bound, tol
        )
    else:
        return binary_search_percentile(
            random_number, distribution, lower_bound, middle_percentile, tol
        )


def sample_data(
    feature_matrix: np.ndarray,
    distribution: str,
    train_size: float,
) -> t.Tuple[np.ndarray, np.ndarray]:

    """Samples the data according to certain probability distribution.

    Args:
        feature_matrix (np.ndarray): dataset to sample from.
        distribution (str): distribution to use to generate the random numbers.
        train_size (float): size of the training set.

    Returns:
        t.Tuple[np.ndarray, np.ndarray]: train and test datasets."""

    shuffled_array = shuffle(feature_matrix)

    n_iter = int(np.ceil(shuffled_array.shape[0] * train_size))

    if distribution == "normal":
        random_number_array = np.random.normal(0, 1, size=(n_iter,))
    elif distribution == "nbinom":
        random_number_array = np.random.negative_binomial(1, 0.1, size=(n_iter,))
    elif distribution == "uniform":
        percentile_list = np.random.uniform(0, 1, size=(n_iter,))
    else:
        raise ValueError(
            "Distribution not recognized. Available distributions are: normal, nbinom,"
            + " uniform"
        )

    # Find percentile for each random number
    if distribution == "normal" or distribution == "negative_binomial":
        percentile_list = [
            binary_search_percentile(random_number, distribution, 0.0, 1.0, 10e-3)
            for _, random_number in enumerate(random_number_array)
        ]

    train_samples = []

    # Draw samples from the shuffled array according to the percentiles
    for _, percentile in enumerate(percentile_list):
        sample_index = int(percentile * shuffled_array.shape[0])
        train_samples.append(shuffled_array[sample_index, :])
        shuffled_array = np.delete(shuffled_array, sample_index, axis=0)
        shuffled_array = shuffle(shuffled_array)

    train_array = np.vstack(train_samples)
    test_array = shuffled_array

    return train_array, test_array
