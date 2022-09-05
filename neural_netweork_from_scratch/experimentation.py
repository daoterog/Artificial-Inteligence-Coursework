"""
Experimentation module.
"""

import numpy as np

from layers import Layer
from neural_network import NeuralNetwork

from utils import train_model, plot_training_history

LEARNING_RATE = 0.01
MODEL_LOSS = "binary_cross_entropy"
N_ITER = 100000
BATCH_SIZE = 4

if __name__ == "__main__":

    # Read Data
    feature_matrix = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    targets = np.array([[0, 0, 0], [0, 1, 1], [0, 1, 1], [1, 1, 0]])

    # Create model
    model = NeuralNetwork(
        [
            Layer(feature_matrix.shape[1], 2, "sigmoid"),
            Layer(2, 2, "sigmoid"),
            Layer(2, targets.shape[1], "sigmoid"),
        ],
        LEARNING_RATE,
        MODEL_LOSS,
    )

    # Train model
    model, train_history = train_model(
        model, feature_matrix, targets, N_ITER, BATCH_SIZE
    )

    # Plot training history
    plot_training_history(train_history)
