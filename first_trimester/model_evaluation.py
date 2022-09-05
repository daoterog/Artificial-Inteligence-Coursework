"""
Model evaluation module.
"""

import numpy as np
import operator as op
from functools import reduce

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

from data_management import sample_data

def evaluate_model(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict:

    """Obtains train and test error for a given model."""

    model.fit(X_train, y_train)

    n_train = X_train.shape[0]
    n_val = X_val.shape[0]
    n_test = X_test.shape[0]

    y_pred_train = model.predict(X_train)
    y_pred_val = model.predict(X_val)
    y_pred_test = model.predict(X_test)

    target_labels = np.unique(y_test)

    labels_errors = {}

    for label in target_labels:

        # C_i
        train_true = set(np.where(y_train == label)[0].tolist())
        val_true = set(np.where(y_val == label)[0].tolist())
        test_true = set(np.where(y_test == label)[0].tolist())

        # h_i
        train_pred = set(np.where(y_pred_train == label)[0].tolist())
        val_pred = set(np.where(y_pred_val == label)[0].tolist())
        test_pred = set(np.where(y_pred_test == label)[0].tolist())

        # C_i intersected h_i
        train_true_pred_inters = train_true.intersection(train_pred)
        val_true_pred_inters = val_true.intersection(val_pred)
        test_true_pred_inters = test_true.intersection(test_pred)

        # C_i union h_i
        train_true_pred_union = train_true.union(train_pred)
        val_true_pred_union = val_true.union(val_pred)
        test_true_pred_union = test_true.union(test_pred)

        # C_i union h_i - C_i intersected h_i
        train_union_minus_intersection = train_true_pred_union.difference(train_true_pred_inters)
        val_union_minus_intersection = val_true_pred_union.difference(val_true_pred_inters)
        test_union_minus_intersection = test_true_pred_union.difference(test_true_pred_inters)

        # Errors
        train_error = len(train_union_minus_intersection) / n_train
        val_error = len(val_union_minus_intersection) / n_val
        test_error = len(test_union_minus_intersection) / n_test

        labels_errors[label] = (train_error, val_error, test_error)

    return labels_errors


def evaluate_classifiers(
    classifiers_dict: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict:

    """Evaluates all classifiers in a dictionary."""

    classifiers_errors = {}

    for classifier_name, classifier in classifiers_dict.items():
        classifiers_errors[classifier_name] = evaluate_model(
            classifier, X_train, y_train, X_val, y_val, X_test, y_test
        )

        # Print results
        print(f"{classifier_name}")
        for label, error in classifiers_errors[classifier_name].items():
            print(f"Label {label} -> Train error {error[0]}, Val error {error[1]}, Test error {error[2]}")

    return classifiers_errors


def n_combined_r(n: int, r: int):

    """Returns the number of combinations of n and r."""

    r = min(r, n - r)
    numer = reduce(op.mul, range(n, n - r, -1), 1)
    denom = reduce(op.mul, range(1, r + 1), 1)
    return numer // denom


def number_of_samples(vc_dim: int, gen_error: float, train_error: float):

    """Give the probably approximated correct learning guarantee for a vc dimension (of hypothesis
    family), a generalization error, and a train error."""

    return (np.log(vc_dim) + np.log(1 / train_error)) / gen_error


def number_of_samples_decision_tree(
    depth: int, gen_error: float, train_error: float, n_features: int
):

    """Give the probably approximated correct learning guarantee for a given depth, a generalization
    error, a train error, and a number of features."""

    return (
        np.log(2)
        * (
            (np.power(2, depth) - 1) * (1 + np.log2(n_features))
            + 1
            + np.log(1 / train_error)
        )
        / (2 * (gen_error**2))
    )


def find_vc_dim(hypthesis_family: str, n_features: int, pol_degree: int = None):

    """Returns vc dimension of a given hypothesis family and number of features."""

    if hypthesis_family == "linear":
        return n_features + 1
    elif hypthesis_family == "polynomial":
        if pol_degree is None:
            raise ValueError("Polynomial degree not specified.")
        return n_combined_r(n_features + pol_degree - 1, pol_degree)
    elif hypthesis_family == "rbf":
        return np.inf
    else:
        raise ValueError(
            "Unknown hypothesis family. Available hypothesis families are decision_tree, linear, polynomial, and rbf."
        )


def get_number_of_samples(
    hypthesis_family: str,
    n_features: int,
    gen_error: float,
    train_error: float,
    pol_degree: int = None,
    depth: int = None,
):

    """Returns the number of samples needed to guarantee correct learning for a given hypothesis family,
    number of features, generalization error, and train error."""

    if hypthesis_family == "decision_tree":
        if depth is None:
            raise ValueError("Depth not specified.")
        return number_of_samples_decision_tree(
            depth, gen_error, train_error, n_features
        )
    return number_of_samples(
        find_vc_dim(hypthesis_family, n_features, pol_degree), gen_error, train_error
    )


def get_model(model_type: str, depth: int = None, pol_degree: int = None) -> tuple:

    """Returns model given certain parameters and the hypothesis family to which it belongs."""

    if model_type == "decision_tree":
        if depth is None:
            raise ValueError("Depth not specified.")
        model = DecisionTreeClassifier(max_depth=depth)
        hypothesis_family = "decision_tree"
    elif model_type == "svm_linear":
        model = SVC(kernel="linear")
        hypothesis_family = "linear"
    elif model_type == "svm_polynomial":
        if pol_degree is None:
            raise ValueError("Polynomial degree not specified.")
        model = SVC(kernel="poly", degree=pol_degree)
        hypothesis_family = "polynomial"
    elif model_type == "svm_rbf":
        model = SVC(kernel="rbf")
        hypothesis_family = "rbf"
    elif model_type == "logistic_regression":
        model = LogisticRegression()
        hypothesis_family = "linear"
    else:
        raise ValueError(
            "Unknown model type. Available model types are decision_tree, svm_linear, svm_polynomial, svm_rbf, and logistic_regression."
        )

    return model, hypothesis_family


def get_sample_percentage(requiered_samples: int, available_samples: int) -> float:

    """Returns trains percentage necesary to obtain n_samples (and satisfy PAG)."""

    if requiered_samples == np.inf:
        print("Infinite samples required, will return standard 60%.")
        return 0.6
    elif requiered_samples > available_samples:
        print(
            f"Required samples: {int(np.ceil(requiered_samples))}, available samples: {available_samples}."
        )
        print("Not enough samples available, will return standard 60%.")
        return 0.6
    elif requiered_samples > np.floor(available_samples * 0.6):
        print(
            f"Required samples: {int(np.ceil(requiered_samples))}, available samples: {available_samples}."
        )
        print("Required samples are greater than the 60%, will return standard 60%.")
    else:
        print(f"{requiered_samples *100 / available_samples}% are required.")
        return requiered_samples / available_samples


def evaluation_pipeline(
    feature_matrix: np.ndarray,
    model_type: str,
    gen_error: float,
    train_error: float,
    distribution: str = "normal",
    depth: int = None,
    pol_degree: int = None,
):

    """Evaluation pipeline."""

    # Get model and hypothesis family
    model, hypothesis_family = get_model(model_type, depth, pol_degree)

    # Calculate necessary parameters
    available_samples = feature_matrix.shape[0]
    n_features = feature_matrix.shape[1]

    # Get number of samples
    required_samples = get_number_of_samples(
        hypothesis_family,
        n_features,
        gen_error,
        train_error,
        pol_degree,
        depth,
    )

    # Get percentage of samples
    sample_percentage = get_sample_percentage(required_samples, available_samples)

    # Sample and split data
    train_array, test_array = sample_data(feature_matrix, distribution, sample_percentage)

    val_array, test_array = sample_data(test_array, distribution, 0.5)

    X_train = train_array[:, :-1]
    y_train = train_array[:, -1]

    X_val = val_array[:, :-1]
    y_val = val_array[:, -1]

    X_test = test_array[:, :-1]
    y_test = test_array[:, -1]

    print(f"{y_train.shape[0]} of the data are used for training.")
    print(f"{y_val.shape[0]} of the data are used for validation.")
    print(f"{y_test.shape[0]} of the data are used for testing.")

    # Evaluate model
    classifiers_dict = {model_type: model}

    classifiers_errors = evaluate_classifiers(
        classifiers_dict, X_train, y_train, X_val, y_val, X_test, y_test
    )

    return classifiers_errors


