"""Fair-Test: A Flower / sklearn app."""

import numpy as np
from sklearn.linear_model import LogisticRegression
from fair_test.datasets.custom_dataset import Custom_Dataset
from fair_test.datasets.mnist import Mnist
from fair_test.datasets.compas import Compas

dataset = None  # Cache FederatedDataset
custom_dataset = Compas()


def load_data(partition_id: int, num_partitions: int, alpha: float):
    """
    Load partition data.
    Args:
        partition_id (int): The partition ID to load.
        num_partitions (int): The total number of partitions.
        alpha (float): The Dirichlet distribution parameter for partitioning.
    """
    # Only initialize `FederatedDataset` once
    global dataset
    global custom_dataset

    if dataset is None:
        dataset = custom_dataset.load(num_partitions, alpha, partition_id)

    feature_keys = custom_dataset.feature_keys
    #print("Feature keys: ", feature_keys)
    #print("features lenght: ", len(feature_keys))
 
    X = dataset[feature_keys]
    y =  dataset[custom_dataset.get_y()]
    #print("X shape: ", X.shape)
    #print("y shape: ", y.shape)
    #print("X sample values: ", X[:5])
    #print("y sample values: ", y[:5])
    # Split the on edge data: 80% train, 20% test
    X_train, X_test = X[: int(0.8 * len(X))], X[int(0.8 * len(X)) :]
    y_train, y_test = y[: int(0.8 * len(y))], y[int(0.8 * len(y)) :]

    return X_train, X_test, y_train, y_test


def get_model(penalty: str, local_epochs: int):

    return LogisticRegression(
        penalty=penalty,
        max_iter=local_epochs,
        warm_start=False,
        verbose=0
    )


def get_model_params(model):
    if model.fit_intercept:
        params = [
            model.coef_,
            model.intercept_,
        ]
    else:
        params = [model.coef_]
    return params


def set_model_params(model, params):
    model.coef_ = params[0]
    if model.fit_intercept:
        model.intercept_ = params[1]
    return model


def set_initial_params(model):
    n_classes = custom_dataset.get_n_classes()
    n_features = custom_dataset.get_n_features()
    model.classes_ = np.array([i for i in range(n_classes)])

    model.coef_ = np.zeros((n_classes, n_features))
    if model.fit_intercept:
        model.intercept_ = np.zeros((n_classes,))
