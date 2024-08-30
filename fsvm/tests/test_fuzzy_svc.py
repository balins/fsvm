"""This file will just show how to write tests for the template classes."""

import numpy as np
import pytest
from numpy.testing import assert_almost_equal
from sklearn.datasets import load_iris

from fsvm import FuzzySVC


@pytest.fixture
def data():
    return load_iris(return_X_y=True)


def calculate_centroid_distance(samples, metric="euclidean"):
    if metric == "euclidean":
        centroid = np.mean(samples, axis=0)
    elif metric == "manhattan":
        centroid = np.median(samples, axis=0)
    else:
        raise ValueError("Invalid metric: {}".format(metric))

    return np.linalg.norm(centroid - samples, axis=1)


def test_smoke_test(data):
    X, y = data
    clf = FuzzySVC()

    # Check the default values
    assert clf.distance_metric == "centroid"
    assert clf.membership_decay == "linear"
    assert clf.beta == 0.1
    assert clf.balanced is True

    clf.fit(X, y)

    # Check the attributes
    assert hasattr(clf, "classes_")
    assert hasattr(clf, "X_")
    assert hasattr(clf, "y_")
    assert hasattr(clf, "distance_")
    assert hasattr(clf, "membership_degree_")

    y_pred = clf.predict(X)
    assert y_pred.shape == (X.shape[0],)


def test_centroid_exponential():
    class_0_samples = [[0, 0], [0, 1]]
    class_1_samples = [[2, 0], [2, 1]]
    X = class_0_samples + class_1_samples
    y = [0, 0, 1, 1]

    clf = FuzzySVC(
        distance_metric="centroid",
        membership_decay="exponential",
        centroid_metric="euclidean",
    )
    clf.fit(X, y)

    class_0_distance = calculate_centroid_distance(class_0_samples)
    class_1_distance = calculate_centroid_distance(class_1_samples)

    assert_almost_equal(
        clf.distance_, np.concatenate((class_0_distance, class_1_distance), axis=None)
    )

    membership_degree = 2 / (1 + np.exp(clf.beta * clf.distance_))

    assert_almost_equal(clf.membership_degree_, membership_degree)

    y_pred = clf.predict(X)
    assert (y_pred == y).all()


def test_centroid_linear():
    class_0_samples = [[0, 2], [2, 4]]
    class_1_samples = [[-2, -4], [0, -2]]
    X = class_0_samples + class_1_samples
    y = [0, 0, 1, 1]

    clf = FuzzySVC(
        distance_metric="centroid",
        membership_decay="linear",
        centroid_metric="euclidean",
    )
    clf.fit(X, y)

    class_0_distance = calculate_centroid_distance(class_0_samples)
    class_1_distance = calculate_centroid_distance(class_1_samples)

    assert_almost_equal(
        clf.distance_, np.concatenate((class_0_distance, class_1_distance), axis=None)
    )

    membership_degree = 1 - (clf.distance_ / (np.max(clf.distance_) + 1e-9))

    assert_almost_equal(clf.membership_degree_, membership_degree)

    y_pred = clf.predict(X)
    assert (y_pred == y).all()


def test_centroid_exponential_manhattan():
    class_0_samples = [[1, -3], [2, 2], [10, 4]]
    class_1_samples = [[-5, -5], [2, -8], [10, -9]]
    X = class_0_samples + class_1_samples
    y = [0, 0, 0, 1, 1, 1]

    class_0_distance = calculate_centroid_distance(class_0_samples, metric="manhattan")
    class_1_distance = calculate_centroid_distance(class_1_samples, metric="manhattan")

    clf = FuzzySVC(
        distance_metric="centroid",
        membership_decay="exponential",
        centroid_metric="manhattan",
    )
    clf.fit(X, y)

    assert_almost_equal(
        clf.distance_, np.concatenate((class_0_distance, class_1_distance), axis=None)
    )

    membership_degree = 2 / (1 + np.exp(clf.beta * clf.distance_))

    assert_almost_equal(clf.membership_degree_, membership_degree)

    y_pred = clf.predict(X)
    assert (y_pred == y).all()


def test_centroid_linear_manhattan():
    class_0_samples = [[1, -3], [2, 2], [10, 4]]
    class_1_samples = [[-5, -5], [2, -8], [10, -9]]

    X = class_0_samples + class_1_samples
    y = [0, 0, 0, 1, 1, 1]

    class_0_distance = calculate_centroid_distance(class_0_samples, metric="manhattan")
    class_1_distance = calculate_centroid_distance(class_1_samples, metric="manhattan")

    clf = FuzzySVC(
        distance_metric="centroid",
        membership_decay="linear",
        centroid_metric="manhattan",
    )
    clf.fit(X, y)

    assert_almost_equal(
        clf.distance_, np.concatenate((class_0_distance, class_1_distance), axis=None)
    )

    membership_degree = 1 - (clf.distance_ / (np.max(clf.distance_) + 1e-9))

    assert_almost_equal(clf.membership_degree_, membership_degree)

    y_pred = clf.predict(X)
    assert y_pred.shape == (len(y),)
