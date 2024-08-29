"""This file will just show how to write tests for the template classes."""

import pytest
from sklearn.datasets import load_iris

from fsvm import FuzzySVC


@pytest.fixture
def data():
    return load_iris(return_X_y=True)


def test_fuzzy_svc(data):
    """Check the internals and behaviour of `FuzzySVC`."""
    X, y = data
    clf = FuzzySVC()
    assert clf.distance_metric == "centroid"
    assert clf.membership_decay == "exponential"
    assert clf.beta == 1.0
    assert clf.balanced is True

    clf.fit(X, y)
    assert hasattr(clf, "classes_")
    assert hasattr(clf, "X_")
    assert hasattr(clf, "y_")
    assert hasattr(clf, "distance_")
    assert hasattr(clf, "membership_degree_")

    y_pred = clf.predict(X)
    assert y_pred.shape == (X.shape[0],)
