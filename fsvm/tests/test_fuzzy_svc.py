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
    assert clf.membership_base == "distance_to_class_center"
    assert clf.membership_decay == "exponential"
    assert clf.balanced is True

    clf.fit(X, y)
    assert hasattr(clf, "classes_")
    assert hasattr(clf, "X_")
    assert hasattr(clf, "y_")

    y_pred = clf.predict(X)
    assert y_pred.shape == (X.shape[0],)
