"""
This is a module defining a fuzzy support vector machine classifier.
"""

from numbers import Real

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, _fit_context, check_is_fitted
from sklearn.neighbors import (
    NearestCentroid as _NearestCentroid,
)
from sklearn.svm import SVC as _SVC
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import column_or_1d


class FuzzySVC(ClassifierMixin, BaseEstimator):
    """A Fuzzy Support Vector Machine classifier.

    Parameters
    ----------
    distance_metric : {'centroid', 'hyperplane'} or callable,                  \
        default='centroid'
        Method to compute the distance of a sample to the expected value,
        which will be the base for calculating the membership degree.
        The membership decay function will be then applied to its output
        to compute the actual membership degree value as in [1]_.

        If `distance_metric='centroid'`, the fuzzy membership is calculated
        based on the distance of the sample to the centroid of its class.
        If `distance_metric='hyperplane'`, the fuzzy membership is calculated
        based on the distance of the sample to the hyperplane of a pre-trained
        SVM classifier.
        If a callable is passed, it should take the input data `X` and return
        values of a custom metric for membership base, eg. the duration since
        a sample was collected. Notice that the membership degree of samples
        with larger values of `distance_metric` values will be lower.

    centroid_metric : {'euclidean', 'manhattan'}, default='euclidean'          \
        Metric to use for the computation of centroids of each class.
        This parameter is only used when `distance_metric='centroid'`.

        If `centroid_metric='euclidean'`, the centroid for the samples
        corresponding to each class is the arithmetic mean, which minimizes
        the sum of squared L1 distances.
        If `centroid_metric='manhattan'`, the centroid is the feature-wise
        median, which minimizes the sum of L1 distances.

    membership_decay : {'exponential', 'linear'} or callable,                  \
        default='linear'
        Method to compute the decay function for membership as in [1]_.
        If a callable is passed, it should take the output of `distance_metric`
        method and return the final membership degree in the interval [0, 1].

    beta : float, default=0.1
        Parameter for the exponential decay function, determining the steepness
        of the decay as in [1]._ Should be in the interval [0, 1].
        This parameter is only used when `membership_decay='exponential'`.

    balanced : bool, default=True
        Whether to use the values of y to automatically adjust weights
        inversely proportional to class frequencies in the input data as
        ``n_samples / (n_classes * np.bincount(y))``.

    C : float, default=1.0
        Regularization parameter. The strength of the regularization is
        inversely proportional to C. Must be strictly positive. The penalty
        is a squared l2 penalty.

    kernel : {'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'} or callable,  \
        default='rbf'
        Specifies the kernel type to be used in the algorithm. If
        none is given, 'rbf' will be used. If a callable is given it is used to
        pre-compute the kernel matrix from data matrices; that matrix should be
        an array of shape ``(n_samples, n_samples)``.

    degree : int, default=3
        Degree of the polynomial kernel function ('poly').
        Must be non-negative. Ignored by all other kernels.

    gamma : {'scale', 'auto'} or float, default='scale'
        Kernel coefficient for 'rbf', 'poly' and 'sigmoid'.

        - if ``gamma='scale'`` (default) is passed then it uses
          1 / (n_features * X.var()) as value of gamma,
        - if 'auto', uses 1 / n_features
        - if float, must be non-negative.

    coef0 : float, default=0.0
        Independent term in kernel function.
        It is only significant in 'poly' and 'sigmoid'.

    shrinking : bool, default=True
        Whether to use the shrinking heuristic.

    probability : bool, default=False
        Whether to enable probability estimates. This must be enabled prior
        to calling `fit`, will slow down that method as it internally uses
        5-fold cross-validation, and `predict_proba` may be inconsistent with
        `predict`.

    tol : float, default=1e-3
        Tolerance for stopping criterion.

    cache_size : float, default=200
        Specify the size of the kernel cache (in MB).

    verbose : bool, default=False
        Enable verbose output. Note that this setting takes advantage of a
        per-process runtime setting in libsvm that, if enabled, may not work
        properly in a multithreaded context.

    max_iter : int, default=-1
        Hard limit on iterations within solver, or -1 for no limit.

    decision_function_shape : {'ovo', 'ovr'}, default='ovr'
        Whether to return a one-vs-rest ('ovr') decision function of shape
        (n_samples, n_classes) as all other classifiers, or the original
        one-vs-one ('ovo') decision function of libsvm which has shape
        (n_samples, n_classes * (n_classes - 1) / 2). However, note that
        internally, one-vs-one ('ovo') is always used as a multi-class strategy
        to train models; an ovr matrix is only constructed from the ovo matrix.
        The parameter is ignored for binary classification.

    break_ties : bool, default=False
        If true, ``decision_function_shape='ovr'``, and number of classes > 2,
        :term:`predict` will break ties according to the confidence values of
        :term:`decision_function`; otherwise the first class among the tied
        classes is returned. Please note that breaking ties comes at a
        relatively high computational cost compared to a simple predict.

    random_state : int, RandomState instance or None, default=None
        Controls the pseudo random number generation for shuffling the data for
        probability estimates. Ignored when `probability` is False.
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    Attributes
    ----------
    distance_: ndarray of shape (n_samples,)
        Calculated distance of each sample to the expected value according to
        the `distance_metric` parameter.

    membership_degree_: ndarray of shape (n_samples,)
        Calculated membership degree of each sample according to the their
        `distance_metric` and `membership_decay`.

    class_weight_ : ndarray of shape (n_classes,)
        Multipliers of parameter C for each class based on class imbalance.

    classes_ : ndarray of shape (n_classes,)
        The classes labels.

    coef_ : ndarray of shape (n_classes * (n_classes - 1) / 2, n_features)
        Weights assigned to the features (coefficients in the primal
        problem). This is only available in the case of a linear kernel.

        `coef_` is a readonly property derived from `dual_coef_` and
        `support_vectors_`.

    dual_coef_ : ndarray of shape (n_classes -1, n_SV)
        Dual coefficients of the support vector in the decision
        function, multiplied by their targets.
        For multiclass, coefficient for all 1-vs-1 classifiers.
        The layout of the coefficients in the multiclass case is somewhat
        non-trivial.

    fit_status_ : int
        0 if correctly fitted, 1 otherwise (will raise warning)

    intercept_ : ndarray of shape (n_classes * (n_classes - 1) / 2,)
        Constants in decision function.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

    n_iter_ : ndarray of shape (n_classes * (n_classes - 1) // 2,)
        Number of iterations run by the optimization routine to fit the model.
        The shape of this attribute depends on the number of models optimized
        which in turn depends on the number of classes.

    support_ : ndarray of shape (n_SV)
        Indices of support vectors.

    support_vectors_ : ndarray of shape (n_SV, n_features)
        Support vectors. An empty array if kernel is precomputed.

    n_support_ : ndarray of shape (n_classes,), dtype=int32
        Number of support vectors for each class.

    probA_ : ndarray of shape (n_classes * (n_classes - 1) / 2)
    probB_ : ndarray of shape (n_classes * (n_classes - 1) / 2)
        If `probability=True`, it corresponds to the parameters learned in
        Platt scaling to produce probability estimates from decision values.
        If `probability=False`, it's an empty array. Platt scaling uses the
        logistic function
        ``1 / (1 + exp(decision_value * probA_ + probB_))``
        where ``probA_`` and ``probB_`` are learned from the dataset [3]_. For
        more information on the multiclass case and training procedure see
        section 8 of [2]_.

    shape_fit_ : tuple of int of shape (n_dimensions_of_X,)
        Array dimensions of training vector ``X``.

    References
    ----------
    .. [1] `Batuwita, R., Palade, V. (2010). "FSVM-CIL: Fuzzy Support Vector Machines
        for Class Imbalance Learning" <https://doi.org/10.1109/TFUZZ.2010.2042721>`_
    .. [2] `LIBSVM: A Library for Support Vector Machines
        <http://www.csie.ntu.edu.tw/~cjlin/papers/libsvm.pdf>`_
    .. [3] `Platt, John (1999). "Probabilistic Outputs for Support Vector
        Machines and Comparisons to Regularized Likelihood Methods"
        <https://citeseerx.ist.psu.edu/doc_view/pid/42e5ed832d4310ce4378c44d05570439df28a393>`_

    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from fsvm import FuzzySVC
    >>> X, y = load_iris(return_X_y=True)
    >>> clf = FuzzySVC().fit(X, y)
    >>> clf.predict(X)
    array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
           1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1,
           1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2,
           2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2,
           2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])
    """

    _parameter_constraints = {
        "distance_metric": [
            StrOptions({"centroid", "hyperplane"}),
            callable,
        ],
        "centroid_metric": [StrOptions({"euclidean", "manhattan"})],
        "membership_decay": [StrOptions({"exponential", "linear"}), callable],
        "beta": [Interval(Real, 0.0, 1.0, closed="both")],
        "balanced": ["boolean"],
        **_SVC._parameter_constraints,
    }
    _parameter_constraints.pop("class_weight")

    _impl = "c__SVC"

    def __init__(
        self,
        *,
        distance_metric="centroid",
        centroid_metric="euclidean",
        membership_decay="linear",
        beta=0.1,
        balanced=True,
        C=1.0,
        kernel="rbf",
        degree=3,
        gamma="scale",
        coef0=0.0,
        shrinking=True,
        probability=False,
        tol=1e-3,
        cache_size=200,
        verbose=False,
        max_iter=-1,
        decision_function_shape="ovr",
        break_ties=False,
        random_state=None,
    ):
        self.distance_metric = distance_metric
        self.centroid_metric = centroid_metric
        self.membership_decay = membership_decay
        self.beta = beta
        self.balanced = balanced
        self.C = C
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.shrinking = shrinking
        self.probability = probability
        self.tol = tol
        self.cache_size = cache_size
        self.verbose = verbose
        self.max_iter = max_iter
        self.decision_function_shape = decision_function_shape
        self.break_ties = break_ties
        self.random_state = random_state

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y):
        """Fit the FSVM model according to the given training data.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features) \
                or (n_samples, n_samples)
            Training vectors, where `n_samples` is the number of samples
            and `n_features` is the number of features.
            For kernel="precomputed", the expected shape of X is
            (n_samples, n_samples).

        y : array-like of shape (n_samples,)
            Target values (class labels in classification, real numbers in
            regression).

        Returns
        -------
        self : object
            Fitted estimator.

        Notes
        -----
        If X and y are not C-ordered and contiguous arrays of np.float64 and
        X is not a scipy.sparse.csr_matrix, X and/or y may be copied.

        If X is a dense array, then the other methods will not support sparse
        matrices as input.
        """
        X, y = self._validate_data(X, y)
        check_classification_targets(y)
        self.X_ = X
        self.y_ = y

        y_ = column_or_1d(y, warn=True)
        classes, y_ = np.unique(y_, return_inverse=True)

        svc_args = {
            "C": self.C,
            "kernel": self.kernel,
            "degree": self.degree,
            "gamma": self.gamma,
            "coef0": self.coef0,
            "shrinking": self.shrinking,
            "probability": self.probability,
            "tol": self.tol,
            "cache_size": self.cache_size,
            "class_weight": "balanced" if self.balanced else None,
            "verbose": self.verbose,
            "max_iter": self.max_iter,
            "decision_function_shape": self.decision_function_shape,
            "break_ties": self.break_ties,
            "random_state": self.random_state,
        }

        if self.distance_metric == "centroid":
            centroids = (
                _NearestCentroid(metric=self.centroid_metric).fit(X, y).centroids_
            )
            self.distance_ = np.linalg.norm(X - centroids[y_], axis=1)
        elif self.distance_metric == "hyperplane":
            hyperplane_svc_args = {**svc_args, "decision_function_shape": "ovr"}
            svc = _SVC(**hyperplane_svc_args).fit(X, y)
            decision_function_output = svc.decision_function(X)

            # For multiclass, extract the distances corresponding
            # to the true class labels
            if decision_function_output.ndim > 1:
                y_indices = np.array(
                    [svc.classes_.tolist().index(class_label) for class_label in y]
                )
                decision_function_output = decision_function_output[
                    np.arange(len(y)), y_indices
                ]

            self.distance_ = np.abs(decision_function_output)
        elif callable(self.distance_metric):
            self.distance_ = self.distance_metric(X)

        self.membership_degree_ = self.__calculate_membership_degree()

        self.svc_ = _SVC(**svc_args).fit(X, y, sample_weight=self.membership_degree_)

        self.class_weight_ = self.svc_.class_weight_
        self.classes_ = self.svc_.classes_
        self.dual_coef_ = self.svc_.dual_coef_
        self.fit_status_ = self.svc_.fit_status_
        self.intercept_ = self.svc_.intercept_
        self.n_features_in_ = self.svc_.n_features_in_
        self.n_iter_ = self.svc_.n_iter_
        self.support_ = self.svc_.support_
        self.support_vectors_ = self.svc_.support_vectors_
        self.n_support_ = self.svc_.n_support_
        self.probA_ = self.svc_.probA_
        self.probB_ = self.svc_.probB_
        self.shape_fit_ = self.svc_.shape_fit_

        if hasattr(self.svc_, "feature_names_in_"):
            self.feature_names_in_ = self.svc_.feature_names_in_
        if hasattr(self.svc_, "coef_"):
            self.coef_ = self.svc_.coef_

        return self

    def predict(self, X):
        """A reference implementation of a prediction for a classifier.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples.
        Returns
        -------
        y : ndarray, shape (n_samples,)
            The label for each sample is the label of the closest sample
            seen during fit.
        """
        check_is_fitted(self)

        X = self._validate_data(X, reset=False)

        return self.svc_.predict(X)

    def __calculate_membership_degree(self):
        if self.membership_decay == "exponential":
            membership = 2 / (1 + np.exp(self.beta * self.distance_))
        elif self.membership_decay == "linear":
            max_distance = np.amax(self.distance_)
            delta = 1e-9
            membership = 1 - (self.distance_ / (max_distance + delta))
        elif callable(self.membership_decay):
            membership = self.membership_decay(self.distance_)
            if not (membership >= 0) & (membership <= 1):
                raise ValueError(
                    "Membership decay function must return values in range [0, 1]."
                )

        return membership
