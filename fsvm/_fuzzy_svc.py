"""
This is a module defining a fuzzy support vector machine classifier.
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, _fit_context, check_is_fitted
from sklearn.svm import SVC as _SVC
from sklearn.utils._param_validation import StrOptions
from sklearn.utils.multiclass import check_classification_targets


class FuzzySVC(ClassifierMixin, BaseEstimator):
    """A Fuzzy Support Vector Machine classifier.

    Parameters
    ----------
    membership_base : {'distance_to_class_center', 'distance_to_hyperplane'}   \
        or callable, default='distance_to_class_center'
        Method to compute the base for membership degree of training samples.
        The membership decay function will be then applied to its output
        to compute the actual membership degree value. If a callable is
        passed, it should take the input data `X` and return the membership
        degree values. For more information, see [1]_.

    membership_decay : {'exponential', 'linear'} or callable,                  \
        default='exponential'
        Method to compute the decay function for membership. If a callable is
        passed, it should take the output of `membership_base` method and
        return the final membership degree. Ignored if `membership_base` is a
        callable. For more information, see [1]_.

    balanced : bool, default=True
        Set the parameter C of class i to r_i*C for FuzzySVC. If True, the
        membership of each sample will be scaled by the number r_i expressing
        its ratio to the number of samples of the majority class.
        For more information, see [1]_.

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
    class_weight_ : ndarray of shape (n_classes,)
        Multipliers of parameter C for each class.
        Computed based on the ``class_weight`` parameter.

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
           2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
           2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])
    """

    _parameter_constraints = {
        "membership_base": [
            StrOptions({"distance_to_class_center", "distance_to_hyperplane"}),
            callable,
        ],
        "membership_decay": [StrOptions({"exponential", "linear"}), callable],
        "balanced": ["boolean"],
        **_SVC._parameter_constraints,
    }
    _parameter_constraints.pop("class_weight")

    _impl = "c__SVC"

    def __init__(
        self,
        *,
        membership_base="distance_to_class_center",
        membership_decay="exponential",
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
        self.membership_base = membership_base
        self.membership_decay = membership_decay
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
        self.classes_ = np.unique(y)
        self.X_ = X
        self.y_ = y

        # TODO: Implement the class balanicng, membership base and decay functions
        sample_weight = None

        self._SVC_ = _SVC(
            C=self.C,
            kernel=self.kernel,
            degree=self.degree,
            gamma=self.gamma,
            coef0=self.coef0,
            shrinking=self.shrinking,
            probability=self.probability,
            tol=self.tol,
            cache_size=self.cache_size,
            verbose=self.verbose,
            max_iter=self.max_iter,
            decision_function_shape=self.decision_function_shape,
            break_ties=self.break_ties,
            random_state=self.random_state,
        ).fit(X, y, sample_weight=sample_weight)

        self.class_weight_ = self._SVC_.class_weight_
        self.classes_ = self._SVC_.classes_
        self.dual_coef_ = self._SVC_.dual_coef_
        self.fit_status_ = self._SVC_.fit_status_
        self.intercept_ = self._SVC_.intercept_
        self.n_features_in_ = self._SVC_.n_features_in_
        self.n_iter_ = self._SVC_.n_iter_
        self.support_ = self._SVC_.support_
        self.support_vectors_ = self._SVC_.support_vectors_
        self.n_support_ = self._SVC_.n_support_
        self.probA_ = self._SVC_.probA_
        self.probB_ = self._SVC_.probB_
        self.shape_fit_ = self._SVC_.shape_fit_

        if hasattr(self._SVC_, "feature_names_in_"):
            self.feature_names_in_ = self._SVC_.feature_names_in_
        if hasattr(self._SVC_, "coef_"):
            self.coef_ = self._SVC_.coef_

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

        return self._SVC_.predict(X)
