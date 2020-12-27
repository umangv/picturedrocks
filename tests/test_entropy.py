import numpy as np
import picturedrocks as pr
import pytest
from scipy import linalg


def _mutual_info_to_joint_entropy(mutual_information, entropies):
    num_cols = mutual_information.shape[0]
    assert mutual_information.shape == (num_cols, num_cols)
    assert entropies.shape == (num_cols,)
    entropy_array = np.repeat(entropies[:, np.newaxis], num_cols, axis=1)
    return entropy_array + entropy_array.T - mutual_information


@pytest.fixture
def sampleX():
    X = np.array(
        # column indices:
        #    0  1  2  3  4  5
        [
            [0, 1, 0, 0, 0, 0],
            [0, 1, 0, 1, 0, 1],
            [0, 1, 0, 0, 1, 2],
            [0, 1, 0, 1, 1, 3],
            [0, 1, 1, 0, 2, 0],
            [0, 1, 1, 1, 2, 1],
            [0, 1, 1, 0, 3, 2],
            [0, 1, 1, 1, 3, 3],
        ]
    )
    entropies = np.array([0, 0, 1, 1, 2, 2])
    mutual_information = np.diag(entropies)
    # fill in non-zero entires in upper triangle
    mutual_information[2, 4] = 1
    mutual_information[3, 5] = 1
    mutual_information[4, 5] = 1
    # fill in non-zero entires in lower triangle
    mutual_information = np.maximum(mutual_information, mutual_information.T)
    return (X, entropies, mutual_information)


@pytest.fixture
def linearxy():
    """Linear XY dataset.
    
    Creates a matrix X of shape (n, n_feats), where n=10000 and n_feats=200.
    The 200 features are split into 20 groups of 10 each. Each group of 10
    features are highly correlated with each other (their covariance matrix
    has eigenvalues 5^0, 5^-1, ..., 5^-9).
    
    Among each feature group, a single feature is chosen as the "true"
    feature (there are 20 "true" features). The vector y is computed by
    projecting X onto the subspace of "true" features and dotting against a
    random vector.

    Returns
    -------
    a 3-tuple containing X, y, and the vector of "true" feature indices.
    """
    random = np.random.RandomState(0)
    n = 10000
    featgrp_size = 10
    n_featgrps = 20
    n_feats = n_featgrps * featgrp_size
    truefeats_rel = random.choice(featgrp_size, n_featgrps)
    truefeats_abs = featgrp_size * np.arange(n_featgrps) + truefeats_rel
    covar_blocks = []
    for _ in range(n_featgrps):
        q, r = np.linalg.qr(random.randn(featgrp_size, featgrp_size))
        S = q @ np.diag(5.0 ** (-np.arange(featgrp_size)))
        q1, r1 = np.linalg.qr(S)
        assert np.allclose(r1 - np.diag(np.diag(r1)), 0)
        assert np.allclose(np.abs(q1.T @ q), np.eye(10))
        covar_blocks.append(S)
    covar = linalg.block_diag(*covar_blocks)
    w = np.zeros((n_feats, 1))
    w[truefeats_abs, 0] = random.randn(n_featgrps)
    X = random.randn(n, n_feats) @ covar
    y = ((np.sign(X @ w) + 1) // 2).astype(int).flatten()
    return (X, y, truefeats_abs)


@pytest.mark.parametrize(
    "infoset_cls", [pr.markers.InformationSet, pr.markers.SparseInformationSet]
)
def test_entropies(infoset_cls, sampleX):
    X, entropies, _ = sampleX
    infoset = infoset_cls(X)
    H = [infoset.entropy(np.array([i])) for i in range(6)]
    assert np.allclose(H, entropies)


@pytest.mark.parametrize(
    "infoset_cls", [pr.markers.InformationSet, pr.markers.SparseInformationSet]
)
def test_entropies_wrt_empty(infoset_cls, sampleX):
    X, entropies, _ = sampleX
    infoset = infoset_cls(X)
    H = infoset.entropy_wrt(np.arange(0))
    assert np.allclose(H, entropies)
    assert np.issubdtype(H.dtype, np.float)


@pytest.mark.parametrize(
    "infoset_cls", [pr.markers.InformationSet, pr.markers.SparseInformationSet]
)
def test_joint_entropies(infoset_cls, sampleX):
    X, entropies, mutual_information = sampleX
    infoset = infoset_cls(X)
    _, num_cols = X.shape
    actual_joint_entropy = np.zeros((num_cols, num_cols))
    for i in range(num_cols):
        for j in range(num_cols):
            actual_joint_entropy[i, j] = infoset.entropy(np.array([i, j]))
    expected_joint_entropy = _mutual_info_to_joint_entropy(
        mutual_information, entropies
    )
    assert np.allclose(actual_joint_entropy, expected_joint_entropy)


@pytest.mark.parametrize(
    "infoset_cls", [pr.markers.InformationSet, pr.markers.SparseInformationSet]
)
def test_joint_entropies_using_wrt(infoset_cls, sampleX):
    X, entropies, mutual_information = sampleX
    infoset = infoset_cls(X)
    _, num_cols = X.shape
    actual_joint_entropy = np.zeros((num_cols, num_cols))
    for i in range(num_cols):
        actual_joint_entropy[i, :] = infoset.entropy_wrt(np.array([i]))
    expected_joint_entropy = _mutual_info_to_joint_entropy(
        mutual_information, entropies
    )
    assert np.allclose(actual_joint_entropy, expected_joint_entropy)


def test_consistent_joint_entropy_sparse_dense(sampleX):
    X, _, _ = sampleX
    infoset_sparse = pr.markers.SparseInformationSet(X)
    infoset_dense = pr.markers.InformationSet(X, False)
    assert np.allclose(
        infoset_sparse.entropy_wrt(np.arange(2)),
        infoset_dense.entropy_wrt(np.arange(2)),
    )
    assert np.allclose(
        infoset_sparse.entropy_wrt(np.arange(3)),
        infoset_dense.entropy_wrt(np.arange(3)),
    )


def test_entropy_wrt_consistent_with_entropy_non_discrete_dense(linearxy):
    X, y, _ = linearxy
    X_q = pr.markers.mutualinformation.infoset.quantile_discretize(X)
    infoset = pr.markers.InformationSet(
        np.concatenate((X_q, y[:, np.newaxis]), axis=1), True
    )
    assert np.allclose(
        infoset.entropy_wrt(np.arange(0)),
        np.array([infoset.entropy(np.array([i])) for i in range(X.shape[1])]),
    )
    assert np.allclose(
        infoset.entropy_wrt(np.array([-1])),
        np.array([infoset.entropy(np.array([-1, i])) for i in range(X.shape[1])]),
    )


def test_entropy_wrt_empty_consistent_with_entropy_non_discrete_sparse(linearxy):
    X, y, _ = linearxy
    X_q = pr.markers.mutualinformation.infoset.quantile_discretize(X)
    infoset = pr.markers.SparseInformationSet(X_q, y)
    assert np.allclose(
        infoset.entropy_wrt(np.arange(0)),
        np.array([infoset.entropy([i]) for i in range(X.shape[1])]),
    )


def test_entropy_wrt_consistent_with_entropy_non_discrete_sparse(linearxy):
    X, y, _ = linearxy
    X_q = pr.markers.mutualinformation.infoset.quantile_discretize(X)
    infoset = pr.markers.SparseInformationSet(X_q, y)
    assert np.allclose(
        infoset.entropy_wrt([-1]),
        np.array([infoset.entropy([-1, i]) for i in range(X.shape[1])]),
    )
