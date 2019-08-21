import numpy as np
import picturedrocks as pr
import pytest
from scipy import linalg


@pytest.fixture
def simpleX():
    return np.array(
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


@pytest.fixture
def linearxy():
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


def test_entropies(simpleX):
    infoset = pr.markers.SparseInformationSet(simpleX)
    H = infoset.entropy_wrt(np.arange(0))
    assert np.allclose(H, [0, 0, 1, 1, 2, 2])
    H_manual = [infoset.entropy(np.array([i])) for i in range(6)]
    assert np.allclose(H, H_manual)


def test_joint_entropies(simpleX):
    infoset = pr.markers.SparseInformationSet(simpleX)
    H = infoset.entropy_wrt(np.arange(0))
    H_0 = infoset.entropy_wrt(np.arange(1))
    H_01 = infoset.entropy_wrt(np.arange(2))
    assert np.allclose(H, H_0)
    assert np.allclose(H, H_01)


def test_sparse_dense(simpleX):
    infoset_sparse = pr.markers.SparseInformationSet(simpleX)
    infoset_dense = pr.markers.InformationSet(simpleX, False)
    assert np.allclose(
        infoset_sparse.entropy_wrt(np.arange(0)),
        infoset_dense.entropy_wrt(np.arange(0)),
    )
    assert np.allclose(
        infoset_sparse.entropy_wrt(np.arange(1)),
        infoset_dense.entropy_wrt(np.arange(1)),
    )
    assert np.allclose(
        infoset_sparse.entropy_wrt(np.arange(2)),
        infoset_dense.entropy_wrt(np.arange(2)),
    )
    assert np.allclose(
        infoset_sparse.entropy_wrt(np.arange(3)),
        infoset_dense.entropy_wrt(np.arange(3)),
    )
    assert np.allclose(
        infoset_sparse.entropy_wrt(np.array([5])),
        infoset_dense.entropy_wrt(np.array([5])),
    )


def test_entropy_wrt_consistency(linearxy):
    X, y, truefeats_abs = linearxy
    X_q = pr.markers.mutualinformation.infoset.quantile_discretize(X)
    infoset = pr.markers.SparseInformationSet(X_q, y)
    assert np.allclose(
        infoset.entropy_wrt(np.arange(0)),
        np.array([infoset.entropy([i]) for i in range(X.shape[1])]),
    )
    assert np.allclose(
        infoset.entropy_wrt([-1]),
        np.array([infoset.entropy([-1, i]) for i in range(X.shape[1])]),
    )
