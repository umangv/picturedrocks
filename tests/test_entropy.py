import numpy as np
import picturedrocks as pr
import pytest


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

