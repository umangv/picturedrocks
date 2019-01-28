import numpy as np
import pandas as pd
import picturedrocks as pr
import pytest


@pytest.fixture()
def diminish_red_df(datadir):
    df = pd.read_csv(datadir / "diminish-red.csv", index_col=0)
    return df


@pytest.fixture()
def diminish_red_inf(diminish_red_df):
    infoset = pr.markers.SparseInformationSet(diminish_red_df.values[:, :-1])
    print(infoset.X.shape)
    y = diminish_red_df["y"].values
    origy, y = np.unique(y, return_inverse=True)
    infoset.set_y(y)
    return infoset


@pytest.fixture()
def xor_df(datadir):
    df = pd.read_csv(datadir / "xor.csv", index_col=0)
    return df


@pytest.fixture()
def xor_inf(xor_df):
    infoset = pr.markers.SparseInformationSet(xor_df.values[:, :-1])
    print(infoset.X.shape)
    y = xor_df["y"].values
    origy, y = np.unique(y, return_inverse=True)
    infoset.set_y(y)
    return infoset


def test_diminish_red(diminish_red_inf):
    cife = pr.markers.CIFE(diminish_red_inf)
    jmi = pr.markers.JMI(diminish_red_inf)
    mim = pr.markers.MIM(diminish_red_inf)
    cife.autoselect(2)
    jmi.autoselect(2)
    assert set(cife.S) == set([0, 1])
    assert set(jmi.S) == set([0, 1])
    assert np.allclose(cife.score, [float("-inf"), float("-inf"), 1, 0])
    assert np.allclose(jmi.score, [float("-inf"), float("-inf"), 1, 0.5])
    assert np.allclose(mim.score, [2, 2, 1, 1])


@pytest.mark.parametrize(
    "objective",
    [
        pr.markers.CIFE,
        pr.markers.JMI,
        pr.markers.MIM,
        pr.markers.CIFEUnsup,
        pr.markers.UniEntropy,
    ],
)
def test_feature_remove(diminish_red_inf, objective):
    fs = objective(diminish_red_inf)
    fs.autoselect(2)
    assert set(fs.S) == set([0, 1])
    fs.remove(0)
    assert len(fs.S) == 1
    fs.autoselect(1)
    assert fs.S[-1] == 0
    fs.remove(1)
    assert len(fs.S) == 1
    fs.autoselect(1)
    assert fs.S[-1] == 1


@pytest.mark.parametrize("objective", [pr.markers.CIFE, pr.markers.JMI])
def test_cife_jmi_on_xor(xor_inf, objective):
    fs = objective(xor_inf)
    assert np.allclose(
        fs.score, [0, 0, 1 + ((1 / 3) * np.log2(1 / 3) + (2 / 3) * np.log2(2 / 3))]
    )
    fs.autoselect(1)
    assert fs.S == [2]
    fs.remove(2)
    fs.add(0)
    assert np.allclose(
        fs.score,
        [float("-inf"), 1, 1 + ((1 / 3) * np.log2(1 / 3) + (2 / 3) * np.log2(2 / 3))],
    )
    fs.autoselect(1)
    assert fs.S == [0, 1]
