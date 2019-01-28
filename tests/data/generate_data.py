import click
import numpy as np
import pandas as pd
import itertools


@click.group(chain=True)
def home():
    print("Welcome")
    pass


@home.command()
def diminish_red():
    print("writing diminish-red.csv")
    rows = []
    for bits in itertools.product([0, 1], repeat=5):
        bits = [str(b) for b in bits]
        row = {}
        row["x1"] = "".join(bits[:2])
        row["x2"] = "".join(bits[2:4])
        row["x3"] = bits[4]
        row["x4"] = "".join(bits[:1])
        row["y"] = "".join(bits)
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv("diminish-red.csv")


@home.command()
def xor():
    print("writing xor.csv")
    rows = []
    for bits in itertools.product(*([[0, 1]] * 2), [0, 1, 2]):
        row = {}
        row["x1"] = bits[0]
        row["x2"] = bits[1]
        row["y"] = bits[0] ^ bits[1]
        row["x3"] = row["y"] ^ (bits[2] == 2)
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv("xor.csv")


@home.command()
def test_dataset():
    print("writing test-dataset.csv")
    df = pd.DataFrame(
        np.arange(16).reshape((4, 4)),
        index=["Cell_A", "Cell_B", "Cell_C", "Cell_D"],
        columns=["Gene_A", "Gene_B", "Gene_C", "Gene_D"],
    )
    df.to_csv("test-dataset.csv")

@home.command()
def test_classlabels():
    df = pd.DataFrame(
        np.arange(4).reshape((4, 1)) % 2,
        index=["Cell_A", "Cell_B", "Cell_C", "Cell_D"],
        columns=["cluster"],
    )
    print("writing test-labels-raw.csv")
    df.to_csv("test-labels-raw.csv", header=False, index=False)
    print("writing test-labels-raw-head.csv")
    df.to_csv("test-labels-raw-head.csv", header=True, index=False)
    print("writing test-labels-meta.csv")
    df.to_csv("test-labels-meta.csv")
    


if __name__ == "__main__":
    home()
