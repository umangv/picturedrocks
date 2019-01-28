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


if __name__ == "__main__":
    home()
