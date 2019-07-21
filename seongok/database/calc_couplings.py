import numpy as np
import pandas as pd
import multiprocessing


train = pd.read_csv("../../data/csv/train.csv")
print(train["atom_index_1"].max())
molecule_names = train["molecule_name"].unique()

grps = train.groupby("molecule_name")


def process(mol):
    df = grps.get_group(mol)
    mask = np.zeros((50,50), dtype=np.uint8)
    mask[df["atom_index_0"].values, df["atom_index_1"].values] = 1

    target = np.zeros((50,50), dtype=np.float32)
    target[df["atom_index_0"].values, df["atom_index_1"].values] = df["scalar_coupling_constant"]
    return target, mask


pool = multiprocessing.Pool(multiprocessing.cpu_count())
results = pool.map(process, molecule_names)

masks = np.asarray([mask for target, mask in results])
targets = np.asarray([target for target, mask in results])

np.save('./CHAMPS/scalar_coupling.npy', targets)
np.save('./CHAMPS/scalar_coupling.mask.npy', masks)
