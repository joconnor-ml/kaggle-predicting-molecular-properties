import numpy as np
import pandas as pd

train = pd.read_csv("../../data/csv/train.csv")
molecule_names = train["molecule_name"].unique()

grps = train.groupby("molecule_name")
masks = []
targets = []
for mol in molecule_names:
    df = grps.get_group(mol)
    mask = np.zeros((50,50), dtype=np.uint8)
    mask[df["atom_index_0"].values, df["atom_index_1"].values] = 1
    masks.append(mask)

    target = np.zeros((50,50), dtype=np.float32)
    target[df["atom_index_0"].values, df["atom_index_1"].values] = df["scalar_coupling_constant"]
    targets.append(target)

masks = np.asarray(masks)
targets = np.asarray(targets)

np.save('./CHAMPS/scalar_coupling.npy', targets)
np.save('./CHAMPS/scalar_coupling.mask.npy', masks)