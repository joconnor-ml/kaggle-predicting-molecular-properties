import numpy as np
from rdkit import Chem
from rdkit.Chem.QED import qed
from rdkit.Chem.Descriptors import ExactMolWt
from rdkit.Chem.Crippen import MolLogP
from rdkit.Chem.rdMolDescriptors import CalcTPSA
from rdkit.Contrib.SA_Score.sascorer import calculateScore
import sys
import pandas as pd

train = pd.read_csv(data_file)
molecule_names = train["molecule_name"].unique()

masks = []
targets = []
for mol in molecule_names:
    df = train[train["molecule_name"] == mol]
    mask = np.zeros((50,50), dtype=np.uint8)
    mask[df["atom0"].values, df["atom1"].values] = 1
    masks.append(mask)

    target = np.zeros((50,50), dtype=np.float32)
    target[df["atom0"].values, df["atom1"].values] = df["scalar_coupling_constant"]
    targets.append(target)

masks = np.asarray(masks)
targets = np.asarray(targets)

np.save('./CHAMPS/scalar_coupling.npy', targets)
np.save('./CHAMPS/scalar_coupling.mask.npy', masks)
