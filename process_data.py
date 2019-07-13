from xyz2mol import MolFromXYZ
import itertools
from collections import defaultdict
from rdkit import Chem

import pandas as pd
import numpy as np
import os
import multiprocessing


SYMBOLS=['H', 'C', 'N', 'O', 'F']
BONDS = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]
HYBRIDIZATIONS=[
    #Chem.rdchem.HybridizationType.S,
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    #Chem.rdchem.HybridizationType.SP3D,
    #Chem.rdchem.HybridizationType.SP3D2,
]


def one_hot_encoding(x, set):
    one_hot = [int(x == s) for s in set]
    return one_hot


def structure_to_graph(structure_file):
    mol = MolFromXYZ(structure_file)

    n_atoms = mol.GetNumAtoms()
    edge_array = []
    bond_features = []

    for i, j in itertools.product(range(n_atoms), repeat=2):
        if i == j:
            continue
        edge_array.append((i, j))

        bond = mol.GetBondBetweenAtoms(i, j)
        if bond:
            bond_type = bond.GetBondType()
        else:
            bond_type = None
        bond_features.append(one_hot_encoding(bond_type, BONDS))

    edge_array = np.array(edge_array).T
    edge_features = np.array(bond_features)

    atom_features = defaultdict(list)

    n_atoms = mol.GetNumAtoms()

    for i in range(n_atoms):
        atom = mol.GetAtomWithIdx(i)
        atom_features["symbol"].append(one_hot_encoding(atom.GetSymbol(), SYMBOLS))
        atom_features["aromatic"].append([atom.GetIsAromatic()])
        atom_features["hybridization"].append(one_hot_encoding(atom.GetHybridization(), HYBRIDIZATIONS))

        atom_features["num_h"].append([atom.GetTotalNumHs(includeNeighbors=True)])
        atom_features["atomic"].append([atom.GetAtomicNum()])

    atom_features = np.concatenate([atom_features["symbol"], atom_features["aromatic"],
                                    atom_features["hybridization"], atom_features["num_h"],
                                    atom_features["atomic"]], axis=1)

    return edge_array, edge_features, atom_features


def process_multiple(data_file, structure_dir, output_dir):
    train = pd.read_csv(data_file)
    molecule_names = train["molecule_name"].unique()
    structure_files = [os.path.join(structure_dir, f"{molecule_name}.xyz")
                       for molecule_name in molecule_names]
    grps = train.groupby("molecule_name")
    targets = [grps.get_group(molecule_name)["scalar_coupling_constant"].values
               for molecule_name in molecule_names]
    pool = multiprocessing.Pool(multiprocessing.cpu_count())
    results = pool.map(structure_to_graph, structure_files)

    for (edge_array, edge_features, atom_features), targets, molecule_name in zip(results, targets, molecule_names):
        np.save(os.path.join(output_dir, f"{molecule_name}.edge_array.npy"), edge_array)
        np.save(os.path.join(output_dir, f"{molecule_name}.edge_features.npy"), edge_features)
        np.save(os.path.join(output_dir, f"{molecule_name}.atom_features.npy"), atom_features)
        np.save(os.path.join(output_dir, f"{molecule_name}.edge_array.npy"), targets)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('data_file',  type=str)
    parser.add_argument('structure_dir',  type=str)
    parser.add_argument('output_dir',  type=str)
    args = parser.parse_args()

    process_multiple(args.data_file, args.structure_dir, args.output_dir)