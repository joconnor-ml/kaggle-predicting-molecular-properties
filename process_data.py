from xyz2mol import MolFromXYZ
import itertools
from collections import defaultdict

import pandas as pd
import numpy as np
import os
import multiprocessing

from rdkit import Chem
from rdkit.Chem import ChemicalFeatures
from rdkit import RDConfig
from sklearn import preprocessing

from dscribe.descriptors import ACSF
from dscribe.core.system import System

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

ACSF_GENERATOR = ACSF(
    species = SYMBOLS,
    rcut = 6.0,
    g2_params=[[1, 1], [1, 2], [1, 3]],
    g4_params=[[1, 1, 1], [1, 2, 1], [1, 1, -1], [1, 2, -1]],
)



def one_hot_encoding(x, set):
    one_hot = [int(x == s) for s in set]
    return one_hot

def one_hot_numpy(x, width):
    b = np.zeros((x.shape[0], width))
    b[np.arange(x.shape[0]), x] = 1
    return b



def structure_to_graph(structure_file):
    mol, smile = MolFromXYZ(structure_file)
    factory = ChemicalFeatures.BuildFeatureFactory(os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef'))
    feature = factory.GetFeaturesForMol(mol)

    structure = pd.read_csv(structure_file, skiprows=1, header=None, sep=" ",
                            names=["atom", "x", "y", "z"])
    structure["radius"] = structure["atom"].map({'H': 0.38, 'C': 0.77, 'N': 0.75, 'O': 0.73, 'F': 0.71})
    xyz = structure[["x", "y", "z"]]
    norm_xyz = preprocessing.normalize(xyz, norm='l2')

    n_atoms = mol.GetNumAtoms()
    edge_array = []
    bond_features = []
    distance = []
    rel_distance = []
    angle = []
    bond_vector = []

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
        r = ((xyz.iloc[i] - xyz.iloc[j])**2).sum()**0.5
        rel_dist = r/(structure.iloc[i]["radius"] +
                      structure.iloc[j]["radius"])
        theta = (norm_xyz[i]*norm_xyz[j]).sum()
        distance.append([r])
        rel_distance.append([rel_dist])  # divide distance by sum of atomic radii
        angle.append([theta])
        bond_vector.append((xyz.iloc[i] - xyz.iloc[j]).tolist())

    #distance = np.digitize(np.array(distance), bins=[0, 1, 2, 4, 8])
    #rel_distance = np.digitize(np.array(rel_distance), bins=[0, 1, 2, 4, 8])
    #angle = np.digitize(np.array(angle), bins=[-1, -.6, -.2, .2, .6])

    edge_array = np.array(edge_array).T
    edge_features = np.concatenate([
        np.array(bond_features),
        np.array(distance) / 4 - 1,
        np.array(rel_distance) / 4 - 1,
        np.array(angle),  # absolute bond angle. Can use to calculate dihedral angles
        np.array(bond_vector)  # difference between coords of atoms i and j
    ], axis=1)

    atom_features = defaultdict(list)

    n_atoms = mol.GetNumAtoms()

    for i in range(n_atoms):
        atom = mol.GetAtomWithIdx(i)
        atom_features["symbol"].append(one_hot_encoding(atom.GetSymbol(), SYMBOLS))
        atom_features["aromatic"].append([atom.GetIsAromatic()])
        atom_features["hybridization"].append(one_hot_encoding(atom.GetHybridization(), HYBRIDIZATIONS))

        atom_features["num_h"].append([atom.GetTotalNumHs(includeNeighbors=True)])
        atom_features["atomic"].append([atom.GetAtomicNum()])

    atom = System(symbols=structure["atom"].values, positions=xyz.values)
    acsf = ACSF_GENERATOR.create(atom)
    atom_features["acsf"] = acsf

    acceptor = np.zeros((n_atoms, 1), np.uint8)
    donor = np.zeros((n_atoms, 1), np.uint8)

    for feat in feature:
        if feat.GetFamily() == 'Donor':
            for i in feat.GetAtomIds():
                donor[i] = 1
        elif feat.GetFamily() == 'Acceptor':
            for i in feat.GetAtomIds():
                acceptor[i] = 1

    print(len(atom_features["acsf"]), len(acceptor), len(donor))
    atom_features = np.concatenate([atom_features["symbol"], atom_features["aromatic"],
                                    atom_features["hybridization"], atom_features["num_h"],
                                    atom_features["atomic"], atom_features["acsf"],
                                    acceptor, donor], axis=1)

    return edge_array, edge_features, atom_features, smile, xyz.values


def get_path(atom0, atom1, edge_array, max_depth=3):
    """TODO: too confusing"""
    nodes = [atom0]
    for i in range(max_depth):
        # get shortest path from atom 0 to atom 1
        possible_hops = [edge_array[edge_array[:,0] == node] for node in nodes]


    return


def process_multiple(data_file, structure_dir, output_dir, test=False):
    train = pd.read_csv(data_file)
    class_counts = train.groupby("type")["id"].count()
    class_counts = class_counts / class_counts.mean()
    class_weight_map = (1 / class_counts).to_dict()

    molecule_names = train["molecule_name"].unique()
    structure_files = [os.path.join(structure_dir, f"{molecule_name}.xyz")
                       for molecule_name in molecule_names]
    grps = train.groupby("molecule_name")
    target_indices = [grps.get_group(molecule_name)[["atom_index_0", "atom_index_1"]].values.T
                      for molecule_name in molecule_names]

    pool = multiprocessing.Pool(multiprocessing.cpu_count())
    results = pool.map(structure_to_graph, structure_files)

    try:
        targets = [grps.get_group(molecule_name)["scalar_coupling_constant"].values
                   for molecule_name in molecule_names]
    except:
        print("Targets not found: assuming this is test data")
        targets = [np.array([]) for _ in molecule_names]

    target_map = {
        '1JHC': 0,
        '1JHN': 1,
        '2JHC': 2,
        '2JHH': 3,
        '2JHN': 4,
        '3JHC': 5,
        '3JHH': 6,
        '3JHN': 7
    }
    target_classes = [grps.get_group(molecule_name)["type"].map(target_map).values
                      for molecule_name in molecule_names]
    target_weights = [grps.get_group(molecule_name)["type"].map(class_weight_map).values
                      for molecule_name in molecule_names]

    with open("{}/smiles.txt".format(output_dir), "wt") as f:
        f.writelines([r[3]+"\n" for r in results])

    for (edge_array, edge_features, atom_features, smile, xyz), targets, target_index, target_class, target_weight, molecule_name in \
            zip(results, targets, target_indices, target_classes, target_weights, molecule_names):
        np.save(os.path.join(output_dir, f"{molecule_name}.edge_array.npy"), edge_array)
        np.save(os.path.join(output_dir, f"{molecule_name}.edge_features.npy"), edge_features)
        np.save(os.path.join(output_dir, f"{molecule_name}.atom_features.npy"), atom_features)
        np.save(os.path.join(output_dir, f"{molecule_name}.targets.npy"), targets)
        np.save(os.path.join(output_dir, f"{molecule_name}.target_indices.npy"), target_index)
        np.save(os.path.join(output_dir, f"{molecule_name}.target_class.npy"), target_class)
        np.save(os.path.join(output_dir, f"{molecule_name}.target_weight.npy"), target_weight)
        np.save(os.path.join(output_dir, f"{molecule_name}.xyz.npy"), xyz)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('data_file',  type=str)
    parser.add_argument('structure_dir',  type=str)
    parser.add_argument('output_dir',  type=str)
    args = parser.parse_args()

    process_multiple(args.data_file, args.structure_dir, args.output_dir)