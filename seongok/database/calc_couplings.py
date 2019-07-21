import numpy as np
import pandas as pd
import multiprocessing


train = pd.read_csv("../../data/csv/train.csv")
print(train["atom_index_1"].max())
molecule_names = train["molecule_name"].unique()

grps = train.groupby("molecule_name")


type_means = {
    #type   #mean, std, min, max
    '1JHC': 94.9761528641869,
    '2JHC': -0.2706244378832,
    '3JHC': 3.6884695895355,
    '1JHN': 47.4798844844683,
    '2JHN': 3.1247536134185,
    '3JHN': 0.9907298624944,
    '2JHH': -10.2866051639817,
    '3JHH': 4.7710233597359,
}

type_stds = {
    #type   #mean, std, min, max
    '1JHC': 18.27722399839607,
    '2JHC': 4.52360876732858,
    '3JHC': 3.07090647005439,
    '1JHN': 10.92204561670947,
    '2JHN': 3.67345877025737,
    '3JHN': 1.31538940138001,
    '2JHH': 3.97960190019757,
    '3JHH': 3.70498129755812,
}



def process(mol):
    df = grps.get_group(mol)
    mask = np.zeros((30,30), dtype=np.uint8)
    mask[df["atom_index_0"].values, df["atom_index_1"].values] = 1

    target = np.zeros((30,30), dtype=np.float32)
    target[df["atom_index_0"].values, df["atom_index_1"].values] = df["scalar_coupling_constant"]

    mean = np.zeros((30,30), dtype=np.uint8)
    std = np.ones((30,30), dtype=np.uint8)
    for type in type_means:
        std[df[df["type"] == type]["atom_index_0"].values, df[df["type"] == type]["atom_index_1"].values] = type_stds[type]
        mean[df[df["type"] == type]["atom_index_0"].values, df[df["type"] == type]["atom_index_1"].values] = type_means[type]

    return target, mask, std, mean


pool = multiprocessing.Pool(multiprocessing.cpu_count())
results = pool.map(process, molecule_names)

targets = np.asarray([r[0] for r in results])
masks = np.asarray([r[1] for r in results])
stds = np.asarray([r[2] for r in results])
means = np.asarray([r[3] for r in results])

np.save('./CHAMPS/scalar_coupling.npy', targets)
np.save('./CHAMPS/scalar_coupling.mask.npy', masks)
np.save('./CHAMPS/scalar_coupling.std.npy', stds)
np.save('./CHAMPS/scalar_coupling.mean.npy', means)
