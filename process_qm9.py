import numpy as np
import pandas as pd
import os
from pathlib import Path
import csv

def main(qm_dir, train_file, test_file):
    PATH_QM9 = Path(qm_dir)
    train = pd.read_csv(train_file)
    test = pd.read_csv(test_file)

    both = pd.concat([train, test], axis=0)
    both = both.set_index('molecule_name', drop=False)

    both.sort_index(inplace=True)


    def processQM9_file(filename):
        print(filename)
        path = PATH_QM9 / filename
        molecule_name = filename[:-4]

        row_count = sum(1 for row in csv.reader(open(path)))
        na = row_count - 5
        freqs = pd.read_csv(path, sep=' |\t', engine='python', skiprows=row_count - 3, nrows=1, header=None)
        sz = freqs.shape[1]
        is_linear = np.nan
        if 3 * na - 5 == sz:
            is_linear = False
        elif 3 * na - 6 == sz:
            is_linear = True

        stats = pd.read_csv(path, sep=' |\t', engine='python', skiprows=1, nrows=1, header=None)
        stats = stats.loc[:, 2:]
        stats.columns = ['rc_A', 'rc_B', 'rc_C', 'mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve', 'U0', 'U', 'H', 'G',
                         'Cv']

        stats['freqs_min'] = freqs.values[0].min()
        stats['freqs_max'] = freqs.values[0].max()
        stats['freqs_mean'] = freqs.values[0].mean()
        stats['linear'] = is_linear

        mm = pd.read_csv(path, sep='\t', engine='python', skiprows=2, skipfooter=3, names=range(5))[4]
        if mm.dtype == 'O':
            mm = mm.str.replace('*^', 'e', regex=False).astype(float)
        stats['mulliken_min'] = mm.min()
        stats['mulliken_max'] = mm.max()
        stats['mulliken_mean'] = mm.mean()

        stats['molecule_name'] = molecule_name

        data = pd.merge(both.loc[[molecule_name], :].reset_index(drop=True), stats, how='left', on='molecule_name')
        data['mulliken_atom_0'] = mm[data['atom_index_0'].values].values
        data['mulliken_atom_1'] = mm[data['atom_index_1'].values].values

        return data

    def processQM9_list(files):
        df = pd.DataFrame()
        for i, filename in enumerate(files):
            stats = processQM9_file(filename)
            df = pd.concat([df, stats], axis=0)
        return df

    all_files = os.listdir(qm_dir)
    df = processQM9_list(all_files)
    df.to_hdf("qm9.hdf", "data")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('qm_dir',  type=str)
    parser.add_argument('train_file',  type=str)
    parser.add_argument('test_file',  type=str)
    args = parser.parse_args()

    main(args.qm_dir, args.train_file, args.test_file)