from scipy import signal
import re
import lmdb
import pickle
import numpy as np
import pandas as pd
import torch
from scipy.signal import resample
from tqdm import tqdm

dataset_dirs = [
    '/mnt/disk1/aiotlab/namth/EEGFoundationModel/datasets/nfeeg/94-Vietnamese-Characters EEG Dataset - Female.csv',
    '/mnt/disk1/aiotlab/namth/EEGFoundationModel/datasets/nfeeg/94-Vietnamese-Characters EEG Dataset - Male.csv'
]

preprocessed_data_dir = '/mnt/disk1/aiotlab/namth/EEGFoundationModel/datasets/nfm_for_BIOT'  # New directory to avoid overwriting

seed = 3407
np.random.seed(seed)

selected_channels = ['Fp1']
train_ratio, val_ratio, test_ratio = 0.7, 0.15, 0.15

def str_to_array(s):
    s = s.replace('[', ' ').replace(']', ' ')
    tokens = re.split(r'[,\s]+', s.strip())
    tokens = [t for t in tokens if t != '']
    return np.array([int(t) for t in tokens], dtype=np.int64)

def str_to_label(s: str) -> int:
    s = s.replace('[', ' ').replace(']', ' ')
    tokens = re.split(r'[,\s]+', s.strip())
    tokens = [t for t in tokens if t != '']
    return int(tokens[0])

def resample_512_to_200(x: np.ndarray) -> torch.Tensor:
    if x.shape[0] != 512:
        raise ValueError(f"Expected input shape (512,), got {x.shape}")
    y = resample(x, 200)
    return torch.from_numpy(y).float().reshape(1, -1)  # Return PyTorch tensor

dataset = {'train': [], 'val': [], 'test': []}

db = lmdb.open(preprocessed_data_dir, map_size=1000000000)
for idx, dataset_dir in enumerate(dataset_dirs):
    print(f'Processing {dataset_dir} ...')
    df_dict = [[] for _ in range(94)]
    signal = []
    df = pd.read_csv(dataset_dir)
    n = len(df)
    for i in range(n):
        row = df.iloc[i]
        x = resample_512_to_200(str_to_array(row['col 1']))  # Returns PyTorch tensor
        y = str_to_label(row['col 3'])
        signal.append(x)
        df_dict[y].append(i)

    for i in tqdm(range(94)):
        np.random.shuffle(df_dict[i])
        n_i = len(df_dict[i])
        n_train = int(n_i * train_ratio)
        n_val = int(n_i * val_ratio)
        indices = {
            'train': df_dict[i][:n_train],
            'val': df_dict[i][n_train:n_train + n_val],
            'test': df_dict[i][n_train + n_val:]
        }
        for split in ['train', 'val', 'test']:
            for j in indices[split]:
                sample_key = f'{idx}-{j}'
                sample = signal[j]  # Already a PyTorch tensor
                data_dict = {'sample': sample, 'label': y}  # Label is an integer
                with db.begin(write=True) as txn:
                    txn.put(sample_key.encode(), pickle.dumps(data_dict, protocol=pickle.HIGHEST_PROTOCOL))
                dataset[split].append(sample_key)

with db.begin(write=True) as txn:
    txn.put('__keys__'.encode(), pickle.dumps(dataset, protocol=pickle.HIGHEST_PROTOCOL))
db.close()

print('Done!')