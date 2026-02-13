import numpy as np
import os
import json
import pandas as pd
import mne
from mne.preprocessing import ICA
import lmdb
import pickle
import torch

selected_channels = [
    'FP1', 'FP2', 'F3', 'F4', 'F7', 'F8',
    'T7', 'T8', 'C3', 'C4', 'P7', 'P8',
    'P3', 'P4', 'O1', 'O2', 'FZ', 'CZ', 'PZ',
]

available_channels = [
    "FZ", "FC1", "FC2", "C3", "CZ", "C4", "CP1", "CP2", "PZ",
    "AF3", "AF4", "F3", "F4", "FC5", "FC6", "CP5", "CP6",
    "P3", "P4", "PO7", "PO8", "OZ"
]

root_dir = '/mnt/disk1/aiotlab/namth/EEGFoundationModel/datasets/Data_UET175'
dest_dir = '/mnt/disk1/aiotlab/namth/EEGFoundationModel/datasets/UET175_for_BIOT'  # New directory

seed = 3407
np.random.seed(seed)

def load_eeg_csv_to_ct(path: str) -> np.ndarray:
    df = pd.read_csv(path)
    if df.shape[1] != 22:
        raise ValueError(f"Expected 22 columns, got {df.shape[1]} in {path}")
    data = df.astype(float).to_numpy()  # shape (T, 22)
    return data.T  # shape (22, T)

def preprocess_eeg_mne_prefix(x: np.ndarray) -> np.ndarray:
    assert isinstance(x, np.ndarray) and x.ndim == 2 and x.shape[0] == 22
    fs = 128.0
    info = mne.create_info(ch_names=available_channels, sfreq=fs, ch_types="eeg")
    raw = mne.io.RawArray(x.astype(float, copy=False), info, verbose=False)
    raw._data -= raw._data.mean(axis=1, keepdims=True)
    raw.filter(l_freq=1.0, h_freq=None, method="iir", iir_params=dict(order=5, ftype="butter"), verbose=False)
    raw.notch_filter(freqs=[50.0], method="iir", verbose=False)
    raw.set_eeg_reference("average", verbose=False)
    ica = mne.preprocessing.ICA(n_components=0.999, method="picard", random_state=seed, max_iter="auto", verbose=False)
    ica.fit(raw, verbose=True)
    raw_clean = ica.apply(raw.copy(), verbose=False)
    return raw_clean.get_data()

def preprocess_eeg_mne_suffix(x: np.ndarray) -> torch.Tensor:
    assert isinstance(x, np.ndarray) and x.ndim == 2 and x.shape[0] == 22
    fs = 128.0
    info = mne.create_info(ch_names=available_channels, sfreq=fs, ch_types="eeg")
    raw = mne.io.RawArray(x.astype(float, copy=False), info, verbose=False)
    raw.resample(200, verbose=False)
    X = raw.get_data()
    X = (X - X.mean(axis=1, keepdims=True)) / (X.std(axis=1, keepdims=True) + 1e-8)
    return torch.from_numpy(X).float()  # Return PyTorch tensor

file = {'Male': [], 'Female': []}
subjects = sorted(os.listdir(root_dir))
ls_subjects = {'train': [], 'val': [], 'test': []}

for subject in subjects:
    path = os.path.join(root_dir, subject)
    with open(f"{path}/Info.json", "r", encoding="utf-8") as f:
        info = json.load(f)
    gender = info['Gender']
    file[gender].append(subject)

train_ratio, val_ratio, test_ratio = 0.7, 0.2, 0.1
for gender in file.keys():
    np.random.shuffle(file[gender])
    n = len(file[gender])
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val
    ls_subjects['train'] += file[gender][:n_train]
    ls_subjects['val'] += file[gender][n_train:n_train + n_val]
    ls_subjects['test'] += file[gender][n_train + n_val:]

dataset = {'train': [], 'val': [], 'test': []}
db = lmdb.open(dest_dir, map_size=1000000000)

def process(tag, subject, day):
    sub_path = os.path.join(root_dir, subject, day)
    files = sorted(os.listdir(sub_path))
    runs = [f.split('_')[0] for f in os.listdir(sub_path) if f.endswith(".csv")]
    for run in runs:
        with open(f"{sub_path}/{run}_Session setup.json", "r", encoding="utf-8") as f:
            setup = json.load(f)
        raw = load_eeg_csv_to_ct(f"{sub_path}/{run}_Raw Signals.csv")  # shape (22, T)
        raw = preprocess_eeg_mne_prefix(raw)  # shape (22, T)
        with open(f"{sub_path}/{run}_Action label.txt", "r", encoding="utf-8") as f:
            actions = [int(line.strip()) for line in f if line.strip()]
        id2name = {3: "RH", 4: "LH", 5: "RF", 6: "LF"}
        mi_labels = [a - 3 for a in actions if a in id2name]
        with open(f"{sub_path}/{run}_Event timestamp.txt", "r", encoding="utf-8") as f:
            ts = [int(line.strip()) for line in f if line.strip()]
        if len(mi_labels) != len(ts):
            continue
        for i in range(len(ts)):
            s = ts[i]
            e = ts[i] + 4 * 128
            y = mi_labels[i]
            x = preprocess_eeg_mne_suffix(raw[:, s:e])  # Returns PyTorch tensor (22, 800)
            if x.shape[1] != 800:
                continue
            x = x.reshape(22, 4, 200)
            sample_key = f'{subject}_{day}_{run}_{i}'
            data_dict = {'sample': x, 'label': y}
            with db.begin(write=True) as txn:
                txn.put(key=sample_key.encode(), value=pickle.dumps(data_dict, protocol=pickle.HIGHEST_PROTOCOL))
            dataset[tag].append(sample_key)

for tag in ls_subjects.keys():
    for subject in ls_subjects[tag]:
        sub_path = os.path.join(root_dir, subject)
        dirs = sorted(os.listdir(sub_path))
        for day in dirs:
            if os.path.isdir(os.path.join(sub_path, day)):
                process(tag=tag, subject=subject, day=day)

with db.begin(write=True) as txn:
    txn.put(key='__keys__'.encode(), value=pickle.dumps(dataset, protocol=pickle.HIGHEST_PROTOCOL))
db.close()