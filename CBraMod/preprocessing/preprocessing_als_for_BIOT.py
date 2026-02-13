import os
import mne
import numpy as np
import pickle
import lmdb
from scipy.signal import butter, resample, filtfilt
import json
from tqdm import tqdm
import torch
import warnings
warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning,
    message=".*annotation.*outside the data range.*"
)

# ------------------ Preprocessing ------------------
root_dir = "/home/aiotlab/.cache/kagglehub/datasets/patrickiitmz/eeget-als-dataset/versions/1/EEGET-ALS Dataset"
output_db = "/mnt/disk1/aiotlab/namth/EEGFoundationModel/datasets/als_for_BIOT"
task = "LRF0"
win_sec = 4
target_len = 800
seed = 3407
np.random.seed(seed)

def process_raw(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    fs = float(raw.info["sfreq"])
    nyq = 0.5 * fs
    b, a = butter(5, [0.3/nyq, 50.0/nyq], btype="band")
    data = raw.get_data(picks="eeg")
    data = data - np.mean(data, axis=1, keepdims=True)
    data_bp = filtfilt(b, a, data, axis=-1)
    info_eeg = raw.copy().pick("eeg").info
    raw_bp = mne.io.RawArray(data_bp, info_eeg, verbose=False)
    raw_bp.notch_filter(freqs=[50.0], picks="eeg", method="iir", verbose=False)
    n_comp = max(5, min(20, raw_bp.info["nchan"] - 1))
    ica = mne.preprocessing.ICA(n_components=n_comp, method="picard", random_state=97, max_iter="auto")
    ica.fit(raw_bp)
    raw_clean_eeg = raw_bp.copy()
    ica.apply(raw_clean_eeg)
    raw_out = raw.copy()
    eeg_picks = mne.pick_types(raw_out.info, eeg=True, exclude=())
    raw_out._data[eeg_picks, :] = raw_clean_eeg.get_data()
    return raw_out

def map_label(annotation: str, scenario_id: int, task="LRF0"):
    if annotation == "Resting":
        return 0
    if task == "LR0":
        if scenario_id == 1: return 1
        if scenario_id == 2: return 2
        return None
    elif task == "LRF0":
        if scenario_id == 1: return 1
        if scenario_id == 2: return 2
        if scenario_id == 3: return 3
        return None
    return None

standard_19 = [
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8",
    "T3", "C3", "Cz", "C4", "T4",
    "T5", "P3", "Pz", "P4", "T6",
    "O1", "O2"
]

map_10_10 = {
    "T3": "T7",
    "T4": "T8",
    "T5": "P7",
    "T6": "P8"
}
mapped_channels = [map_10_10.get(ch, ch) for ch in standard_19]

train_ratio, val_ratio = 0.85, 0.15
subjects = [x for x in os.listdir(root_dir) if x.startswith("id")]
np.random.shuffle(subjects)
n_subj = len(subjects)
n_train = int(n_subj * train_ratio)
n_val = n_subj - n_train

ls_dict = {
    "train": subjects[:n_train],
    "val": subjects[n_train:n_train + n_val],
    "test": [x for x in os.listdir(root_dir) if x.startswith("ALS")],
}

print("Train subjects:", ls_dict["train"])
print("Val subjects:", ls_dict["val"])
print("Test subjects:", ls_dict["test"])

rev_ls_dict = {subj: split for split, subjs in ls_dict.items() for subj in subjs}

db = lmdb.open(output_db, map_size=16106127360)
dataset = {"train": [], "val": [], "test": []}

for subject in tqdm(sorted(os.listdir(root_dir)), desc="Processing subjects"):
    subj_path = os.path.join(root_dir, subject)
    if not os.path.isdir(subj_path):
        continue

    for time_folder in sorted(os.listdir(subj_path)):
        time_path = os.path.join(subj_path, time_folder)
        if not os.path.isdir(time_path):
            continue

        for scenario_folder in sorted(os.listdir(time_path)):
            scenario_path = os.path.join(time_path, scenario_folder)
            if not os.path.isdir(scenario_path):
                continue

            scenario_id = int(scenario_folder.replace("scenario", ""))
            eeg_file = os.path.join(scenario_path, "EEG.edf")
            anno_file = os.path.join(scenario_path, "eeg.json")

            if not (os.path.exists(eeg_file) and os.path.exists(anno_file)):
                continue

            with open(anno_file, "r") as f:
                meta = json.load(f)
            scenario_anno = meta.get("annotation", "Resting")

            raw = mne.io.read_raw_edf(eeg_file, preload=True, verbose=False)
            fs = int(raw.info["sfreq"])
            raw = process_raw(raw)
            raw.pick_channels([ch for ch in mapped_channels if ch in raw.ch_names])

            data = raw.get_data()

            for ann in raw.annotations:
                onset = int(ann["onset"] * fs)
                duration = int(ann["duration"] * fs)
                desc = str(ann["description"])

                label = map_label(desc if desc else scenario_anno, scenario_id, task=task)
                if label is None:
                    continue

                seg_data = data[:, onset:onset+duration]
                win_len = fs * win_sec
                n_win = seg_data.shape[1] // win_len

                for w in range(n_win):
                    seg = seg_data[:, w*win_len:(w+1)*win_len]
                    seg = resample(seg, target_len, axis=-1)
                    seg = seg.reshape(seg.shape[0], 4, 200)
                    seg = torch.from_numpy(seg).float()  # Convert to PyTorch tensor

                    sample_key = f"{subject}-{time_folder}-{scenario_folder}-{ann['onset']:.2f}-win{w}"
                    data_dict = {"sample": seg, "label": label}

                    with db.begin(write=True) as txn:
                        txn.put(sample_key.encode(), pickle.dumps(data_dict, protocol=pickle.HIGHEST_PROTOCOL))
                        dataset[rev_ls_dict[subject]].append(sample_key)

with db.begin(write=True) as txn:
    txn.put("__keys__".encode(), pickle.dumps(dataset, protocol=pickle.HIGHEST_PROTOCOL))
db.close()

print("✅ Done: 4s windows aligned with EDF annotations saved to LMDB")