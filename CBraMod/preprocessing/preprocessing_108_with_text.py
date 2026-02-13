import mne
import pandas as pd
import torch
import os
import numpy as np
import lmdb
import pickle
from multimodal import clinical_t5_encoder

root_dir = "/mnt/disk1/aiotlab/namth/EEGFoundationModel/datasets/text_epilepsy"
dest_dir = '/mnt/disk1/aiotlab/namth/EEGFoundationModel/datasets/text_epilepsy_preprocessed'

time_window = 10

labeler = pd.read_csv(f"{root_dir}/subject_to_label.csv")
label_dict = {}
label_list = {0: [], 1:[]}
for idx, row in labeler.iterrows():
    # row là 1 Series
    label_dict[row["subject_id"]] = row["label"]
    label_list[row["label"]].append(row["subject_id"])
np.random.shuffle(label_list[0])
np.random.shuffle(label_list[1])

text_data = pd.read_csv(f"{root_dir}/textdata_en.csv")

selected_channels = [
    'Fp1','Fp2','F3','F4','F7','F8',
    'T3','T4','C3','C4','T5','T6',
    'P3','P4','O1','O2','Fz','Cz','Pz',
]

eeg_dict = {}
text_dict = {}

texter = clinical_t5_encoder.ClinicalT5Encoder()
for idx, row in text_data.iterrows():
    subject_id = row["subject_id"]
    text = row["text"]
    encoded_text = texter.encode(text, max_length=512)
    text_dict[subject_id] = encoded_text

def pre_reading_eeg(subject_id):
    subject_path = os.path.join(root_dir, subject_id)
    files = os.listdir(subject_path)
    for file in files:
        path = os.path.join(subject_path, file)
        try:
            raw = mne.io.read_raw_edf(path, preload=True, verbose=False)
            raw.crop(tmin=2.0)   # tmin tính theo giây
        except Exception as e:
            print(f'[WARN] failed to read {path}: {e}')
            continue

        # ensure required channels exist; skip if missing
        have = set(ch.upper() for ch in raw.ch_names)
        need = [ch.upper() for ch in selected_channels]
        if not all(ch in have for ch in need):
            print(f'[WARN] missing channels in {path}, skip.')
            continue

        raw.pick(selected_channels)
        raw.reorder_channels(selected_channels)
        raw.filter(0.3, 75)
        raw.notch_filter(50)
        try:
            raw.resample(200)
        except Exception as e:
            print(f'[WARN] resample failed {path}: {e}')
            continue

        # data in microvolts as float32
        try:
            eeg = raw.get_data(units='uV', reject_by_annotation='omit').astype(np.float32)
        except TypeError:
            eeg = (raw.get_data(reject_by_annotation='omit') * 1e6).astype(np.float32)

        chs, points = eeg.shape
        a = points % (time_window * 200)
        if a != 0:
            eeg = eeg[:, :-a]
        if eeg.size == 0:
            continue

        eeg = eeg.reshape(chs, -1, time_window, 200).transpose(1, 0, 2, 3)

        eeg_dict[file] = eeg
        label_dict[file] = label_dict[subject_id]

for subject_id in labeler['subject_id']:
    pre_reading_eeg(subject_id)

def stratified_split(fold_idx):
    splits = {'train': [], 'test': [], 'val': []}
    for lbl, lst in label_list.items():
        n = len(lst)
        n_test = n // 5
        test_start = fold_idx * n_test
        test_end = test_start + n_test if fold_idx < 4 else n  # last fold takes the remainder
        splits['test'] += lst[test_start:test_end]
        train = lst[:test_start] + lst[test_end:]
        num_val = len(train) // 5
        splits['train'] += train[num_val:]
        splits['val'] += train[:num_val]
    return splits

def fold_construct(fold_idx):
    print(f'Constructing fold {fold_idx} ...')
    
    dataset = {
        'train': list(),
        'val': list(),
        'test': list(),
    }
    
    subjects = stratified_split(fold_idx)
    
    files_dict = {
        'train': [],
        'val': [],
        'test': [],
    }
    
    for key in subjects.keys():
        for subject in subjects[key]:
            subject_path = os.path.join(root_dir, subject)
            files = os.listdir(subject_path)
            for file in files:
                files_dict[key].append(file)

    print(subjects['train'])
    print(subjects['val'])
    print(subjects['test'])

    db = lmdb.open(f'{dest_dir}/fold_{fold_idx}', map_size=1000000000)
    for files_key in files_dict.keys():
        for file in files_dict[files_key]:
            eeg = eeg_dict.get(file, None)
            if eeg is None:
                # print(f'[WARN] no pre-read eeg for {file}, skip.')
                continue
            # label per subject
            label = label_dict[file]
            text = text_dict.get(file[:8], None)
            assert text is not None, f'[ERROR] no text for subject {file[:8]}'

            for i, sample in enumerate(eeg):
                sample_key = f'{file}-{i}'  # Fixed typo from file[:-4]
                data_dict = {
                    'sample': sample, 'label': label, 'text': text
                }
                txn = db.begin(write=True)
                txn.put(key=sample_key.encode(), value=pickle.dumps(data_dict, protocol=pickle.HIGHEST_PROTOCOL))
                txn.commit()
                dataset[files_key].append(sample_key)

    txn = db.begin(write=True)
    txn.put(key='__keys__'.encode(), value=pickle.dumps(dataset, protocol=pickle.HIGHEST_PROTOCOL))
    txn.commit()
    db.close()

for fold_idx in range(5):
    fold_construct(fold_idx)