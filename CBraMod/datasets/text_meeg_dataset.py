import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from utils.util import to_tensor
import os
import random
import lmdb
import pickle
from torch.nn.utils.rnn import pad_sequence

class CustomDataset(Dataset):
    def __init__(
            self,
            data_dir,
            mode='train',
    ):
        super(CustomDataset, self).__init__()
        self.db = lmdb.open(data_dir, readonly=True, lock=False, readahead=True, meminit=False)
        with self.db.begin(write=False) as txn:
            self.keys = pickle.loads(txn.get('__keys__'.encode()))[mode]

    def __len__(self):
        return len((self.keys))

    def __getitem__(self, idx):
        key = self.keys[idx]
        with self.db.begin(write=False) as txn:
            pair = pickle.loads(txn.get(key.encode()))
        data = pair['sample']
        label = pair['label']
        text = pair['text']
        # print(label)
        return (data/100, text), label

    def collate(self, batch):
        x_data = np.array([x[0][0] for x in batch])
        y_label = np.array([x[1] for x in batch])
        
        
        seqs = [x[0][1].squeeze(0) for x in batch]  # (L_i, d)
        device = seqs[0].device
        dtype = seqs[0].dtype
        x_text = pad_sequence(seqs, batch_first=True).to(device=device, dtype=dtype)
        lengths = torch.tensor([s.size(0) for s in seqs], device=device)
        max_len = x_text.size(1)
        # arange: (max_len,) -> (1, max_len) -> (B, max_len)
        idxs = torch.arange(max_len, device=device).unsqueeze(0).expand(len(seqs), -1)
        x_mask = idxs < lengths.unsqueeze(1)   # (B, max_len), bool
                
        return (to_tensor(x_data), x_text, x_mask), to_tensor(y_label)


class LoadDataset(object):
    def __init__(self, params):
        self.params = params
        self.datasets_dir = params.datasets_dir

    def get_data_loader(self):
        train_set = CustomDataset(self.datasets_dir, mode='train')
        val_set = CustomDataset(self.datasets_dir, mode='val')
        test_set = CustomDataset(self.datasets_dir, mode='test')
        print(len(train_set), len(val_set), len(test_set))
        print(len(train_set)+len(val_set)+len(test_set))
        data_loader = {
            'train': DataLoader(
                train_set,
                batch_size=self.params.batch_size,
                collate_fn=train_set.collate,
                shuffle=True,
            ),
            'val': DataLoader(
                val_set,
                batch_size=self.params.batch_size,
                collate_fn=val_set.collate,
                shuffle=True,
            ),
            'test': DataLoader(
                test_set,
                batch_size=self.params.batch_size,
                collate_fn=test_set.collate,
                shuffle=True,
            ),
        }
        return data_loader
