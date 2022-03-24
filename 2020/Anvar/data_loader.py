from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
import pandas as pd
import torch
from dpipe.io import load_numpy


class BraTSDataset(Dataset):
    def __init__(self, meta: pd.DataFrame, source_folder: [str, Path], nonzero_mask=False, transform=None):
        if isinstance(source_folder, str):
            source_folder = Path(source_folder)
            
        if nonzero_mask:
            meta = meta[meta.sample_id.isin(meta.query('is_nonzero_mask == True').sample_id)]
            
        self.source_folder = source_folder
        self.meta_images = meta.query('is_mask == False').sort_values(by='sample_id').reset_index(drop=True)
        self.meta_masks = meta.query('is_mask == True').sort_values(by='sample_id').reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return self.meta_images.shape[0]

    def __getitem__(self, i):
        image = load_numpy(self.source_folder / self.meta_images.iloc[i]['relative_path'], allow_pickle=True, decompress=True)
        mask = load_numpy(self.source_folder / self.meta_masks.iloc[i]['relative_path'], allow_pickle=True, decompress=True)
        sample = image, mask
        
        if self.transform:
            image, mask = self.transform(sample)

        return torch.from_numpy(image).reshape(1, 240, 240), torch.from_numpy(mask).reshape(1, 240, 240).double()
