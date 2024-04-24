import os

import numpy as np
import torch
import yaml

from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

class CompetitionDataset(Dataset):
    def __init__(self, directory, normalize=False, data_name='custom', dataset_type='labeled', limit=-1):
        '''
        dataset_type: 'labeled' / 'unlabeled' / 'hidden'
        limit: just used for quick debugging
        '''
        super().__init__()
        self.video_folders = [f for f in sorted(Path(directory).iterdir()) if f.is_dir()][:limit]
        self.dataset_type = dataset_type

        # self.data_name = data_name #TODO: try if data_name is not defined
        self.mean = None
        self.std = None

        if normalize:
            mean = data.mean(axis=(0, 1, 2, 3)).reshape(1, 1, -1, 1, 1)
            std = data.std(axis(0, 1, 2, 3)).reshape(1, 1, -1, 1, 1)
            data = (data - mean)/std
            self.mean = mean
            self.std = std
        
        self.transform_image = transforms.Compose([
            transforms.ToTensor()
        ])

        self.transform_mask = lambda x: torch.from_numpy(x).long()

    def __len__(self):
        return len(self.video_folders)
    
    def __getitem__(self, index):
        files = sorted(self.video_folders[index].glob('*.png'), key=lambda x: int(x.stem.split('_')[-1]))
        video_frames = [self.transform_image(Image.open(f).convert('RGB')) for f in files ]
        pre_seqs = torch.stack(video_frames[:11]).float()

        if self.dataset_type == 'hidden':
            return pre_seqs
        
        else:
            aft_seqs = torch.stack(video_frames[11:]).float()
            if self.dataset_type == 'labeled':
                masks_dir = self.video_folders[index].joinpath("mask.npy")
                masks = self.transform_mask(np.load(masks_dir))
                return pre_seqs, aft_seqs, masks
            
            elif self.dataset_type == 'unlabeled':
                return pre_seqs, aft_seqs

