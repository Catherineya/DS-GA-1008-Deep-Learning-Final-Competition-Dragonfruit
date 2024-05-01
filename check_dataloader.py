import os


import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import Dataset, ConcatDataset, DataLoader, random_split
from dragonfruitvp.data.custom_dataset import CompetitionDataset


def check_mask_integrity():
    counter = 0
    directory = './dataset/hidden'
    limit = -1
    video_folders = [str(f) for f in sorted(Path(directory).iterdir()) if f.is_dir()][:limit]

    # true_names = [f'video_{i}' for i in range(15000, 20000)]
    # diff = set(true_names) - set(video_folders)
    # print(video_folders)
    # print(diff)
    print(len(video_folders))


    # for index in tqdm(range(len(video_folders))):
    
    #     masks_dir = video_folders[index].joinpath("mask.npy")
    #     masks = np.load(masks_dir)

    #     counter += masks.shape[0]
    #     # print(masks.shape)
    # print(counter)
    

if __name__ == "__main__":
    # base_datadir = './dataset'
    # limit = -1
    # train_set = CompetitionDataset(os.path.join(base_datadir, 'train'), dataset_type='labeled', limit=limit) # we treat trainset as unlabeled here
    # val_set = CompetitionDataset(os.path.join(base_datadir, 'val'), dataset_type='labeled', limit=limit)
    # unlabeled_set = CompetitionDataset(os.path.join(base_datadir, 'unlabeled'), dataset_type='labeled', limit=limit)
    # # concat train and unet labeled unlabeled set together
    # augmented_set = ConcatDataset([train_set, unlabeled_set])

    # num_workers = 1
    # BATCH_SIZE = 6
    # dataloader_train = torch.utils.data.DataLoader(
    #     train_set, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=num_workers
    # )
    # dataloader_val = torch.utils.data.DataLoader(
    #     val_set, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=num_workers
    # )
    # dataloader_unlabeled = torch.utils.data.DataLoader(
    #     unlabeled_set, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=num_workers
    # )

    # dataloader_augmented = torch.utils.data.DataLoader(
    #     augmented_set, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=num_workers
    # )

    # hidden_set = CompetitionDataset(os.path.join(base_datadir, 'hidden'), dataset_type='hidden')
    # dataloader_hidden = torch.utils.data.DataLoader(
    #     hidden_set, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=num_workers
    # )

    # print('start checking')
    # for batch, value in enumerate(dataloader_val):
    #     print(f'----batch {batch}----')

    # check the shape of submission
    result = torch.load('./team_12.pt')
    print(result.shape)

    # check_mask_integrity()