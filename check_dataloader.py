import os

import torch
from torch.utils.data import Dataset, ConcatDataset, DataLoader, random_split
from dragonfruitvp.data.custom_dataset import CompetitionDataset


if __name__ == "__main__":
    base_datadir = './dataset'
    limit = -1
    train_set = CompetitionDataset(os.path.join(base_datadir, 'train'), dataset_type='labeled', limit=limit) # we treat trainset as unlabeled here
    val_set = CompetitionDataset(os.path.join(base_datadir, 'val'), dataset_type='labeled', limit=limit)
    unlabeled_set = CompetitionDataset(os.path.join(base_datadir, 'unlabeled'), dataset_type='labeled', limit=limit)
    # concat train and unet labeled unlabeled set together
    augmented_set = ConcatDataset([train_set, unlabeled_set])

    num_workers = 1
    BATCH_SIZE = 6
    dataloader_train = torch.utils.data.DataLoader(
        train_set, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=num_workers
    )
    dataloader_val = torch.utils.data.DataLoader(
        val_set, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=num_workers
    )
    dataloader_unlabeled = torch.utils.data.DataLoader(
        unlabeled_set, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=num_workers
    )

    dataloader_augmented = torch.utils.data.DataLoader(
        augmented_set, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=num_workers
    )

    hidden_set = CompetitionDataset(os.path.join(base_datadir, 'hidden'), dataset_type='hidden')
    dataloader_hidden = torch.utils.data.DataLoader(
        hidden_set, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=num_workers
    )

    print('start checking')
    for batch, value in enumerate(dataloader_val):
        print(f'----batch {batch}----')