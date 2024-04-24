import time
import os

import argparse
import numpy as np
import pickle
import torch
import yaml

from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from tqdm import tqdm

from dragonfruitvp.src.pretrainer import DragonFruitPretrain
from dragonfruitvp.utils.parser import create_parser, default_parser
from dragonfruitvp.data.custom_dataset import CompetitionDataset
    

if __name__ == "__main__":

    args = create_parser().parse_args()
    config = vars(args)

    with open(config['model_config_file']) as model_config_file:
        custom_model_config = yaml.safe_load(model_config_file)
    
    with open(config['training_config_file']) as training_config_file:
        custom_training_config = yaml.safe_load(training_config_file)

    # update default parameters
    default_values = default_parser()
    for attribute in default_values.keys():
        if config[attribute] is None:
            config[attribute] = default_values[attribute]

    # update the training config
    config.update(custom_training_config)
    # update the model config
    config.update(custom_model_config)
    config['ex_name'] = config['model_config_file'][:-5].split('/')[-1] + '_' + config['training_config_file'][:-5].split('/')[-1]
    # print(config['ex_name'], type(config['ex_name']))
    print('model weights will be exported to: ', os.path.join(config['res_dir'], config['ex_name']))

    # Define some hyper parameters
    BATCH_SIZE=custom_training_config['batch_size']

    # load dataset
    limit = -1
    base_datadir = '/scratch/yg2709/CSCI-GA-2572-Deep-Learning-Final-Competition-Dragonfruit/dataset'
    train_set = CompetitionDataset(os.path.join(base_datadir, 'train'), dataset_type='unlabeled', limit=limit) # we treat trainset as unlabeled here
    val_set = CompetitionDataset(os.path.join(base_datadir, 'val'), dataset_type='unlabeled', limit=limit)
    unlabeled_set = CompetitionDataset(os.path.join(base_datadir, 'unlabeled'), dataset_type='unlabeled', limit=limit)

    dataloader_train = torch.utils.data.DataLoader(
        train_set, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=1
    )
    dataloader_val = torch.utils.data.DataLoader(
        val_set, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=1
    )
    dataloader_unlabeled = torch.utils.data.DataLoader(
        unlabeled_set, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=1
    )

    # start pretraining, use unlabeled as training data, val as validation, train as test
    vp = DragonFruitPretrain(args, dataloaders=(dataloader_unlabeled, dataloader_val, dataloader_train), strategy='auto')

    print('>'*35 + ' training ' + '<'*35)
    vp.train()

    print('>'*35 + ' testing  ' + '<'*35)
    vp.test()