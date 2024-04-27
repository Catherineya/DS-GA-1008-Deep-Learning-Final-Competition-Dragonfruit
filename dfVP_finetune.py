import os

import numpy as np
import yaml
import torch

from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path
from torchvision import transforms
from PIL import Image

from dragonfruitvp.src.finetuner import DragonFruitFinetune
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
    # custom_training_config['batch_size'] = 8

    # update the training config
    config.update(custom_training_config)
    # update the model config
    config.update(custom_model_config)
    # config['res_dir'] = 'pretrainedVP_weights'
    config['ex_name'] = config['training_config_file'][:-5].split('/')[-1] + '_' + config['model_config_file'][:-5].split('/')[-1]
    # print(config['ex_name'], type(config['ex_name']))

    config['vp_weight'] = os.path.join(config['res_dir'], config['pretrain'] + '_' + config['model_config_file'][:-5].split('/')[-1], 'checkpoints', 'best.ckpt')
    config['unet_weight'] = os.path.join(config['res_dir'],'unet', 'best_model.pth')

    # Define some hyper parameters
    BATCH_SIZE=custom_training_config['batch_size']

    limit = -1
    base_datadir = config['data_root']
    train_set = CompetitionDataset(os.path.join(base_datadir, 'train'), dataset_type='labeled', limit=limit) # we treat trainset as unlabeled here
    val_set = CompetitionDataset(os.path.join(base_datadir, 'val'), dataset_type='labeled', limit=limit)
    hidden_set = CompetitionDataset(os.path.join(base_datadir, 'hidden'), dataset_type='hidden', limit=limit)

    num_workers = config['num_workers']
    dataloader_train = torch.utils.data.DataLoader(
        train_set, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=num_workers
    )
    dataloader_val = torch.utils.data.DataLoader(
        val_set, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=num_workers
    )

    dataloader_hidden = torch.utils.data.DataLoader(
        hidden_set, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=num_workers
    )    


    exp = DragonFruitFinetune(args, dataloaders=(dataloader_train, dataloader_val, dataloader_hidden), strategy='auto')

    print('>'*35 + ' training ' + '<'*35)
    exp.train()
    
    print('>'*35 + ' validating ' + '<'*35)
    exp.val()

    # print('>'*35 + 'testing' + '<'*35)
    # results = exp.test()
    # print(results.shape)
    # file_path = "submission.pt"
    # torch.save(results, file_path)
