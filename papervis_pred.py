import time
import os

import argparse
import numpy as np
import pickle
import torch
import yaml

from PIL import Image
from pathlib import Path

from dragonfruitvp.data.custom_dataset import CompetitionDataset
from dragonfruitvp.src.finetuner import DragonFruitFinetune
from dragonfruitvp.src.mptrainer import DragonFruitMPTrain
from dragonfruitvp.utils.parser import create_parser, default_parser
from dragonfruitvp.utils.vis import save_masks, save_images

CONFIGS = {
    'simunet': [
        ('finetune_e50lr3oc', 'vp_gsta'),
        ('finetune_e50lr3st', 'vp_gsta'),
    ],
    'simmp': [
        ('mptrain_e3lr3oc', 'mpl_gsta'),
        ('mptrain_e10lr3oc', 'mp_gsta'),
        ('mptrain_e20lr4st_eps1', 'mp_gsta'),
    ],
    'simmp2': [
        ('mptrain_e20lr3cos', 'mp2_gsta'),
        ('mptrain_e20lr3cos', 'mp2l_gsta'),
    ],
}

def load_model(training_config_file, model_config_file, dataloaders):
    args = create_parser().parse_args()
    config = vars(args)

    # print('--config check--', type(config), config)
    # print('--config check--', training_config_file, model_config_file)

    with open(model_config_file) as cfg_file:
        custom_model_config = yaml.safe_load(cfg_file)
    
    with open(training_config_file) as cfg_file:
        custom_training_config = yaml.safe_load(cfg_file)
    
    # update default parameters
    default_values = default_parser()
    # config = default_parser()
    for attribute in default_values.keys():
        if config[attribute] is None:
            config[attribute] = default_values[attribute]
    # update the training config
    config.update(custom_training_config)
    # update the model config
    config.update(custom_model_config)

    config['ex_name'] = str(training_config_file)[:-5].split('/')[-1] + '_' + str(model_config_file)[:-5].split('/')[-1]

    if config['method'].lower() == 'simvp':
        ltn_model = DragonFruitFinetune(args, dataloaders=dataloaders, strategy='auto')
    else:
        ltn_model = DragonFruitMPTrain(args, dataloaders=dataloaders, strategy='auto')
    
    return ltn_model


if __name__ == '__main__':
    limit = None
    base_datadir = './dataset'
    val_set = CompetitionDataset(os.path.join(base_datadir, 'val'), dataset_type='labeled', limit=limit)
    dataloader_val = torch.utils.data.DataLoader(
        val_set, batch_size=4, shuffle=False, pin_memory=True, num_workers=1
    )
    config_root_dir = './dragonfruitvp/custom_configs'
    save_root_dir = './papervis'
    results_dir = os.path.join(save_root_dir, 'results')
    image_dir = os.path.join(save_root_dir, 'images')
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(image_dir, exist_ok=True)
    for key in CONFIGS:
        for exp in CONFIGS[key]:
            training_config_file = os.path.join(config_root_dir, 'training_configs', f'{exp[0]}.yaml')
            model_config_file = os.path.join(config_root_dir, 'model_configs', f'{exp[1]}.yaml')
            
            dataloaders = (dataloader_val, dataloader_val, dataloader_val) # only the last slot is used
            model = load_model(training_config_file, model_config_file, dataloaders)
            # results = model.test()
            model.test()
            # if isinstance(results, tuple):
            #     results, frames = results
                # frame_save_path = os.path.join(image_dir,  f'{exp[0]}_{exp[1]}')
                # save_images(frames, frame_save_path, 'image')
            
            # mask_save_path = os.path.join(image_dir, f'{exp[0]}_{exp[1]}')
            # save_masks(results, mask_save_path, 'mask')
            # print('results shape:', results.shape)
            # tensor_save_path = os.path.join(results_dir, f'{exp[0]}_{exp[1]}.pt')
            # torch.save(results, tensor_save_path)
            # print(f'results saved to {tensor_save_path}')
    
