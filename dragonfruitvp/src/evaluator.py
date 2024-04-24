import os.path as osp
import sys
import time

import argparse
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import pytorch_lightning.callbacks as plc
import torch
import torchmetrics

from fvcore.nn import FlopCountAnalysis, flop_count_table
from pytorch_lightning import seed_everything, Trainer
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm

from dragonfruitvp.data.base_module import BaseDataModule
from dragonfruitvp.src.simvp import SimVP
from dragonfruitvp.src.unet import UNet
from dragonfruitvp.src.simunet import SimUNet
from dragonfruitvp.utils.callbacks import (SetupCallback, EpochEndCallback, BestCheckpointCallback)


class DragonFruitEvaluate:
    def __init__(self, args, dataloaders=None, strategy='auto'):
        self.args = args
        self.config = self.args.__dict__

        print('config file for evaluate: ', self.config)

        self._dist = self.args.dist
        self.device = torch.device('cuda' if torch.cuda.is_available else 'cpu') 

        base_dir = args.res_dir if args.res_dir is not None else 'work_dirs'
        save_dir = osp.join(base_dir, args.ex_name.split(args.res_dir+'/')[-1])
        ckpt_dir = osp.join(save_dir, 'checkpoints')

        self.data = BaseDataModule(*dataloaders)
        self.method = SimUNet(
            steps_per_epoch = len(self.data.train_loader),
            test_mean = self.data.test_mean,
            test_std = self.data.test_std,
            save_dir = save_dir,
            load_vp = True,
            load_unet = True,
            # vp_weight = osp.join(ckpt_dir, 'best.ckpt'),
            # unet_weight = 'unet/best_model.pth',
            **self.config
        )

        # unet_statedict = torch.load('unet/best_model.pth')
        # self.unet = UNet()
        # self.unet.load_state_dict(unet_statedict)

        # ckpt = torch.load(osp.join(ckpt_dir, 'best.ckpt')) #TODO
        # self.method.load_state_dict(ckpt['state_dict'])
        
    def val(self):
        self.method.to(self.device)
        self.method.eval()

        jaccard = torchmetrics.JaccardIndex(task="multiclass", num_classes=49).to(self.device)

        val_iou = []
        for pre_seq, aft_seq, masks in tqdm(self.data.val_dataloader(), total=len(self.data.val_dataloader())):
            pre_seq, aft_seq, masks = pre_seq.to(self.device), aft_seq.to(self.device), mask.to(self.device)
            masks_pred1, masks_pred2 = self.method(x_raw=pre_seq, x_aft=aft_seq)
            print(masks_pred1.shape, masks_pred2.shape)

        
        # for batch, (datas, masks) in tqdm(enumerate(self.data.val_dataloader()), total=len(self.data.val_dataloader())):
        #     datas, masks= datas.to(self.device), masks.to(self.device)
        #     frames_pred = self.method(datas)
        #     # print(frames_pred.shape)
        #     frames_pred = frames_pred[:, -1, :, :, :].squeeze(1)
        #     # labels = labels[:, -1, :, :, :].squeeze(1)
        #     # print(frames_pred.shape)
        #     masks_pred = self.unet(frames_pred).argmax(1)
        #     # print('mask_pred shape:', masks_pred.shape, 'mask_true shape:', masks.shape)

        #     val_iou.append(jaccard(masks_pred, masks).cpu().item())
            

        #     # frame_true = to_pil_image(labels[-1].cpu())
        #     frame_pred = to_pil_image(frames_pred[-1].cpu())
        #     mask_true = to_pil_image(masks[-1].byte().cpu().data)
        #     mask_pred = to_pil_image(masks_pred[-1].byte().cpu().data)
            
        #     # plt.imsave('exp_frame_true.png', np.array(frame_true))
        #     plt.imsave('exp_frame_pred.png', np.array(frame_pred))
        #     plt.imsave('exp_mask_true.png', mask_true)
        #     plt.imsave('exp_mask_pred.png', mask_pred)

        
        # mean_val_iou = sum(val_iou) / len(val_iou)
        # print('validation mean iou: ', mean_val_iou)
    
    def test(self):
        self.method.to(self.device)
        # self.unet.to(self.device)

        self.method.eval()
        # self.unet.eval()

        results = None
        for batch, (x, y) in tqdm(enumerate(self.data.test_dataloader()), total=len(self.data.test_dataloader())):
            x = x.to(self.device)
            frames_pred = self.method(x)
            # print(frames_pred.shape)
            frames_pred = frames_pred[:, -1, :, :, :].squeeze(1)
            masks_pred = self.unet(frames_pred).argmax(1)
            results = torch.cat((results, masks_pred), dim=0) if results is not None else masks_pred
        
        return results







        


