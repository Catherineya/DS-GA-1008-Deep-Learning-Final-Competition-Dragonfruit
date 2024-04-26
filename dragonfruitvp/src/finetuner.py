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


class DragonFruitFinetune:
    def __init__(self, args, dataloaders=None, strategy='auto'):
        self.args = args
        self.config = self.args.__dict__

        # print('config file for evaluate: ', self.config)

        self._dist = self.args.dist
        self.device = torch.device('cuda' if torch.cuda.is_available else 'cpu') 

        base_dir = args.res_dir if args.res_dir is not None else 'weights_hub'
        save_dir = osp.join(base_dir, args.ex_name.split(args.res_dir+'/')[-1])
        ckpt_dir = osp.join(save_dir, 'checkpoints')

        seed_everything(self.args.seed)

        self.data = BaseDataModule(*dataloaders)
        self.method = SimUNet(
            steps_per_epoch = len(self.data.train_loader),
            test_mean = self.data.test_mean,
            test_std = self.data.test_std,
            save_dir = save_dir,
            # load_vp = True,
            # load_unet = True,
            # fix_vp = True,
            # fix_unet = False,
            # vp_weight = osp.join(ckpt_dir, 'best.ckpt'),
            # unet_weight = 'unet/best_model.pth',
            **self.config
        )

        callbacks, self.save_dir = self._load_callbacks(args, save_dir, ckpt_dir)
        self.trainer = self._init_trainer(self.args, callbacks, strategy)

    def _init_trainer(self, args, callbacks, strategy):
        return Trainer(devices = args.gpus,
                       max_epochs = args.epoch,
                       strategy = strategy,
                       accelerator = 'gpu',
                       callbacks = callbacks
        )

    def _load_callbacks(self, args, save_dir, ckpt_dir):
        method_info = None
        if self._dist == 0:
            if not self.args.no_display_method_info:
                method_info = self.display_method_info(args)
            
        setup_callback = SetupCallback(
            prefix = 'train' if (not args.test) else 'test',
            setup_time = time.strftime('%Y%m%d_%H%M%S', time.localtime()),
            save_dir = save_dir, 
            ckpt_dir = ckpt_dir,
            args = args,
            method_info = method_info,
            argv_content = sys.argv + [f"gpus: {torch.cuda.device_count()}"]
        )

        ckpt_callback = BestCheckpointCallback(
            monitor = args.metric_for_bestckpt,
            filename = 'best-{epoch:02d}-{val_loss:.3f}',
            mode = 'min',
            save_last = True,
            dirpath = ckpt_dir,
            verbose = True,
            every_n_epochs = args.log_step,
        )

        epochend_callback  = EpochEndCallback()
        callbacks = [setup_callback, ckpt_callback, epochend_callback]
        if args.sched:
            callbacks.append(plc.LearningRateMonitor(logging_interval=None))
        return callbacks, save_dir

    def train(self):
        self.trainer.fit(self.method, self.data)


    def val(self):
        self.method.to(self.device)
        self.method.eval()

        jaccard = torchmetrics.JaccardIndex(task="multiclass", num_classes=49).to(self.device)

        val_pred_iou = []
        val_true_iou = []
        for pre_seq, aft_seq, masks in tqdm(self.data.val_dataloader(), total=len(self.data.val_dataloader())):
            pre_seq, aft_seq, masks = pre_seq.to(self.device), aft_seq.to(self.device), masks.to(self.device)
            pmask, tmask_pre, tmask_aft = self.method(batch_x=pre_seq, batch_x_aft=aft_seq)
            # print(masks_pred1.shape, masks_pred2.shape, masks.shape)
            B, T, _, _, _ = pre_seq.shape

            _, pmask = torch.max(pmask, 1)
            _, tmask_aft = torch.max(tmask_aft, 1)
            # print('shape after argmax', masks_pred1.shape)
            _, H, W = pmask.shape
            pmask = pmask.reshape(B, T, H, W)[:,-1,:,:].squeeze(1)
            tmask_aft = tmask_aft.reshape(B, T, H, W)[:,-1,:,:].squeeze(1)
            masks = masks[:,-1,:,:].squeeze(1)
            val_pred_iou.append(jaccard(pmask, masks).cpu().item())
            val_true_iou.append(jaccard(tmask_aft, masks).cpu().item())


        mean_val_pred_iou = sum(val_pred_iou) / len(val_pred_iou)
        mean_val_true_iou = sum(val_true_iou) / len(val_true_iou)
        print('validation mean iou: ', mean_val_pred_iou, mean_val_true_iou)
        
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

    def display_method_info(self, args):
        device = torch.device(args.device)
        if args.device == 'cuda':
            assign_gpu = 'cuda:' + (str(args.gpus[0] if len(args.gpus) == 1 else '0'))
            device = torch.device(assign_gpu)
        T, C, H, W = args.in_shape
        pre_dummy = torch.ones(1, args.pre_seq_length, C, H, W).to(device)
        aft_dummy = torch.ones(1, args.pre_seq_length, C, H, W).to(device)
    
        dash_line = '-' * 80 + '\n'
        info = self.method.model.__repr__()
        flops = FlopCountAnalysis(self.method.model.to(device), (pre_dummy, aft_dummy))
        flops = flop_count_table(flops)
        if args.fps:
            fps = measure_throughput(self.method.model.to(device), (pre_dummy, aft_dummy))
            fps = 'Throughputs of {}: {:.3f}\n'.format(args.method, fps)
        else:
            fps = ''
        return info, flops, fps, dash_line





        


