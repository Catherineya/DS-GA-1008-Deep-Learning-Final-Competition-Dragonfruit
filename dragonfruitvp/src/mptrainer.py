import os.path as osp
import sys
import time

import argparse
import pytorch_lightning as pl
import pytorch_lightning.callbacks as plc
import torch

from fvcore.nn import FlopCountAnalysis, flop_count_table
from pytorch_lightning import seed_everything, Trainer

from dragonfruitvp.data.base_module import BaseDataModule
from dragonfruitvp.src.simmp import SimMP 
from dragonfruitvp.src.simmp2 import SimMP2
from dragonfruitvp.utils.callbacks import (SetupCallback, EpochEndCallback, BestCheckpointCallback)




class DragonFruitMPTrain:
    def __init__(self, args, dataloaders=None, strategy='auto'):
        self.args = args
        self.config = self.args.__dict__

        self.method = 'simmp'  # TODO
        self.args.method = self.args.method.lower() # TODO
        self._dist = self.args.dist

        base_dir = args.res_dir if args.res_dir is not None else 'work_dirs'
        save_dir = osp.join(base_dir, args.ex_name.split(args.res_dir+'/')[-1])
        ckpt_dir = osp.join(save_dir, 'checkpoints')

        seed_everything(self.args.seed)
        self.data = BaseDataModule(*dataloaders) #TODO: implement BaseDataModule later

        if self.args.method == 'simmp':
            self.method = SimMP(
                steps_per_epoch=len(self.data.train_loader),
                test_mean = self.data.test_mean,
                test_std = self.data.test_std,
                save_dir = save_dir, 
                **self.config,
            )
        elif self.args.method == 'simmp2':
            self.method = SimMP2(
                steps_per_epoch=len(self.data.train_loader),
                test_mean = self.data.test_mean,
                test_std = self.data.test_std,
                save_dir = save_dir, 
                **self.config,
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
    
    def test(self):
        if self.args.test == True:
            print('weight loaded')
            ckpt = torch.load(osp.join(self.save_dir, 'checkpoints', 'best.ckpt'))
            self.method.load_state_dict(ckpt['state_dict'])
        self.trainer.test(self.method, self.data)
    
    def display_method_info(self, args):
        device = torch.device(args.device)
        if args.device == 'cuda':
            assign_gpu = 'cuda:' + (str(args.gpus[0] if len(args.gpus) == 1 else '0'))
            device = torch.device(assign_gpu)
        T, C, H, W = args.in_shape
        input_dummy = torch.ones(1, args.pre_seq_length, C, H, W).to(device)
    
        dash_line = '-' * 80 + '\n'
        info = self.method.model.__repr__()
        flops = FlopCountAnalysis(self.method.model.to(device), input_dummy)
        flops = flop_count_table(flops)
        if args.fps:
            fps = measure_throughput(self.method.model.to(device), input_dummy)
            fps = 'Throughputs of {}: {:.3f}\n'.format(args.method, fps)
        else:
            fps = ''
        return info, flops, fps, dash_line
        


