import os
import os.path as osp
import logging

import numpy as np
import pytorch_lightning as pl
import torch.nn as nn

from dragonfruitvp.utils.optim_scheduler import get_optim_scheduler
from dragonfruitvp.utils.metrics import metric
from dragonfruitvp.utils.vis import save_images

class Base_method(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.metric_list = kwargs['metrics']
        self.spatial_norm = False
        self.channel_names = None

        self.save_hyperparameters()
        self.model = self._build_model(**kwargs)
        self.criterion = nn.MSELoss()
        self.test_outputs = []

        self.vis_val = kwargs['vis_val'] # draw pictures during validation
        self.ex_name = kwargs['ex_name'] # for saving the visualized images

    def _build_model(self):
        raise NotImplementedError
    
    def configure_optimizers(self):
        optimizer, scheduler, by_epoch = get_optim_scheduler(
            self.hparams,
            self.hparams.epoch,
            self.model,
            self.hparams.steps_per_epoch
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch" if by_epoch else "step"
            },
        }
    
    def forward(self, batch):
        NotImplementedError
    
    def training_step(self, batch, batch_idx):
        NotImplementedError
    
    def validation_step(self, batch, batch_idx):
        NotImplementedError
    
    def test_step(self, batch, batch_idx):
        batch_x, batch_y = batch
        pred_y = self(batch_x, batch_y)
        outputs = {'inputs': batch_x.cpu().numpy(), 'preds': pred_y.cpu().numpy(), 'trues': batch_y.cpu().numpy()}
        self.test_outputs.append(outputs)
        return outputs
    
    def on_test_epoch_end(self):
        results_all = {}
        for k in self.test_outputs[0].keys():
            results_all[k] = np.concatenate([batch[k] for batch in self.test_outputs], axis=0)

        eval_res, eval_log = metric(
            pred = results_all['preds'], 
            true = results_all['trues'], 
            mean = self.hparams.test_mean,
            std = self.hparams.test_std,
            metrics = self.metric_list,
            channel_names = self.channel_names,
            spatial_norm = self.spatial_norm,
            threshold = self.hparams.get('metric_threshold', None)
        )

        results_all['metrics'] = np.array([eval_res[m] for m in self.metric_list])

        if self.trainer.is_global_zero:
            print(eval_log)
            logging.info(eval_log)
            folder_path = osp.join(self.hparams.save_dir, 'saved')
            if not osp.exists(folder_path):
                os.makedirs(folder_path)
        
            for np_data in ['metrics', 'inputs', 'trues', 'preds']:
                np.save(osp.join(folder_path, np_data + '.npy'), results_all[np_data])
        return results_all