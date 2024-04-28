import os

import torch

from torch import nn

from dragonfruitvp.src.base_method import Base_method
from dragonfruitvp.src.simvp import SimVP_Model, SimVP
from dragonfruitvp.src.unet import UNet
from dragonfruitvp.utils.main import load_model_weights
from dragonfruitvp.utils.metrics import metric
from dragonfruitvp.utils.vis import save_masks, save_images

class SimUNet(Base_method):
    def __init__(self, alpha=2.0, beta=1.0, gamma=1.0, **kwargs):
        '''
        mask loss: alpha * pmask + beta * tmask_pre + gamma * tmask_aft
        '''
        super().__init__(**kwargs)
        self.criterion = nn.CrossEntropyLoss() # use crossentropy for unet
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def _build_model(self, **kwargs):
        return SimUNet_Model(**kwargs)
    
    def forward(self, batch_x, batch_x_aft=None, batch_y=None, **kwargs):
        pre_seq_length, aft_seq_length = self.hparams.pre_seq_length, self.hparams.aft_seq_length
        assert pre_seq_length == aft_seq_length
        # print('batch_x shape: ', batch_x.shape, 'batch_x_aft shape: ', batch_x_aft.shape)
        if batch_x_aft is not None:
            x_pred, pmask, tmask_pre, tmask_aft = self.model(batch_x, batch_x_aft)
            return x_pred, pmask, tmask_pre, tmask_aft
        else:
            x_pred, pmask = self.model(batch_x)
            return x_pred, pmask
    
    def training_step(self, batch, batch_idx):
        assert len(batch) == 3
        batch_x, batch_aft, batch_y = batch

        x_pred, pmask, tmask_pre, tmask_aft = self(batch_x, batch_aft, batch_y)

        B, T, H, W = batch_y.shape
        t = T // 2
        batch_py = batch_y[:,t:,:,:].reshape(B*t, H, W)
        batch_ty_pre = batch_y[:,:t,:,:].reshape(B*t, H, W)
        batch_ty_aft = batch_py

        loss = self.alpha * self.criterion(pmask, batch_py) + self.beta * self.criterion(tmask_pre, batch_ty_pre) + self.gamma * self.criterion(tmask_aft, batch_ty_aft)
        # loss = self.criterion(pred_y1, batch_y) + self.criterion(pred_y2, batch_y)
        assert loss is not None
        self.log('train_loss', loss, on_step=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # TODO: check if this works
        '''
        if dataset is labeled, we use it to train unet, the batch is [pre_seqs, aft_seqs, masks]
        if dataset is unlabeled, we use it to pretrain vp, batch is [pre_seqs, aft_seqs]
        hidden dataset will never be used for validation
        p: masks predicted with output frames from SimVP
        t: masks predicted with input ground truth frames
        '''
        assert len(batch) == 3
        batch_x, batch_aft, batch_y = batch
        x_pred, pmask, tmask_pre, tmask_aft = self(batch_x, batch_aft, batch_y)
        # print('prediction shape', pmask.shape, 'truth shape', tmask_pre.shape, tmask_aft.shape, 'mask shape', batch_y.shape)
        B, T, H, W = batch_y.shape
        t = T // 2
        batch_py = batch_y[:,t:,:,:].reshape(B*t, H, W)
        batch_ty_pre = batch_y[:,:t,:,:].reshape(B*t, H, W)
        batch_ty_aft = batch_py

        # print('pmask shape:', pmask.shape, 'batch_py shape:', batch_py.shape)

        loss = self.criterion(pmask, batch_py)
        assert loss is not None
        eval_res, eval_log = metric(
            pred = pmask.cpu().numpy(), 
            true = batch_py.cpu().numpy(), 
            mean = self.hparams.test_mean,
            std = self.hparams.test_std,
            metrics = self.metric_list,
            channel_names = self.channel_names,
            spatial_norm = self.spatial_norm,
            threshold = self.hparams.get('metric_threshold', None)
        )

        eval_res['val_loss'] = loss
        for key, value in eval_res.items():
            self.log(key, value, on_step=True, on_epoch=True, prog_bar=False)

        if self.vis_val:
            # save all images
            save_path = f'vis_finetune/batch_{batch_idx}'
            _, pmask_vis = torch.max(pmask, 1)
            _, tmask_pre_vis = torch.max(tmask_pre, 1)
            _, tmask_aft_vis = torch.max(tmask_aft, 1)
            save_masks(pmask_vis, save_path, 'aft_pmask')
            save_masks(tmask_pre_vis, save_path, 'pre_tmask')
            save_masks(tmask_aft_vis, save_path, 'aft_tmask')
            save_masks(batch_ty_pre, save_path, 'pre_label')
            save_masks(batch_ty_aft, save_path, 'aft_label')

            save_images(x_pred, save_path, 'pred')
            save_images(batch_aft.reshape(*x_pred.shape), save_path, 'true')

        return loss


class SimUNet_Model(nn.Module):
    def __init__(self, vp_weight, unet_weight, load_vp=True, fix_vp=True, load_unet=False, fix_unet=False, **kwargs):
        super().__init__()
        self.vp = SimVP(**kwargs)
        self.unet = UNet()

        print('vp config:', 'load_vp', load_vp, 'fix_vp', fix_vp, 'load_unet', load_unet, 'fix_unet', fix_unet)

        #load weights 
        if load_vp:
            assert vp_weight is not None
            self.vp = load_model_weights(self.vp, vp_weight, is_ckpt=True, fix=fix_vp)
        
        if load_unet:
            assert unet_weight is not None
            self.unet = load_model_weights(self.unet, unet_weight, is_ckpt=False, fix=fix_unet)
        
        # ckpt = torch.load(vp_weight)
        # self.vp.load_state_dict(ckpt['state_dict'])
        # if fix_vp:
        #     for param in self.vp.parameters():
        #         param.requires_grad = False

        # unet_statedict = torch.load(unet_weight)
        # self.unet.load_state_dict(unet_statedict)

    
    def forward(self, x_raw, x_aft, **kwargs):
        # print('shape of input x in simunet: ', x_raw.shape)
        B, T, C, H, W = x_raw.shape

        x_pred = self.vp(x_raw)
        x_pred = x_pred.reshape(B*T, C, H, W) # reshape the batch here

        pmask = self.unet(x_pred) # predict masks using predicted frame
        if x_aft is not None:
            x_aft = x_aft.reshape(B*T, C, H, W)
            tmask_pre = self.unet(x_raw.reshape(B*T, C, H, W)) # predict masks for first 11 frames
            tmask_aft = self.unet(x_aft) # predict masks for last 11 frames
            return x_pred, pmask, tmask_pre, tmask_aft
        else:
            return x_pred, pmask

        
