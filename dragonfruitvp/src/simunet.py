import torch

from torch import nn

from dragonfruitvp.src.base_method import Base_method
from dragonfruitvp.src.simvp import SimVP_Model, SimVP
from dragonfruitvp.src.unet import UNet
from dragonfruitvp.utils.main import load_model_weights

class SimUNet(Base_method):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _build_model(self, **kwargs):
        return SimUNet_Model(**kwargs)
    
    def forward(self, batch_x, batch_x_aft=None, batch_y=None, **kwargs):
        pre_seq_length, aft_seq_length = self.hparams.pre_seq_length, self.hparams.aft_seq_length
        assert pre_seq_length == aft_seq_length
        if batch_x_aft is not None:
            pred_y1, pred_y2 = self.model(batch_x, batch_x_aft)
            return pred_y1, pred_y2
        else:
            pred_y = self.model(batch_x)
            return pred_y
    
    def training_step(self, batch, batch_idx):
        assert len(batch) == 3
        batch_x, batch_aft, batch_y = batch

        pred_y1, pred_y2 = self(batch_x, batch_aft, batch_y)
        loss = self.criterion(pred_y1, batch_y)


class SimUNet_Model(nn.Module):
    def __init__(self, vp_weight, unet_weight, load_vp=True, fix_vp=True, load_unet=False, fix_unet=False, **kwargs):
        super().__init__()
        self.vp = SimVP(**kwargs)
        self.unet = UNet()

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

    
    def forward(self, x_raw, x_aft, mask, **kwargs):
        x_pred = self.vp(x_raw)
        Y1 = self.unet(x_pred) # predict the mask using predicted frame
        if x_aft is not None:
            Y2 = self.unet(x_aft) # predict the mask using real frame
            return Y1, Y2
        else:
            return Y1

        
