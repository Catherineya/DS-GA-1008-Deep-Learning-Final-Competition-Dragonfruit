import os 

import cv2
import numpy as np
import pandas as pd
import torch
import torchmetrics

from pathlib import Path
from torch import nn
from tqdm import tqdm

def load_gt(directory):
    transform_mask = lambda x: torch.from_numpy(x).long()
    video_folders = [f for f in sorted(Path(directory).iterdir()) if f.is_dir() and f.name != 'video_01370']
    total_masks = []
    for f in tqdm(video_folders):
        masks_dir = f.joinpath("mask.npy")
        masks = transform_mask(np.load(masks_dir))
        masks = masks[-1, :, :]
        # print(masks.shape)
        total_masks.append(masks)
    return torch.stack(total_masks, dim=0)

# def load_pred(directory):

class Metrics:
    def __init__(self, logts, pred, true):
        self.logits = logits
        self.pred = pred
        self.true = true

        # print('----shape check----')
        # print(self.logits.shape, self.pred.shape, self.true.shape)
    
    def calc(self, metric):
        if metric == 'iou':
            return self.calc_iou()
        elif metric == 'ssim':
            return self.calc_ssim()
        elif metric == 'ce':
            return self.calc_ce()
        elif metric == 'mse':
            return self.calc_mse()
        else:
            raise NotImplementedError

    def calc_iou(self):
        jaccard = torchmetrics.JaccardIndex(task="multiclass", num_classes=49)
        iou = jaccard(self.pred, self.true).item()
        return iou

    def calc_ssim(self):
        pred, true = self.pred.numpy(), self.true.numpy()
        C1 = (0.01 * 255)**2
        C2 = (0.03 * 255)**2

        img1 = pred.astype(np.float64)
        img2 = true.astype(np.float64)
        kernel = cv2.getGaussianKernel(11, 1.5)
        window = np.outer(kernel, kernel.transpose())

        mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
        mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
        mu1_sq = mu1**2
        mu2_sq = mu2**2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
        sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
        sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                                (sigma1_sq + sigma2_sq + C2))
        return ssim_map.mean()

    def calc_ce(self):
        criterion = nn.CrossEntropyLoss()
        ce = criterion(self.logits, self.true)
        return ce.item()

    def calc_mse(self):
        criterion = nn.MSELoss()
        mse = criterion(self.pred, self.true.float())
        return mse.item()

if __name__ == "__main__":
    ground_truth = load_gt('./dataset/val')
    # print(grount_truth.shape)

    TENSORS = [
        'finetune_e50lr3oc_vp_gsta',
        'mptrain_e10lr3oc_mp_gsta',
        'mptrain_e20lr3cos_mp2_gsta',
        'mptrain_e20lr3cos_mp2l_gsta',
    ]

    METRICS = [
        'ce',
        'mse',
        'ssim',
        'iou'
    ]

    table_dict = {'name': [t.replace('_', ' ') for t in TENSORS]}

    tensor_dir = './papervis/results'
    # ious = []
    # ssims = []
    for t in tqdm(TENSORS):
        save_path = os.path.join(tensor_dir, f'{t}.pt')
        # print(save_path)
        logits = torch.load(save_path).cpu()
        _, pred = torch.max(logits, dim=1)

        M = Metrics(logits, pred, ground_truth)
        for met in METRICS:
            res = M.calc(met)
            table_dict[met] = table_dict.get(met, []) + [res]

        # print('pred', pred.shape)
    #     ssims.append(calc_ssim(pred, ground_truth))
    #     ious.append(calc_iou(pred, ground_truth))
    #     # print('---------------')
    # table_dict['ssim'] = ssims
    # table_dict['iou'] = ious
    print(table_dict)

    # Convert dictionary to pandas DataFrame
    df = pd.DataFrame(table_dict)

    # Transpose the DataFrame
    df_transposed = df.transpose()

    # Convert transposed DataFrame to LaTeX table
    latex_table_transposed = df_transposed.to_latex(header=False)

    # Print or save the LaTeX table
    print(latex_table_transposed)
    