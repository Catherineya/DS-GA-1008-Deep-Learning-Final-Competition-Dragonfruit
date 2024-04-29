import os

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
from pathlib import Path
from torchvision import transforms
from tqdm import tqdm
from typing import Dict

from dragonfruitvp.src.unet import UNet


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', default='dataset/unlabeled', type=str, help='path of the dataset to be labeled')
    parser.add_argument('--unet_weight', default='best_model.pth', type=str, help='path of the trained unet weight')
    args = parser.parse_args()
    
    model = UNet() 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    def extract_number(s):
                return int(str(s).split('_')[-1].rstrip('.png'))
    transformations = transforms.Compose([
        transforms.Resize((160, 240)),  # Example resize to match model input size
        transforms.ToTensor(),
    ])

    def transform_image(image):
        return transformations(image)

    # dataset_path = "dataset/unlabeled"
    dataset_path = args.dataset_path
    video_folders = [f for f in sorted(Path(dataset_path).iterdir()) if f.is_dir()]
    model.load_state_dict(torch.load(args.unet_weight))
    model.eval()
    with torch.no_grad():
        for i,folder in tqdm(enumerate(video_folders), total=len(video_folders)):
            preds_list = []
            imgs = sorted(folder.glob('*.png'), key=extract_number)
            for i,img in enumerate(imgs):
                image = transform_image(Image.open(img).convert('RGB')).unsqueeze(0).to(device)
                output = model(image)
                _, preds = torch.max(output, 1)
                preds = preds.squeeze().cpu().numpy()
                preds_list.append(preds)

            
            if len(preds_list) > 0:
                stacked_preds_numpy = np.stack(preds_list)
                assert stacked_preds_numpy.shape[0] == 22, f"Expected 22 predictions, got {stacked_preds_numpy.shape[0]}"
                save_path = os.path.join(folder, 'mask.npy')

            # print(stacked_preds_numpy.shape)
                np.save(save_path, stacked_preds_numpy)  # Save as numpy array
                # print(f"Saved predictions to {folder}")


