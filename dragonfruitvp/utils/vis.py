import os

import torch
import numpy as np
import matplotlib.pyplot as plt

from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
from PIL import Image
from pathlib import Path
from tqdm import tqdm

def save_masks(masks, save_path, name='pmask'):
    os.makedirs(save_path, exist_ok=True)
    for i in range(masks.shape[0]):
        mask = to_pil_image(masks[i].byte().data)
        plt.imsave(os.path.join(save_path, f'{i}_{name}.png'), mask)