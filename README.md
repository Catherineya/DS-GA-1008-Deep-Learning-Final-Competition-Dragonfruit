# CSCI-GA-2572-Deep-Learning-Final-Competition-Dragonfruit

**Team12: DragonFruit** - Yuanhe Guo, Jianing Zhang, Haotong Wu

*Ranked 2nd in 2024 Spring final competition*

## Getting Started

### Step1: Clone Repo
```
git clone https://github.com/RicercarG/CSCI-GA-2572-Deep-Learning-Final-Competition-Dragonfruit.git
```
```
cd CSCI-GA-2572-Deep-Learning-Final-Competition-Dragonfruit
```

### Step2: Prepare Dataset
Put dataset inside `dataset` folder, with structure being
```
CSCI-GA-2572-Deep-Learning-Final-Competition-Dragonfruit
└── dataset
    ├── hidden
    ├── train
    ├── unlabeled
    └── val
```

### Step3: Prepare conda environment
```
conda env create -n dfvp -f base_environment.yml
conda activate dfvp
```

### Step4: Train the U-Net
```
python dfUNet_train.py
```
Weights will be saved to `weights_hub/unet`

### Step5: Generate labels for unlabeled dataset
```
python dfLabeler.py --dataset_path './dataset/unlabeled' --unet_weight './weights_hub/unet/best_model.pth'
```

### Step6: Train SimMP2
All configurations could be found in `dragonfruitvp/custom_configs`. Set `test: False`, `submission: False` in training config. Adjust `gpus` and `num_workers` based on your gpu numbers.
```
python dfMP_train.py --model_config_file 'dragonfruitvp/custom_configs/model_configs/mpl_gsta.yaml' --training_config_file 'dragonfruitvp/custom_configs/training_configs/mptrain_e10lr3oc.yaml'
```
 Weights and logs will be saved to './weights_hub/<training_config>_<model_config>'. If `vis_val: True` in training config, then images for visualization during validation and test epochs will be saved to `vis_*` folder correspondingly.


### Step7: Get prediction results for hidden set
Label the hidden set
```
python dfLabeler.py --dataset_path './dataset/hidden' --unet_weight './weights_hub/unet/best_model.pth'
```

Go to the desired training config yaml file (`'dragonfruitvp/custom_configs/training_configs/mptrain_e10lr3oc.yaml'`), and set `test: True`, `submission: True`.
Then run dfMP_train again
```
python dfMP_train.py --model_config_file 'dragonfruitvp/custom_configs/model_configs/mpl_gsta.yaml' --training_config_file 'dragonfruitvp/custom_configs/training_configs/mptrain_e10lr3oc.yaml'
```
Result will be saved as `team_12.pt`.

## File Structure
- `weights_hub`: All model weights will be saved here. Could be changed in configuration.
- `lightning_logs`: Logs saved by pytorch lightning during training
- `vis_*`: Images and Masks for visualization.


# GCP Notes
## Connet to GCP
### Login burst
```
ssh burst
```
### Interactive runtime
CPU:
```
srun --partition=interactive --account csci_ga_2572_002-2024sp-x --pty /bin/bash
```
GPU:
```
srun --partition=n1s8-v100-1 --gres=gpu:1 --account csci_ga_2572_002-2024sp-x --time=04:00:00 --pty /bin/bash
```
### Send files
```
scp <NetID>@greene-dtn.hpc.nyu.edu:/path/to/files /home/<NetID>/
```
For sending the whole folder:
```
scp -r <NetID>@greene-dtn.hpc.nyu.edu:/path/to/folder /home/<NetID>/
```