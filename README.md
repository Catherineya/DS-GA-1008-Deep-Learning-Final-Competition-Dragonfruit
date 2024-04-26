# CSCI-GA-2572-Deep-Learning-Final-Competition-Dragonfruit

**Team12: DragonFruit** - Haotong Wu, Jianing Zhang, Yuanhe Guo

## Getting Started
### Dataset
Dataset are recommended to be put inside `dataset` folder.

### Pretrain model for video prediction
```
sbatch dfVP_pretrain.slurm
```
or 
```
python dfVP_pretrain.py \
    --model_config_file 'dragonfruitvp/custom_configs/model_configs/TODO.yaml'\
    --training_config_file 'dragonfruitvp/custom_configs/training_configs/pretrain_e3lr3_eps05.yaml'
```

### Finetune unet for mask prediction
```
sbatch dfVP_finetune.slrum
```
or
```
TODO
```

## File Structure
- `weights_hub`: All model weights will be saved here. Could be changed in configuration.
- `lightning_logs`: Logs saved by pytorch lightning during training

## Connet to GCP
### Login burst
```
ssh burst
```
### Interactive runtime
CPU:
```
srun --partition=interactive --account csci_ga_2572_002-2024sp-12 --time=04:00:00 --pty /bin/bash
```
GPU:
```
srun --partition=n1s8-v100-1 --gres=gpu:1 --account csci_ga_2572_002-2024sp-12 --time=04:00:00 --pty /bin/bash
```
### Send files
```
scp <NetID>@greene-dtn.hpc.nyu.edu:/path/to/files /home/<NetID>/
```
For sending the whole folder:
```
scp -r <NetID>@greene-dtn.hpc.nyu.edu:/path/to/folder /home/<NetID>/