#!/bin/bash

#SBATCH --partition=n1c24m128-v100-4
#SBATCH --account csci_ga_2572_002-2024sp-12
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=24
#SBATCH --time=12:00:00
#SBATCH --output=pretrain_%j.out
#SBATCH --error=pretrain_%j.err
#SBATCH --exclusive

mkdir /tmp/$USER
export SINGULARITY_CACHEDIR=/tmp/$USER

cp -rp /scratch/DL23SP/dataset_student.sqsh /tmp
echo "Dataset is copied to /tmp"

singularity exec --nv --bind /scratch --overlay /tmp/dataset_student.sqsh --overlay /scratch/yg2709/env_dl/overlay-50G-10M.ext3:ro /scratch/yg2709/env_dl/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif /bin/bash -c "
    source /ext3/env.sh
    conda activate dfvp
    python dfVP_pretrain.py \
        --model_config_file 'dragonfruitvp/custom_configs/model_configs/guo_gsta_l.yaml'\
        --training_config_file 'dragonfruitvp/custom_configs/training_configs/pretrain_e40lr3_eps1_gcp.yaml'
"
