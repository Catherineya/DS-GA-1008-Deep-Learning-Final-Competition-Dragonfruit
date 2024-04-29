#!/bin/bash

#SBATCH --job-name=dfMP_train
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64GB
#SBATCH --time=8:00:00         # max time for running this job
#SBATCH --gres=gpu:1           # request for any gpu available
##SBATCH --gres=gpu:v100:1     # [alternative]request for a specific gpu

# Singularity path
ext3_path=/scratch/yg2709/env_dl/overlay-50G-10M.ext3
sif_path=/scratch/yg2709/env_dl/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif

# start running
singularity exec --nv \
--overlay ${ext3_path}:ro \
${sif_path} /bin/bash -c "
    source /ext3/env.sh
    conda activate base
    python dfMP_train.py \
        --model_config_file 'dragonfruitvp/custom_configs/model_configs/gsta_mp.yaml'\
        --training_config_file 'dragonfruitvp/custom_configs/training_configs/mptrain_e10lr3_onecycle_hpc.yaml'
"