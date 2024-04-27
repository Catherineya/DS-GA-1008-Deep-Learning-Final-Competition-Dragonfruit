#!/bin/bash

#SBATCH --job-name=dfVP_Pretrain
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64GB
#SBATCH --time=8:00:00         # max time for running this job
##SBATCH --gres=gpu:1           # request for any gpu available
##SBATCH --gres=gpu:v100:1     # [alternative]request for a specific gpu

tar -cvf dataset.tar dataset