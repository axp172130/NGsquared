#!/bin/bash

#SBATCH --job-name=ys_GPUp40_efficientnetB4exCV67step1
#SBATCH --partition=GPUp40
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=03-00:00:00
#SBATCH --output=ys_GPUp40.%j_efficientnetB4exCV67step1.out
#SBATCH --error=ys_GPUp40.%j_efficientnetB4exCV67step1.err
#SBATCH --mail-user=yeeseng.ng@utsouthwestern.edu

module add python/3.6.4-anaconda

export CUDA_VISIBLE_DEVICES=0

source activate ${PWD}/RSNAihdEnv
cd 191005_ICH

#module load cuda90/9.0.176

python ys_231_efficientNetModel-pseudo3D-CV67_step1.py