#!/bin/bash

# switch to the environment
source activate pdm_part2
cd "/home/sachahai/Documents/VAE_manifold/Code"

### Use case 1: training and pretraining from scratch ###
# 3 channel
# python refactor_run_2stageVAE.py --dataset "BBBC" --input_channel 3 --num_pretrain 5 --epochs 5 --out "/mnt/Linux_Storage/outputs/1_experiment" --eval True
python refactor_run_2stageVAE.py --dataset "new"  --out "/mnt/Linux_Storage/outputs/1_experiment"  --input_channel 4 --num_pretrain 2 --epochs 2 --eval True

# 4 channel
python refactor_run_2stageVAE.py --dataset "Felix_Full_64" --input_channel 4 --epochs 2 --out "/mnt/Linux_Storage/outputs/1_experiment" --eval True

### Use case 2: already have pretriand model, need to train and eval
python refactor_run_2stageVAE.py --dataset "Felix_Full_64" --input_channel 4 --epochs 2 --pretrained '/mnt/Linux_Storage/outputs/1_experiment/2StageVaDE 2021-11-07-23:13' --out "/mnt/Linux_Storage/outputs/1_experiment" --eval True

### Use case 3: already pretrained and trained, only need to eval (generate embeddings)
#python refactor_run_2stageVAE.py --dataset "Felix_Full_64" --input_channel 4 --epochs 2 --train False --saved_model_path '/mnt/Linux_Storage/outputs/1_experiment/2StageVaDE 2021-11-07-23:13/logs' --out "/mnt/Linux_Storage/outputs/1_experiment" --eval True

### Use casce 4: interpolation
# python Interpolation.py --dataset "Felix_Full_64" --input_channel 4 --saved_model_path '/mnt/Linux_Storage/outputs/1_experiment/twoStageInfoMaxVaDE_2/logs/last.ckpt' --out "/mnt/Linux_Storage/outputs/testInterpolation"

### Tensorboard ###
#tensorboard --logdir logs