#!/bin/bash

#SBATCH --gres=gpu:4
#SBATCH --time=72:00:00
#SBATCH --output=pg-%j.out

singularity run --nv -B $(pwd),$(pwd)/.. pg.simg python train.py --outdir=./training-runs/ --cfg=fastgan --data=./data/cat_faces.zip --gpus=4 --batch=64 --mirror=1 --batch-gpu=4 --kimg=500 --snap=24 --nobench 0 --metrics fid10k_full --slot_dim=256 --oc_type=DINOSAUR --desc=500kimg
