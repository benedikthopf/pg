#!/bin/bash

#SBATCH --gres=gpu:4
#SBATCH --time=72:00:00
#SBATCH --output=pg-%j.out

# cd $WORK2/projected-gan

# singularity run --nv -B $(pwd),$(pwd)/.. pg.simg python train.py --outdir=./training-runs/ --cfg=fastgan --data=./data/pokemon256.zip --gpus=2 --batch=64 --mirror=0 --batch-gpu=8 --kimg=10000 --snap=10 --nobench 1 --metrics fid10k_full,fid50k_full --desc=default_config_large_sa

# singularity run --nv -B $(pwd),$(pwd)/.. pg.simg python train.py --outdir=./training-runs/ --cfg=fastgan --data=./data/dogs_faces.zip --gpus=4 --batch=64 --mirror=1 --batch-gpu=4 --kimg=10000 --snap=24 --nobench 0 --metrics fid10k_full --slot_dim=256 --oc_type=DINOSAUR --desc=with_working_sre
# singularity run --nv -B $(pwd),$(pwd)/.. pg.simg python train.py --outdir=./training-runs/ --cfg=fastgan --data=./data/dogs.zip --gpus=4 --batch=64 --mirror=1 --batch-gpu=4 --kimg=10000 --snap=24 --nobench 0 --metrics fid10k_full --slot_dim=256 --oc_type=DINOSAUR --desc=with_working_sre
# singularity run --nv -B $(pwd),$(pwd)/.. pg.simg python train.py --outdir=./training-runs/ --cfg=fastgan --data=./data/cat_faces.zip --gpus=4 --batch=64 --mirror=1 --batch-gpu=4 --kimg=10000 --snap=24 --nobench 0 --metrics fid10k_full --slot_dim=256 --oc_type=DINOSAUR --desc=with_working_sre
# singularity run --nv -B $(pwd),$(pwd)/.. pg.simg python train.py --outdir=./training-runs/ --cfg=fastgan --data=./data/moongate.zip --gpus=4 --batch=64 --mirror=1 --batch-gpu=4 --kimg=10000 --snap=24 --nobench 0 --metrics fid10k_full --slot_dim=256 --oc_type=DINOSAUR --desc=with_working_sre
# singularity run --nv -B $(pwd),$(pwd)/.. pg.simg python train.py --outdir=./training-runs/ --cfg=fastgan --data=./data/fauvism.zip --gpus=4 --batch=64 --mirror=1 --batch-gpu=4 --kimg=10000 --snap=24 --nobench 0 --metrics fid10k_full --slot_dim=256 --oc_type=DINOSAUR --desc=with_working_sre
singularity run --nv -B $(pwd),$(pwd)/.. pg.simg python train.py --outdir=./training-runs/ --cfg=fastgan --data=../datasets/bedroom --gpus=4 --batch=64 --mirror=0 --batch-gpu=2 --kimg=10000 --snap=24 --nobench 0 --metrics fid10k_full --slot_dim=256 --oc_type=DINOSAUR --oc_path=../models/dinosaur_bedroom.ckpt --desc=sre_only_continue


# singularity run --nv -B $(pwd),$(pwd)/.. pg.simg python train.py --outdir=./training-runs/ --cfg=fastgan --data=./data/clever6.zip --gpus=4 --batch=64 --mirror=0 --batch-gpu=4 --kimg=10000 --snap=10 --nobench 0 --metrics fid10k_full --slot_dim=64 --oc_type=TI_SLOTS --desc=ti_slots_cc_disc_tf_gen_tf

# singularity run --nv -B $(pwd),$(pwd)/.. ../cliplike_autoencoder/dc.simg python test.py
