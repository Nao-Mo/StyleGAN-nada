#! /bin/bash
#SBATCH --job-name=Test
#################RESSOURCES#################
#SBATCH --partition=24-2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=1
############################################
#SBATCH --output=NADA.out
#SBATCH --error=NADA.err
#SBATCH -v

source ~/anaconda3/etc/profile.d/conda.sh
conda activate NADA
###########################################
###############################
python train.py --size 1024 \
                --batch 2 \
                --n_sample 35 \
                --output_dir ./NADA/output \
                --lr 0.002 \
                --frozen_gen_ckpt ./NADA/pretrained_model/stylegan2-ffhq-config-f.pt \
                --iter 301 \
                --source_class "Photo" \
                --target_class "Tolkien elf" \
                --auto_layer_k 18 \
                --auto_layer_iters 1 \
                --auto_layer_batch 8 \
                --output_interval 50 \
                --clip_models "ViT-B/32" "ViT-B/16" \
                --clip_model_weights 1.0 1.0 \
                --mixing 0.0\
                --save_interval 150
