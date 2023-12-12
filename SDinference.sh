#! /bin/bash
#SBATCH --job-name=Test
#################RESSOURCES#################
#SBATCH --partition=48-4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=1
############################################
#SBATCH --output=Eval_ExampleScript.out
#SBATCH --error=Eval_ExampleScript.err
#SBATCH -v

export MODEL_NAME="runwayml/stable-diffusion-v1-5"

source ~/anaconda3/etc/profile.d/conda.sh
conda activate Accelerate_SDTrain
###########################################
###############################
accelerate launch   --mixed_precision="fp16"  ./ZSSGAN/NADA+SD.py \
                                --pretrained_model_name_or_path=$MODEL_NAME \
                                --output_dir="./ZSSGAN/SDmodels/sd-model-NADAfinetuned" \
                                --cache_dir="./ZSSGAN/SDcache/" \
                                --use_ema \
                                --seed=41 \
                                --resolution=512 \
                                --train_batch_size=16 \
                                --gradient_accumulation_steps=4 \
                                --gradient_checkpointing \
                                --max_train_steps=20 \
                                --learning_rate=1e-04 \
                                --lr_scheduler="constant" --lr_warmup_steps=0 \
                                --max_grad_norm=1 \
                                --num_train_epochs=1 \
                                --num_steps_per_epoch=300 \
                                --validation_prompts="A image of a modern car in oil painting style"\
                                --source_class="photo" \
                                --target_class="oilPainting" \
                                --clip_models="ViT-B/16"
