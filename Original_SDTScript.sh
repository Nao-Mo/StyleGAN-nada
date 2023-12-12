#! /bin/bash
#SBATCH --job-name=Train_ExampleScript
#################RESSOURCES#################
#SBATCH --partition=48-4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=1
############################################
#SBATCH --output=Eval_ExampleScript.out
#SBATCH --error=Eval_ExampleScript.err
#SBATCH -v

. /usr/share/Modules/init/profile.sh
module load cuda/11.3
module load python/3.8.12

# source ~/anaconda3/etc/profile.d/conda.sh
# conda activate Accelerate_SDTrain

###########################################
###############################
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export dataset_name="lambdalabs/pokemon-blip-captions"

 accelerate launch train_text_to_image.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$dataset_name \
  --resolution=512 --center_crop --random_flip \
  --mixed_precision="fp16" \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --max_train_steps=5000 \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --output_dir="sd-pokemon-model" \
  --push_to_hub