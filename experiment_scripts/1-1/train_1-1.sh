#!/bin/bash

#######################################################################

export EXP_NUM="1-1"
export CUDA_VISIBLE_DEVICES=3

#######################################################################
mkdir -p ./logs

# 연관 배열 선언
declare -A token_pairs=(
    ["backpack_dog"]="backpack"
    ["berry_bowl"]="bowl"
    ["cat2"]="cat"
    ["clock"]="clock"
    ["grey_sloth_plushie"]="plushie"
    ["monster_toy"]="toy"
    ["poop_emoji"]="emoji"
    ["rc_car"]="toy"
    ["robot_toy"]="toy"
    ["teapot"]="teapot"
)

# 순서 배열 따로 선언
ordered_keys=(
    "backpack_dog"
    "berry_bowl"
    # "cat2"
    # "clock"
    # "grey_sloth_plushie"
    # "monster_toy"
    # "poop_emoji"
    # "rc_car"
    # "robot_toy"
    # "teapot"
)

for placeholder_token in "${ordered_keys[@]}"; do
    initializer_token="${token_pairs[$placeholder_token]}"
    current_date=$(date +"%Y%m%d_%H%M%S")
    log_file="./logs/training_${EXP_NUM}_${placeholder_token}_${current_date}.log"
    
    export WANDB_RUN_NAME="${placeholder_token}_${EXP_NUM}_${current_date}"
    echo "✅ Training for ${placeholder_token}"
    python ./experiment_scripts/${EXP_NUM}/train_${EXP_NUM}.py \
        --output_dir="experiment_outputs/${EXP_NUM}/${placeholder_token}" 

    
    echo "✅ Inference for ${placeholder_token}"
    for step in {100..200..100}; do
        python ./experiment_scripts/${EXP_NUM}/infer_${EXP_NUM}.py \
            --embed_dir="experiment_outputs/${EXP_NUM}/${placeholder_token}" \
            --output_dir="experiment_output_images/${EXP_NUM}/${placeholder_token}/${step}" 
    done
done

# python ./z_img_grid.py --exp_num ${EXP_NUM}