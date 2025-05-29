#!/bin/bash

#########################################################################

export EXP_NUM="1-1"
gpu_list=(1 2 3)

#########################################################################
export MODEL_NAME="stable-diffusion-v1-5/stable-diffusion-v1-5"
learning_rate=1e-3
mkdir -p ./logs

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

export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"

n_gpus=${#gpu_list[@]}
job_count=0

run_job() {
    local placeholder_token=$1
    local initializer_token=$2
    local assigned_gpu=$3
    local current_date
    current_date=$(date +"%Y%m%d_%H%M%S")
    local log_file="./logs/training_${EXP_NUM}_${placeholder_token}_${current_date}.log"

    export CUDA_VISIBLE_DEVICES="$assigned_gpu"
    export WANDB_RUN_NAME="${placeholder_token}_${EXP_NUM}_${current_date}"

    echo "[$(date +"%Y%m%d_%H%M%S")] GPU ${assigned_gpu}에서 ${placeholder_token} 작업 시작" >> "${log_file}" 2>&1

    echo "[$(date +"%Y%m%d_%H%M%S")] Training 시작..." >> "${log_file}" 2>&1
    echo "Training for ${placeholder_token} -> GPU ${assigned_gpu}"
    accelerate launch ./experiment_scripts/${EXP_NUM}/train_${EXP_NUM}.py \
        --output_dir="experiment_outputs/${EXP_NUM}/${placeholder_token}" \
        >> "${log_file}" 2>&1

    
    echo "[$(date +"%Y%m%d_%H%M%S")] Training 완료. Inference 시작..." >> "${log_file}" 2>&1
    echo "Inference for ${placeholder_token} -> GPU ${assigned_gpu}"
    for step in {500..1500..500}; do
        python ./experiment_scripts/${EXP_NUM}/infer_${EXP_NUM}_batch_acc.py \
            --embed_dir="experiment_outputs/${EXP_NUM}/${placeholder_token}" \
            --output_dir="experiment_output_images/${EXP_NUM}/${placeholder_token}/${step}" \
            >> "${log_file}" 2>&1
    done

    echo "[$(date +"%Y%m%d_%H%M%S")] GPU ${assigned_gpu}에서 ${placeholder_token} 작업 종료" >> "${log_file}" 2>&1
}

for placeholder_token in "${ordered_keys[@]}"; do
    initializer_token="${token_pairs[$placeholder_token]}"

    while [ "$(jobs -r | wc -l)" -ge "$n_gpus" ]; do
        sleep 2
    done

    gpu_index=$(( job_count % n_gpus ))
    assigned_gpu=${gpu_list[$gpu_index]}

    run_job "$placeholder_token" "$initializer_token" "$assigned_gpu" &
    
    job_count=$((job_count + 1))
done

wait
echo "################## training, inference 완료 ##################"

current_date=$(date +"%Y%m%d_%H%M%S")
grid_log_file="./logs/z_img_grid_${EXP_NUM}_${current_date}.log"
echo "[$(date +"%Y%m%d_%H%M%S")] 이미지 그리드 생성 시작" >> "${grid_log_file}" 2>&1
python ./z_img_grid.py --exp_num ${EXP_NUM} >> "${grid_log_file}" 2>&1
echo "[$(date +"%Y%m%d_%H%M%S")] 이미지 그리드 생성 완료" >> "${grid_log_file}" 2>&1
