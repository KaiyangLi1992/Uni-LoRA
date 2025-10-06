#!/bin/bash

# base_lrs=("5e-3" "2e-3" "1e-3" "1e-2")
# head_lrs=("2e-3" "5e-3" "1e-2" "2e-2" "5e-2")
# epochs=(20)
# datasets=("oxfordpets" "standfordcars" "dtd" "eurosat")

# for dataset in "${datasets[@]}"; do
#   for base_lr in "${base_lrs[@]}"; do
#     for head_lr in "${head_lrs[@]}"; do
#       for epoch in "${epochs[@]}"; do

#         echo "🚀 Launching: dataset=$dataset, base_lr=$base_lr, head_lr=$head_lr, epoch=$epoch"

#         CUDA_VISIBLE_DEVICES=2,3,4,5,6 accelerate launch \
#           --multi_gpu \
#           --num_processes 5 \
#           fine_tuning_ViT_ds.py \
#             --dataset $dataset \
#             --base_lr $base_lr \
#             --head_lr $head_lr \
#             --num_train_epochs $epoch

#       done
#     done
#   done
# done

#!/bin/bash

# 
declare -A head_lrs
declare -A base_lrs

head_lrs["cifar100"]="0.01"
base_lrs["cifar100"]="0.01"

head_lrs["cifar10"]="0.05"
base_lrs["cifar10"]="0.01"

head_lrs["dtd"]="0.05"
base_lrs["dtd"]="0.01"

head_lrs["eurosat"]="0.05"
base_lrs["eurosat"]="0.01"

head_lrs["oxfordpets"]="0.005"
base_lrs["oxfordpets"]="0.005"

head_lrs["standfordcars"]="0.05"
base_lrs["standfordcars"]="0.01"

head_lrs["resisc45"]="0.05"
base_lrs["resisc45"]="0.01"

# datasets=("cifar100" "cifar10" "dtd" "eurosat" "oxfordpets" "standfordcars" "resisc45")
datasets=("oxfordpets")
seeds=(5)
epoch=20

for dataset in "${datasets[@]}"; do
  head_lr=${head_lrs[$dataset]}
  base_lr=${base_lrs[$dataset]}
  
  for seed in "${seeds[@]}"; do
    echo "🚀 Launching: dataset=$dataset, base_lr=$base_lr, head_lr=$head_lr, epoch=$epoch, seed=$seed"

    CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 accelerate launch \
      --multi_gpu \
      --num_processes 6 \
      fine_tuning_ViT_ds.py \
        --dataset $dataset \
        --base_lr $base_lr \
        --head_lr $head_lr \
        --num_train_epochs $epoch \
        --output_prefix seed$seed \
        --seed $seed
  done
done