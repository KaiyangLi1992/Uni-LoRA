declare -A head_lrs
declare -A base_lrs

head_lrs["cifar100"]="0.005"
base_lrs["cifar100"]="0.01"


head_lrs["cifar10"]="0.005"
base_lrs["cifar10"]="0.01"


head_lrs["dtd"]="0.01"
base_lrs["dtd"]="0.01"


head_lrs["eurosat"]="0.02"
base_lrs["eurosat"]="0.01"


head_lrs["oxfordpets"]="0.02"
base_lrs["oxfordpets"]="0.01"


head_lrs["standfordcars"]="0.01"
base_lrs["standfordcars"]="0.01"


head_lrs["resisc45"]="0.01"
base_lrs["resisc45"]="0.01"


head_lrs["fgvc"]="0.01"
base_lrs["fgvc"]="0.01"

# datasets=("cifar100" "cifar10" "dtd" "eurosat" "oxfordpets" "standfordcars" "resisc45")
datasets=("fgvc" "cifar100" "cifar10" "dtd" "eurosat" "oxfordpets" "standfordcars" "resisc45" )
seeds=(1 2 3 4)
epoch=20

for dataset in "${datasets[@]}"; do

  head_lr=${head_lrs[$dataset]}
  base_lr=${base_lrs[$dataset]}
  
  for seed in "${seeds[@]}"; do
    echo "🚀 Launching: dataset=$dataset, base_lr=$base_lr, head_lr=$head_lr, epoch=$epoch, seed=$seed"
    mkdir -p logs  #  logs 
    log_file="logs/large_${dataset}_seed${seed}_bl${base_lr}_hl${head_lr}.log"

    python fine_tuning_ViT_large.py \
        --dataset $dataset \
        --base_lr $base_lr \
        --head_lr $head_lr \
        --num_train_epochs $epoch \
        --output_prefix seed$seed \
        --seed $seed \
        > "$log_file" 2>&1
  done
done