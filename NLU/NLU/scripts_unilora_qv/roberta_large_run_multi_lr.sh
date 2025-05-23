#!/bin/bash

vb_lrs=(2e-3 5e-3)
seeds=(1 2 3 4 5) 
lrs=(1e-4 2e-4 5e-4 1e-3 2e-3 5e-3 1e-2 2e-2)
gpus=(0 1 2 3 4 5 6 7)

# vb_lrs=(1e-3)
# seeds=(1) 
# lrs=(5e-4)
# gpus=(7)


# 定义函数：每个 GPU 负责一个 lr 的所有 seeds（串行执行）
run_on_gpu_sst2() {
 lr=$1
  gpu=$2

  echo "Launching 5 seeds for lr=$lr on GPU $gpu"

  for vb_lr in "${vb_lrs[@]}"; do
  for seed in "${seeds[@]}"; do
    outdir="./output/large_sst2_lr${lr}_vb_lr${vb_lr}_seed${seed}"

    CUDA_VISIBLE_DEVICES=$gpu \
    PYTHONHASHSEED=0 \
    CUBLAS_WORKSPACE_CONFIG=":16:8" \
    python examples/text-classification/run_glue.py \
      --model_name_or_path roberta-large \
      --task_name sst2 \
      --do_train \
      --do_eval \
      --max_seq_length 128 \
      --per_device_train_batch_size 32 \
      --learning_rate $lr \
      --learning_rate_vector_bank $vb_lr \
      --learning_rate_logits 0 \
      --num_train_epochs 20 \
      --output_dir $outdir \
      --logging_steps 10 \
      --logging_dir $outdir/log \
      --evaluation_strategy epoch \
      --save_strategy no \
      --warmup_ratio 0.06 \
      --vb_module value,query \
      --rank 4 \
      --topk 2 \
      --num_vectors 90 \
      --vector_length 23040 \
      --seed $seed \
      --weight_decay 0.1 
  done
  done
}

run_on_gpu_mrpc() {
  lr=$1
  gpu=$2

  echo "Launching 5 seeds for lr=$lr on GPU $gpu"

  for vb_lr in "${vb_lrs[@]}"; do
  for seed in "${seeds[@]}"; do
    outdir="./output/large_mrpc_lr${lr}_vb_lr${vb_lr}_seed${seed}"

    CUDA_VISIBLE_DEVICES=$gpu \
    PYTHONHASHSEED=0 \
    CUBLAS_WORKSPACE_CONFIG=":16:8" \
    python examples/text-classification/run_glue.py \
      --model_name_or_path roberta-large \
      --task_name mrpc \
      --do_train \
      --do_eval \
      --max_seq_length 128 \
      --per_device_train_batch_size 32 \
      --learning_rate $lr \
      --learning_rate_vector_bank $vb_lr \
      --learning_rate_logits 0 \
      --num_train_epochs 40 \
      --output_dir $outdir \
      --logging_steps 10 \
      --logging_dir $outdir/log \
      --evaluation_strategy epoch \
      --save_strategy no \
      --warmup_ratio 0.06 \
      --vb_module value,query \
      --rank 4 \
      --topk 2 \
      --num_vectors 90 \
      --vector_length 23040 \
      --seed $seed \
      --weight_decay 0.1 
  done
  done
}


run_on_gpu_rte() {
  lr=$1
  gpu=$2

  echo "Launching 5 seeds for lr=$lr on GPU $gpu"

  for vb_lr in "${vb_lrs[@]}"; do
  for seed in "${seeds[@]}"; do
    outdir="./output/large_rte_lr${lr}_vb_lr${vb_lr}_seed${seed}"

    CUDA_VISIBLE_DEVICES=$gpu \
    PYTHONHASHSEED=0 \
    CUBLAS_WORKSPACE_CONFIG=":16:8" \
    python examples/text-classification/run_glue.py \
      --model_name_or_path roberta-large \
      --task_name rte \
      --do_train \
      --do_eval \
      --max_seq_length 128 \
      --per_device_train_batch_size 32 \
      --learning_rate $lr \
      --learning_rate_vector_bank $vb_lr \
      --learning_rate_logits 0 \
      --num_train_epochs 40 \
      --output_dir $outdir \
      --logging_steps 10 \
      --logging_dir $outdir/log \
      --evaluation_strategy epoch \
      --save_strategy no \
      --warmup_ratio 0.06 \
      --vb_module value,query \
      --rank 4 \
      --topk 2 \
      --num_vectors 90 \
      --vector_length 23040 \
      --seed $seed \
      --weight_decay 0.1 
  done
  done
}


run_on_gpu_stsb() {
  lr=$1
  gpu=$2

  echo "Launching 5 seeds for lr=$lr on GPU $gpu"

  for vb_lr in "${vb_lrs[@]}"; do
  for seed in "${seeds[@]}"; do
    outdir="./output/large_stsb_lr${lr}_vb_lr${vb_lr}_seed${seed}"

    CUDA_VISIBLE_DEVICES=$gpu \
    PYTHONHASHSEED=0 \
    CUBLAS_WORKSPACE_CONFIG=":16:8" \
    python examples/text-classification/run_glue.py \
      --model_name_or_path roberta-large \
      --task_name stsb \
      --do_train \
      --do_eval \
      --max_seq_length 128 \
      --per_device_train_batch_size 32 \
      --learning_rate $lr \
      --learning_rate_vector_bank $vb_lr \
      --learning_rate_logits 0 \
      --num_train_epochs 40 \
      --output_dir $outdir \
      --logging_steps 10 \
      --logging_dir $outdir/log \
      --evaluation_strategy epoch \
      --save_strategy no \
      --warmup_ratio 0.06 \
      --vb_module value,query \
      --rank 4 \
      --topk 2 \
      --num_vectors 90 \
      --vector_length 23040 \
      --seed $seed \
      --weight_decay 0.1 
  done
  done
}

run_on_gpu_cola() {
  lr=$1
  gpu=$2

  echo "Launching 5 seeds for lr=$lr on GPU $gpu"

  for vb_lr in "${vb_lrs[@]}"; do
  for seed in "${seeds[@]}"; do
    outdir="./output/large_cola_lr${lr}_vb_lr${vb_lr}_seed${seed}"

    CUDA_VISIBLE_DEVICES=$gpu \
    PYTHONHASHSEED=0 \
    CUBLAS_WORKSPACE_CONFIG=":16:8" \
    python examples/text-classification/run_glue.py \
      --model_name_or_path roberta-large \
      --task_name cola \
      --do_train \
      --do_eval \
      --max_seq_length 128 \
      --per_device_train_batch_size 32 \
      --learning_rate $lr \
      --learning_rate_vector_bank $vb_lr \
      --learning_rate_logits 0 \
      --num_train_epochs 40 \
      --output_dir $outdir \
      --logging_steps 10 \
      --logging_dir $outdir/log \
      --evaluation_strategy epoch \
      --save_strategy no \
      --warmup_ratio 0.06 \
      --vb_module value,query \
      --rank 4 \
      --topk 2 \
      --num_vectors 90 \
      --vector_length 23040 \
      --seed $seed \
      --weight_decay 0.1 
  done
  done
}

run_on_gpu_qnli() {
  lr=$1
  gpu=$2

  echo "Launching 5 seeds for lr=$lr on GPU $gpu"

  for vb_lr in "${vb_lrs[@]}"; do
  for seed in "${seeds[@]}"; do
    outdir="./output/large_qnli_lr${lr}_vb_lr${vb_lr}_seed${seed}"

    CUDA_VISIBLE_DEVICES=$gpu \
    PYTHONHASHSEED=0 \
    CUBLAS_WORKSPACE_CONFIG=":16:8" \
    python examples/text-classification/run_glue.py \
      --model_name_or_path roberta-large \
      --task_name qnli \
      --do_train \
      --do_eval \
      --max_seq_length 128 \
      --per_device_train_batch_size 32 \
      --learning_rate $lr \
      --learning_rate_vector_bank $vb_lr \
      --learning_rate_logits 0 \
      --num_train_epochs 20 \
      --output_dir $outdir \
      --logging_steps 10 \
      --logging_dir $outdir/log \
      --evaluation_strategy epoch \
      --save_strategy no \
      --warmup_ratio 0.06 \
      --vb_module value,query \
      --rank 4 \
      --topk 2 \
      --num_vectors 90 \
      --vector_length 23040 \
      --seed $seed \
      --weight_decay 0.1 
  done
  done
}

for i in "${!lrs[@]}"; do
  run_on_gpu_sst2 ${lrs[$i]} ${gpus[$i]} &
done


# # 每个 GPU 的任务队列
# gpu_jobs_6=()
# gpu_jobs_7=()

# # 轮流把任务分配到两个队列中
# for i in "${!lrs[@]}"; do
#   lr=${lrs[$i]}
#   gpu_index=$((i % 2))
#   if [ "$gpu_index" -eq 0 ]; then
#     gpu_jobs_6+=("$lr")
#   else
#     gpu_jobs_7+=("$lr")
#   fi
# done

# # 执行任务函数（按 GPU 串行，但整体并发）
# run_serial_jobs_mrpc_on_gpu() {
#   local gpu=$1
#   shift
#   local lrs=("$@")
  
#   for lr in "${lrs[@]}"; do
#     echo "[GPU $gpu] Running lr=$lr..."
#     run_on_gpu_mrpc $lr $gpu
#   done
# }

# # 分别后台运行两张 GPU 的串行任务
# run_serial_jobs_mrpc_on_gpu 6 "${gpu_jobs_6[@]}" &
# run_serial_jobs_mrpc_on_gpu 7 "${gpu_jobs_7[@]}" &


# # 执行任务函数（按 GPU 串行，但整体并发）
# run_serial_jobs_rte_on_gpu() {
#   local gpu=$1
#   shift
#   local lrs=("$@")
  
#   for lr in "${lrs[@]}"; do
#     echo "[GPU $gpu] Running lr=$lr..."
#     run_on_gpu_rte $lr $gpu
#   done
# }

# # 分别后台运行两张 GPU 的串行任务
# run_serial_jobs_rte_on_gpu 6 "${gpu_jobs_6[@]}" &
# run_serial_jobs_rte_on_gpu 7 "${gpu_jobs_7[@]}" &


# # 执行任务函数（按 GPU 串行，但整体并发）
# run_serial_jobs_cola_on_gpu() {
#   local gpu=$1
#   shift
#   local lrs=("$@")
  
#   for lr in "${lrs[@]}"; do
#     echo "[GPU $gpu] Running lr=$lr..."
#     run_on_gpu_cola $lr $gpu
#   done
# }

# # 分别后台运行两张 GPU 的串行任务
# run_serial_jobs_cola_on_gpu 6 "${gpu_jobs_6[@]}" &
# run_serial_jobs_cola_on_gpu 7 "${gpu_jobs_7[@]}" &