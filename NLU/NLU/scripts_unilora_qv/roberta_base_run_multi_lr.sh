vb_lrs=(1e-3)
seeds=(1 2 3 4 5) 
# seeds=(1) 
lrs=(1e-2 2e-2)
gpus=(6 7)

run_on_gpu_sst2() {
  lr=$1
  gpu=$2

  echo "Launching 5 seeds for lr=$lr on GPU $gpu"

  for vb_lr in "${vb_lrs[@]}"; do
  for seed in "${seeds[@]}"; do
    outdir="./output/base_sst2_lr${lr}_vb_lr${vb_lr}_seed${seed}"

    CUDA_VISIBLE_DEVICES=$gpu \
    PYTHONHASHSEED=0 \
    CUBLAS_WORKSPACE_CONFIG=":16:8" \
    python examples/text-classification/run_glue.py \
      --model_name_or_path roberta-base \
      --task_name sst2 \
      --do_train \
      --do_eval \
      --max_seq_length 512 \
      --per_device_train_batch_size 32 \
      --learning_rate $lr \
      --learning_rate_vector_bank $vb_lr \
      --learning_rate_logits 0 \
      --num_train_epochs 80 \
      --output_dir $outdir \
      --logging_steps 10 \
      --logging_dir $outdir/log \
      --evaluation_strategy epoch \
      --save_strategy no \
      --warmup_ratio 0.06 \
      --vb_module value,query \
      --rank 4 \
      --topk 2 \
      --num_vectors 60 \
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
    outdir="./output/base_mrpc_lr${lr}_vb_lr${vb_lr}_seed${seed}"

    CUDA_VISIBLE_DEVICES=$gpu \
    PYTHONHASHSEED=0 \
    CUBLAS_WORKSPACE_CONFIG=":16:8" \
    python examples/text-classification/run_glue.py \
      --model_name_or_path roberta-base \
      --task_name mrpc \
      --do_train \
      --do_eval \
      --max_seq_length 512 \
      --per_device_train_batch_size 32 \
      --learning_rate $lr \
      --learning_rate_vector_bank $vb_lr \
      --learning_rate_logits 0 \
      --num_train_epochs 30 \
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
    outdir="./output/base_cola_lr${lr}_vb_lr${vb_lr}_seed${seed}"

    CUDA_VISIBLE_DEVICES=$gpu \
    PYTHONHASHSEED=0 \
    CUBLAS_WORKSPACE_CONFIG=":16:8" \
    python examples/text-classification/run_glue.py \
      --model_name_or_path roberta-base \
      --task_name cola \
      --do_train \
      --do_eval \
      --max_seq_length 512 \
      --per_device_train_batch_size 32 \
      --learning_rate $lr \
      --learning_rate_vector_bank $vb_lr \
      --learning_rate_logits 0 \
      --num_train_epochs 80 \
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
    outdir="./output/base_qnli_lr${lr}_vb_lr${vb_lr}_seed${seed}"

    CUDA_VISIBLE_DEVICES=$gpu \
    PYTHONHASHSEED=0 \
    CUBLAS_WORKSPACE_CONFIG=":16:8" \
    python examples/text-classification/run_glue.py \
      --model_name_or_path roberta-base \
      --task_name qnli \
      --do_train \
      --do_eval \
      --max_seq_length 512 \
      --per_device_train_batch_size 32 \
      --learning_rate $lr \
      --learning_rate_vector_bank $vb_lr \
      --learning_rate_logits 0 \
      --num_train_epochs 25 \
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


