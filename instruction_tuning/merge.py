
import os
from peft import PeftModel
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
base_model_path = "meta-llama/Llama-2-13b-hf"


base_save_dir = "/home/kli16/Uni-LoRA_iLoRA/instruction_tuning_normalized/instruction_tuning/output"

adapter_names = [
    # "llama2_13b_vb_lr1e-3",
    # "llama2_13b_vb_lr2e-3",
    # "llama2_13b_vb_lr4e-3",
    # "llama2_13b_vb_lr6e-4",
    "llama2_13b_vb_lr8e-4",
]


# 加载 tokenizer

for adapter_name in adapter_names:
    print(f"Processing {adapter_name}...")
    tokenizer_path = f"/home/kli16/Uni-LoRA_iLoRA/instruction_tuning_normalized/instruction_tuning/output/{adapter_name}"
    adapter_path = f"/home/kli16/Uni-LoRA_iLoRA/instruction_tuning_normalized/instruction_tuning/output/{adapter_name}/checkpoint-3171/adapter_model"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    output_path = os.path.join(base_save_dir, f"{adapter_name}_merged")

    # 加载 base model
    base_model = AutoModelForCausalLM.from_pretrained(base_model_path, device_map="cuda:7")
    base_model.resize_token_embeddings(len(tokenizer))

    # 加载 adapter 并合并
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model = model.merge_and_unload()

    # 保存模型和 tokenizer
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    print(f"Saved merged model to {output_path}")