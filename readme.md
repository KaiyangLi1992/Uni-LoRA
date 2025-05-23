##  Uni-LoRA: One Vector is All You Need

This repo contains the source code of "Uni-LoRA: One Vector is All You Need".

Uni-LoRA is implemented following the standard interface of the 🤗 Hugging Face Parameter-Efficient Fine-Tuning (PEFT) library (see instruction_tuning/peft), making it easy to integrate into existing workflows. Our implementation is fully compatible with PEFT, and we plan to submit it for potential inclusion in the official PEFT library in the future.

 
<p align="center">
<img src="./unilora.png" alt="Uni-LoRA Architecture" width="300"/>
</p>

**Uni-LoRA**  introduces a fixed, sparse, and isometric projection matrix P^(D × d), where d<<D and each row contains exactly one nonzero entry. By multiplying P with a compact trainable vector θ_d (length d), Uni-LoRA reconstructs the full LoRA parameter θ_D (length D), enabling efficient fine-tuning with minimal trainable parameters and no architectural modifications.

Empirically, Uni-LoRA matches the performance of standard LoRA while updating only 0.52M parameters on GEMMA-7B — only 0.0061% of the base model size and 0.26% of the LoRA parameter count. 

## Steps to reproduce the results

## NLU
- Modified code for running experiments for Natural Language Understanding experiments.
- Adapted from [LoRA source code](https://github.com/microsoft/LoRA).
#### Create and activate conda env
```console
cd NLU/NLU
conda env create -f environment.yml
conda activate Uni_LoRA_NLU
```
#### Install the pre-requisites
uni-lora:
```console
pip install -e ..
```
NLU:
```console
pip install -e .
```
#### Start the experiments
The scripts are located in the "NLU/scripts_unilora_qv".

For example,
```console
./scripts_unilora_qv/roberta_base_mrpc.sh
```


## Instruction Tuning

- The code for running Llama2 is adapted from [qlora source code](https://github.com/artidoro/qlora).
- Fine-tuning the Llama2 model requires access to the model weights on HuggingFace. Ensure you have the access before running the code.
- The bitsandbytes package in the environment may require local compilation. Please refer to the [official installation guide](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/main/docs/source/installation.mdx) for detailed instructions.

#### Create and activate conda env
```console
cd instruction_tuning
conda create -n instruction_tuning python==3.10
conda activate instruction_tuning
```

#### Install the pre-requisites
```console
pip install -r requirements.txt
cd peft
pip install -e .
```

#### Start the experiments
The scripts are located in the "instruction_tuning/scripts" folder.

For example,
```console
cd instruction_tuning
./scripts/finetune_llama2_7b_unilora.sh
```

For evaluation, please use [LLM Judge](https://github.com/lm-sys/FastChat/tree/main/fastchat/llm_judge).

## Math Instruction Tuning
#### Create and activate conda env
```console
cd math_instruction_tuning
conda create -n math_instruction_tuning python==3.8.13
conda activate math_instruction_tuning
```

#### Install the pre-requisites
```console
pip install -r requirements.txt
cd peft
pip install -e .
```

#### Start the experiments
The scripts are located in the "instruction_tuning/scripts" folder.

For example,
```console
cd math_instruction_tuning
./run_instruction_tuning_unilora.sh
```
