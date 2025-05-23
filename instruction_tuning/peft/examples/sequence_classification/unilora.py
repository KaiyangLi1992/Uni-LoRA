import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from peft import (
    get_peft_model,
    UniLoRAConfig,
    PeftType,
)

import evaluate
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"

def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if "lm_head" in lora_module_names:  # needed for 16-bit
        lora_module_names.remove("lm_head")
    return list(lora_module_names)



batch_size = 32
model_name_or_path = "roberta-large"
task = "mrpc"
peft_type = PeftType.UNILORA
device = "cuda:2"
num_epochs = 1
rank = 4
max_length = 128
num_vectors = 90
vector_length = 256*90
torch.manual_seed(0)


head_lr = 4e-3
vector_bank_lr = 5e-3
logits_lr = 0

if any(k in model_name_or_path for k in ("gpt", "opt", "bloom")):
    padding_side = "left"
else:
    padding_side = "right"

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, padding_side=padding_side)
if getattr(tokenizer, "pad_token_id") is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

datasets = load_dataset("glue", task)
metric = evaluate.load("glue", task)


def tokenize_function(examples):
    # max_length=None => use the model max length (it's actually the default)
    outputs = tokenizer(examples["sentence1"], examples["sentence2"], truncation=True, max_length=max_length)
    return outputs


tokenized_datasets = datasets.map(
    tokenize_function,
    batched=True,
    remove_columns=["idx", "sentence1", "sentence2"],
)

# We also rename the 'label' column to 'labels' which is the expected name for labels by the models of the
# transformers library
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")


def collate_fn(examples):
    return tokenizer.pad(examples, padding="longest", return_tensors="pt")


# Instantiate dataloaders.
train_dataloader = DataLoader(tokenized_datasets["train"], shuffle=True, collate_fn=collate_fn, batch_size=batch_size)
eval_dataloader = DataLoader(
    tokenized_datasets["validation"], shuffle=False, collate_fn=collate_fn, batch_size=batch_size
)

model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, return_dict=True, max_length=None)
modules = find_all_linear_names(model)
peft_config = UniLoRAConfig(
    task_type="SEQ_CLS", 
    r=rank,
    topk=2,
    target_modules=['key', 'value', 'query', 'output.dense', 'intermediate.dense'],
    num_vectors=num_vectors,
    vector_length=vector_length,# Set to True to reduce storage space. Note that the saved parameters cannot be used to resume training from checkpoints.
    unilora_dropout=0.,
    save_only_topk_weights = False
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
model.print_savable_parameters()



from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.trainer_pt_utils import get_parameter_names

decay_parameters = get_parameter_names(model, ALL_LAYERNORM_LAYERS)
decay_parameters = [name for name in decay_parameters if "bias" not in name]
vector_bank_parameters = [name for name, _ in model.named_parameters() if "vector_bank" in name]
logits_parameters = [name for name, _ in model.named_parameters() if "logits" in name ]

optimizer_grouped_parameters = [
    {
        "params": [p for n, p in model.named_parameters() if n in decay_parameters and \
                    n not in logits_parameters and n not in vector_bank_parameters],
        "weight_decay": 0.1,
        "lr": head_lr,
    },
    {
        "params": [p for n, p in model.named_parameters() if n not in decay_parameters and \
                    n not in logits_parameters and n not in vector_bank_parameters],
        "weight_decay": 0.0,
        "lr": head_lr,
    },
    {
        "params": [p for n, p in model.named_parameters() if n in vector_bank_parameters],
        "lr": vector_bank_lr,
        "weight_decay": 0.0,
    },

]

optimizer = AdamW(optimizer_grouped_parameters)
lr_scheduler = get_linear_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=0.06 * (len(train_dataloader) * num_epochs),
    num_training_steps=(len(train_dataloader) * num_epochs),
)

model.to(device)

for epoch in range(num_epochs):
    model.train()
    for step, batch in enumerate(tqdm(train_dataloader)):
        batch.to(device)
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

    model.eval()
    for step, batch in enumerate(tqdm(eval_dataloader)):
        batch.to(device)
        with torch.no_grad():
            outputs = model(**batch)
        predictions = outputs.logits.argmax(dim=-1)
        predictions, references = predictions, batch["labels"]
        metric.add_batch(
            predictions=predictions,
            references=references,
        )

    eval_metric = metric.compute()
    print(f"epoch {epoch}:", eval_metric)
save_path = "./saved_unilora_model"
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
from safetensors.torch import load_file

state_dict = load_file("./saved_unilora_model/adapter_model.safetensors")

print(state_dict.keys())  
# account_id = 'Kaiyang92' # your Hugging Face Hub account ID
# model.push_to_hub(f"{account_id}/roberta-large-peft-unilora")

# import torch
# from peft import PeftModel, PeftConfig
# from transformers import AutoTokenizer

# peft_model_id = f"{account_id}/roberta-large-peft-unilora"
# config = PeftConfig.from_pretrained(peft_model_id)
# inference_model = AutoModelForSequenceClassification.from_pretrained(config.base_model_name_or_path)
# tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

# # Load the model
# inference_model = PeftModel.from_pretrained(inference_model, peft_model_id)

# inference_model.to(device)
# inference_model.eval()
# for step, batch in enumerate(tqdm(eval_dataloader)):
#     batch.to(device)
#     with torch.no_grad():
#         outputs = inference_model(**batch)
#     predictions = outputs.logits.argmax(dim=-1)
#     predictions, references = predictions, batch["labels"]
#     metric.add_batch(
#         predictions=predictions,
#         references=references,
#     )

# eval_metric = metric.compute()
# print(eval_metric)
