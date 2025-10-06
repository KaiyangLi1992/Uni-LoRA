import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import transformers
import accelerate
from datasets import ClassLabel
import peft
from peft import (
    get_peft_model,
    UniLoRAConfig,
)
import random
import numpy as np
import torch

print(f"Transformers version: {transformers.__version__}")
print(f"Accelerate version: {accelerate.__version__}")
print(f"PEFT version: {peft.__version__}")
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--head_lr", type=float, default=3e-3)
parser.add_argument("--base_lr", type=float, default=4e-3)
parser.add_argument("--output_prefix", type=str, default="")
parser.add_argument("--rank", type=int, default=4)
parser.add_argument("--num_train_epochs", type=int, default=20)
parser.add_argument("--seed", type=int, default=42)
# parser.add_argument("--dataset", type=str, default="food101", choices=["food101", "cifar100", "flowers102", "resisc45"])
parser.add_argument("--dataset", type=str, default="standfordcars", choices=["oxfordpets","standfordcars","cifar10", "cifar100", "dtd","flowers102","eurosat", "resisc45"])

args_custom = parser.parse_args()
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(args_custom.seed)

model_checkpoint = "google/vit-base-patch16-224-in21k"  # pre-trained model from which to fine-tune

from datasets import load_dataset

dataset_name = args_custom.dataset
print(f"📦 Loading dataset: {dataset_name}")

if dataset_name == "cifar10":
    train_ds = load_dataset("cifar10", split="train")
    val_ds = load_dataset("cifar10", split="test")
elif dataset_name == "cifar100":
    train_ds = load_dataset("cifar100", split="train")
    val_ds = load_dataset("cifar100", split="test")
elif dataset_name == "flowers102":
    train_ds = load_dataset("oxford_flowers102", split="train")
    val_ds = load_dataset("oxford_flowers102", split="test")
elif dataset_name == "resisc45":
    train_ds = load_dataset("timm/resisc45", split="train")
    val_ds = load_dataset("timm/resisc45", split="test")
elif dataset_name == "oxfordpets":
    train_val_ds = load_dataset("timm/oxford-iiit-pet", split="train")
    train_valid_split = train_val_ds.train_test_split(test_size=0.1)
    val_ds = train_valid_split['test']
    train_ds = train_valid_split['train']
    test_ds = load_dataset("timm/oxford-iiit-pet", split="test")
elif dataset_name == "standfordcars":
    train_ds = load_dataset("tanganke/stanford_cars", split="train")
    val_ds = load_dataset("tanganke/stanford_cars", split="test")
elif dataset_name == "dtd":
    train_ds = load_dataset("tanganke/dtd", split="train")
    val_ds = load_dataset("tanganke/dtd", split="test")
elif dataset_name == "eurosat":
    train_ds = load_dataset("tanganke/eurosat", split="train")
    val_ds = load_dataset("tanganke/eurosat", split="test")
else:
    raise ValueError(f"Unsupported dataset: {dataset_name}")


# 
if "label" in train_ds.features:
    label_column = "label"
elif "fine_label" in train_ds.features:
    label_column = "fine_label"
else:
    raise ValueError(f"Cannot find label column in dataset: {train_ds.features.keys()}")

if isinstance(train_ds.features[label_column], ClassLabel):
    labels = train_ds.features[label_column].names
    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for i, label in enumerate(labels)}
else:
    # fallback: label
    unique_labels = set(train_ds[label_column])
    label2id = {str(i): i for i in range(len(unique_labels))}
    id2label = {i: str(i) for i in range(len(unique_labels))}

print(id2label[2])

from transformers import AutoImageProcessor,get_scheduler

image_processor = AutoImageProcessor.from_pretrained(model_checkpoint)
print(image_processor)


from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)

normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
train_transforms = Compose(
    [
        RandomResizedCrop(image_processor.size["height"]),
        RandomHorizontalFlip(),
        ToTensor(),
        normalize,
    ]
)

val_transforms = Compose(
    [
        Resize(image_processor.size["height"]),
        CenterCrop(image_processor.size["height"]),
        ToTensor(),
        normalize,
    ]
)

test_transforms = Compose(
    [
        Resize(image_processor.size["height"]),
        CenterCrop(image_processor.size["height"]),
        ToTensor(),
        normalize,
    ]
)
# 
if "image" in train_ds.features:
    image_column = "image"
elif "img" in train_ds.features:
    image_column = "img"
else:
    raise ValueError(f"Can't find image column in dataset: {train_ds.features.keys()}")

def preprocess(example_batch):
    image = example_batch["image"].convert("RGB")  # or use image_column if it's a variable
    example_batch["pixel_values"] = train_transforms(image)
    # example_batch["pixel_values"] = [train_transforms(image.convert("RGB")) for image in example_batch[image_column]]
    return example_batch

def preprocess_val(example_batch):
    image = example_batch["image"].convert("RGB")  # or use image_column if it's a variable
    example_batch["pixel_values"] = val_transforms(image)
    # example_batch["pixel_values"] = [val_transforms(image.convert("RGB")) for image in example_batch[image_column]]
    return example_batch

def preprocess_test(example_batch):
    image = example_batch["image"].convert("RGB")  # or use image_column if it's a variable
    example_batch["pixel_values"] = test_transforms(image)
    # example_batch["pixel_values"] = [test_transforms(image.convert("RGB")) for image in example_batch[image_column]]
    return example_batch

# split up training into training + validation


train_ds = train_ds.map(preprocess,batched=False)
val_ds = val_ds.map(preprocess,batched=False )
test_ds = test_ds.map(preprocess,batched=False)

train_ds.set_format(type="torch")
val_ds.set_format(type="torch") 
test_ds.set_format(type="torch") 

# train_ds = train_ds.map(preprocess_train, batched=True)
# val_ds = val_ds.map(preprocess_val, batched=True)
# train_ds.set_format(type="torch")
# val_ds.set_format(type="torch")


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )

from transformers import AutoModelForImageClassification, TrainingArguments, Trainer

model = AutoModelForImageClassification.from_pretrained(
    model_checkpoint,
    label2id=label2id,
    id2label=id2label,
    ignore_mismatched_sizes=True,  # provide this in case you're planning to fine-tune an already fine-tuned checkpoint
)
print_trainable_parameters(model)

from peft import LoraConfig, get_peft_model
config = UniLoRAConfig(
            r=args_custom.rank,
            vector_length=72000,
            unilora_dropout=0,
            target_modules=["query", "value"],
            num_vectors=1,
        )

lora_model = get_peft_model(model, config)
print_trainable_parameters(lora_model)


from transformers import TrainingArguments, Trainer
from torch.optim import AdamW

model_name = model_checkpoint.split("/")[-1]
batch_size = 128
optimizer = AdamW([
    {"params": lora_model.classifier.parameters(), "lr": 3e-3},
    {"params": [p for n, p in lora_model.named_parameters() if "classifier" not in n], "lr": 4e-3},
], weight_decay=0.0)
output_dir = f"./output/base_{args_custom.output_prefix}-{args_custom.dataset}-headlr{args_custom.head_lr}-baselr{args_custom.base_lr}-epoch{args_custom.num_train_epochs}\
    -seed_{args_custom.seed}"
class CustomTrainer(Trainer):
    def create_optimizer(self):
        # 
        head_lr = args_custom.head_lr
        base_lr = args_custom.base_lr
        weight_decay = 0.01

        classifier_params = []
        base_params = []

        for name, param in self.model.named_parameters():
            if "classifier" in name:
                param.requires_grad = True
                classifier_params.append(param)
                continue

            if not param.requires_grad:
                continue
            else:
                base_params.append(param)

        optimizer_grouped_parameters = [
            {"params": base_params, "lr": base_lr, "weight_decay": weight_decay},
            {"params": classifier_params, "lr": head_lr, "weight_decay": weight_decay},
        ]

        self.optimizer = AdamW(optimizer_grouped_parameters)

    def create_scheduler(self, num_training_steps: int, optimizer=None):
        # linear scheduler with warmup
        self.lr_scheduler = get_scheduler(
            name="linear",
            optimizer=self.optimizer,
            num_warmup_steps=int(self.args.warmup_ratio * num_training_steps),
            num_training_steps=num_training_steps,
        )


args = TrainingArguments(
    output_dir=output_dir,
    remove_unused_columns=False,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=4,
    per_device_eval_batch_size=batch_size,
    fp16=True,
    num_train_epochs=args_custom.num_train_epochs,
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    push_to_hub=False,
    label_names=["labels"],
    report_to="tensorboard",
    seed=args_custom.seed,
)


import numpy as np
import evaluate

metric = evaluate.load("accuracy")


# the compute_metrics function takes a Named Tuple as input:
# predictions, which are the logits of the model as Numpy arrays,
# and label_ids, which are the ground-truth labels as Numpy arrays.
def compute_metrics(eval_pred):
    """Computes accuracy on a batch of predictions"""
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return metric.compute(predictions=predictions, references=eval_pred.label_ids)





def make_collate_fn(label_column):
    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        labels = torch.tensor([example[label_column] for example in examples])
        return {"pixel_values": pixel_values, "labels": labels}
    return collate_fn

trainer = CustomTrainer(
    lora_model,
    args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=image_processor,
    compute_metrics=compute_metrics,
    data_collator=make_collate_fn(label_column),
)
train_results = trainer.train()

# trainer.evaluate(val_ds)
test_results = trainer.evaluate(test_ds)
print(test_results)