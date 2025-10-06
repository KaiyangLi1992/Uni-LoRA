import transformers
import accelerate
import peft


print(f"Transformers version: {transformers.__version__}")
print(f"Accelerate version: {accelerate.__version__}")
print(f"PEFT version: {peft.__version__}")

model_checkpoint = "google/vit-base-patch16-224-in21k"  # pre-trained model from which to fine-tune

from datasets import load_dataset

dataset = load_dataset("food101", split="train[:5000]")

labels = dataset.features["label"].names
label2id, id2label = dict(), dict()
for i, label in enumerate(labels):
    label2id[label] = i
    id2label[i] = label

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


def preprocess_train(example_batch):
    """Apply train_transforms across a batch."""
    example_batch["pixel_values"] = [train_transforms(image.convert("RGB")) for image in example_batch["image"]]
    return example_batch


def preprocess_val(example_batch):
    """Apply val_transforms across a batch."""
    example_batch["pixel_values"] = [val_transforms(image.convert("RGB")) for image in example_batch["image"]]
    return example_batch

# split up training into training + validation
train_ds = load_dataset("food101", split="train")  # 
val_ds = load_dataset("food101", split="validation")  # 


train_ds.set_transform(preprocess_train)
val_ds.set_transform(preprocess_val)


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

config = LoraConfig(
    r=16,
    lora_alpha=16,
    target_modules=["query", "value"],
    lora_dropout=0.1,
    bias="none",
    modules_to_save=["classifier"],
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

class CustomTrainer(Trainer):
    def create_optimizer(self):
        # 
        head_lr = 3e-3
        base_lr = 4e-3
        weight_decay = 0.01

        classifier_params = []
        base_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "classifier" in name:
                classifier_params.append(param)
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
    f"{model_name}-finetuned-lora-food101",
    remove_unused_columns=False,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=4,
    per_device_eval_batch_size=batch_size,
    fp16=True,
    num_train_epochs=5,
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    push_to_hub=False,
    label_names=["labels"],
    report_to="none",
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


import torch


def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}

trainer = CustomTrainer(
    lora_model,
    args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=image_processor,
    compute_metrics=compute_metrics,
    data_collator=collate_fn,
)
train_results = trainer.train()

trainer.evaluate(val_ds)