# 2_finetune_peft.py

from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model
import torch
import torch.nn as nn

# === Paths & Config ===
MODEL_NAME = "mistralai/Mistral-7B-v0.1"
CSV_PATH = "/content/drive/MyDrive/football-llm/data/processed/football_qa.csv"
OUTPUT_DIR = "/content/drive/MyDrive/football-llm/model/fine_tuned"

# === Quantization Config ===
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

# === Load & Tokenize Dataset ===
dataset = load_dataset("csv", data_files=CSV_PATH)["train"]
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, padding_side="right")
tokenizer.pad_token = tokenizer.eos_token

def tokenize_function(examples):
    tokens = tokenizer(
        [f"Question: {q}\nAnswer: {a}" for q, a in zip(examples["question"], examples["answer"])],
        truncation=True,
        padding="max_length",
        max_length=512,
    )
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# === Load Base Model with Quantization ===
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16,
)

# === Apply LoRA ===
peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, peft_config)

# === Verify Trainable Params ===
model.print_trainable_parameters()

# === Data Collator ===
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True)

# === Custom Trainer to Ensure Loss ===
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs["labels"]
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=labels,
        )
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss

# === Training Arguments (pointing into Drive) ===
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=2,
    num_train_epochs=3,
    save_steps=100,
    logging_steps=50,
    learning_rate=2e-5,
    fp16=True,
    optim="paged_adamw_8bit",
)

# === Trainer ===
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
)

# === Train! ===
trainer.train()

# === Save LoRA Adapters & Tokenizer into Drive ===
model.save_pretrained(f"{OUTPUT_DIR}/adapter_model")
tokenizer.save_pretrained(f"{OUTPUT_DIR}/tokenizer")

# Save LoRA configuration
peft_config.save_pretrained(f"{OUTPUT_DIR}/adapter_model")