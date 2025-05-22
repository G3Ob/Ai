# finetune_tinyllama.py

from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset, Dataset
from peft import get_peft_model, LoraConfig, TaskType
import json
import torch

# Load and tokenize data
def load_training_data(json_path):
    with open(json_path) as f:
        data = json.load(f)
    return Dataset.from_list(data)

model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

# Apply LoRA
peft_config = LoraConfig(
    r=8,
    lora_alpha=32,
    task_type=TaskType.CAUSAL_LM,
    lora_dropout=0.05,
    bias="none"
)
model = get_peft_model(model, peft_config)

dataset = load_training_data("qa_dataset.json")

def tokenize_fn(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=512)

tokenized_dataset = dataset.map(tokenize_fn)

# Training setup
training_args = TrainingArguments(
    output_dir="./tinyllama-finetuned",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_steps=10,
    save_total_limit=2,
    logging_steps=5,
    learning_rate=2e-4,
    fp16=torch.cuda.is_available(),
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
)

trainer.train()
trainer.save_model("./tinyllama-finetuned")
