from transformers import (
    AutoTokenizer,
    GPT2LMHeadModel,
    AutoConfig,
    DataCollatorForLanguageModeling,
)
from Dataset import Dataset
from utils import generate_for_testing

tokenizer = AutoTokenizer.from_pretrained(
    "huggingface-course/code-search-net-tokenizer"
)

dataset = Dataset(tokenizer=tokenizer)
tokenized_train_dataset, tokenized_val_dataset, tokenized_test_dataset = (
    dataset.load_dataset(for_project_model=False)
)

config = AutoConfig.from_pretrained(
    "gpt2",
    vocab_size=len(tokenizer),
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
)

model = GPT2LMHeadModel(config)

tokenizer.pad_token = tokenizer.eos_token
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

from transformers import Trainer, TrainingArguments

args = TrainingArguments(
    output_dir="codeparrot-ds",
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    evaluation_strategy="steps",
    eval_steps=1000,
    logging_steps=1000,
    gradient_accumulation_steps=8,
    num_train_epochs=200,
    weight_decay=0.1,
    warmup_steps=1_000,
    lr_scheduler_type="cosine",
    learning_rate=5e-4,
    save_steps=5_000,
    fp16=True,
)

trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=args,
    data_collator=data_collator,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_val_dataset,
)

model.save_pretrained("codegpt")
generate_for_testing(model, tokenizer)
