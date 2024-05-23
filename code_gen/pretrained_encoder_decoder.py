import os
from transformers import (
    EncoderDecoderModel,
    BertConfig,
    BertLMHeadModel,
    AutoModel,
    AutoTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
)
import json
from interface.config import Config
from Dataset import Dataset
from Metrics import Metrics

with open("./code_gen/model_config.json", "r") as f:
    config_data = json.load(f)

config = Config(
    tokenizer=config_data["tokenizer"],
    data_collator=config_data["data_collator"],
    lr=config_data["lr"],
    batch_size=config_data["batch_size"],
    dataset_path=config_data["dataset_path"],
    encoder_model=config_data["encoder_model"],
    query_encoder_model=config_data["query_encoder_model"],
    model=config_data["model"],
)

data_collator_str = config.data_collator
lr = config.lr
batch_size = config.batch_size
tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")

dataset = Dataset(tokenizer=tokenizer)
tokenized_train_dataset, tokenized_val_dataset, tokenized_test_dataset = (
    dataset.load_dataset(for_project_model=False)
)
compute_metrics = Metrics(tokenizer=tokenizer)
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=data_collator_str)

# Define a temporary directory to save the decoder
decoder_save_path = "./custom_decoder"

# Load the tokenizer

# Configure the decoder
decoder_config = BertConfig(
    vocab_size=len(tokenizer),
    hidden_size=768,
    num_hidden_layers=12,
    num_attention_heads=12,
    is_decoder=True,
    add_cross_attention=True,
)

# Initialize the decoder model with the decoder configuration
decoder = BertLMHeadModel(config=decoder_config)

# Save the custom decoder
decoder.save_pretrained(decoder_save_path)

# Load the pretrained encoder
encoder = AutoModel.from_pretrained("microsoft/codebert-base")

# Combine encoder and decoder into an EncoderDecoderModel
model = EncoderDecoderModel.from_encoder_decoder_pretrained(
    encoder_pretrained_model_name_or_path="microsoft/codebert-base",
    decoder_pretrained_model_name_or_path=decoder_save_path,
)

model.config.decoder_start_token_id = tokenizer.cls_token_id
model.config.pad_token_id = tokenizer.pad_token_id
model.config.vocab_size = model.config.decoder.vocab_size

training_args = Seq2SeqTrainingArguments(
    output_dir=f"./model_{12}_logs",
    evaluation_strategy="steps",
    learning_rate=lr,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=int(batch_size / 2),
    weight_decay=0.01,
    save_total_limit=5,
    eval_steps=250,
    logging_steps=1000,
    save_steps=1000,
    num_train_epochs=200,
    load_best_model_at_end=True,
    fp16=True,
    gradient_accumulation_steps=4,
    gradient_checkpointing=True,
    eval_accumulation_steps=2,
    # use_cpu=True  # Use CPU for evaluation
)
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,  # .select(range(1)),
    eval_dataset=tokenized_val_dataset,  # .select(range(1)),
    tokenizer=tokenizer,
    data_collator=data_collator,
    # compute_metrics=compute_metrics,
    callbacks=[
        EarlyStoppingCallback(early_stopping_patience=3),
    ],
)
print("\nTraining the model\n")
trainer.train()
model.save_pretrained(f"./saved_models/model{12}")
