import json
import os
from interface.config import Config

from transformers import (
    DataCollatorForSeq2Seq,
    BertConfig,
    AutoTokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    EarlyStoppingCallback,
)
from PrepareModel import PreParedModel
from Dataset import Dataset
from Metrics import Metrics
from CallBacks import PrintPredictionsCallback


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

tokenizer_str = config.tokenizer
data_collator_str = config.data_collator
lr = config.lr
batch_size = config.batch_size
dataset_path = config.dataset_path
encoder_model = config.encoder_model
query_encoder_model = config.query_encoder_model


tokenizer = AutoTokenizer.from_pretrained(tokenizer_str)
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=data_collator_str)
dataset = Dataset(tokenizer=tokenizer, file_path=dataset_path)
tokenized_dataset = dataset.load_dataset_from_csv()
compute_metrics = Metrics(tokenizer=tokenizer)


for i, model_config in enumerate(config.model):

    epochs = model_config["num_train_epochs"]
    layers = model_config["config"]["num_layers"]
    heads = model_config["config"]["num_attention_heads"]

    print(f"Training model with {layers} layers and {heads} heads for {epochs} epochs")
    configuration = BertConfig(
        vocab_size=len(tokenizer),
        hidden_size=768,
        num_hidden_layers=layers,
        num_attention_heads=heads,
        is_decoder=True,
        add_cross_attention=True,
    )

    model = PreParedModel(
        tokenizer=tokenizer,
        t5_model=encoder_model,
        bert_config=configuration,
        query_encoder=query_encoder_model,
    )

    model = model.prepare()

    training_args = Seq2SeqTrainingArguments(
        output_dir=f"./model_{i}",
        evaluation_strategy="steps",
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=0.01,
        save_total_limit=5,
        logging_steps=100,
        num_train_epochs=epochs,
        load_best_model_at_end=True,
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],  # .select(range(1)),
        eval_dataset=tokenized_dataset["test"],  # .select(range(1)),
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=3),
        ],
    )

    trainer.train()
