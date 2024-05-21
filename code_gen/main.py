import json
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
import os

if os.path.exists("./saved_models") == False:
    os.mkdir("./saved_models")
if os.path.exists("./evaluation") == False:
    os.mkdir("./evaluation")

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
dataset = Dataset(tokenizer=tokenizer)
tokenized_train_dataset, tokenized_val_dataset, tokenized_test_dataset = (
    dataset.load_dataset_from_csv()
)
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
        output_dir=f"./model_{layers}_logs",
        evaluation_strategy="steps",
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=int(batch_size / 2),
        weight_decay=0.01,
        save_total_limit=5,
        eval_steps=250,
        logging_steps=1000,
        save_steps=1000,
        num_train_epochs=epochs,
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
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=3),
        ],
    )
    print("\nTraining the model\n")
    trainer.train()
    model.save_pretrained(f"./saved_models/model{layers}")
    model.save_pretrained_with_state(f"./saved_models/model{layers}_state")
    tokenizer.save_pretrained(f"./saved_models/model{layers}")

    print("\nGenerating code for testing the model out\n")
    for i in range(3):
        text = """
        public class Test {
            public static void main(String[] args){
                System.out.println("Hello World");
            }
        }
        """

        inputs = tokenizer(
            text, return_tensors="pt", padding=True, truncation=True, max_length=512
        )

        input_ids = inputs.input_ids
        input_ids = input_ids.to(model.device)

        outputs = model.generate(inputs=input_ids)
        out = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(out)
