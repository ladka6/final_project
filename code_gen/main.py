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

tokenizer = AutoTokenizer.from_pretrained("Salesforce/codet5-small")

data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer, model="Salesforce/codet5-small"
)
dataset = Dataset(
    tokenizer=tokenizer,
    file_path="/Users/ladka6/Desktop/final_project/code_gen/data/preprocessed (6).csv",
)

tokenized_dataset = dataset.load_dataset_from_csv()

compute_metrics = Metrics(tokenizer=tokenizer)

configuration = BertConfig(
    vocab_size=len(tokenizer),
    hidden_size=768,
    num_hidden_layers=12,
    num_attention_heads=12,
    is_decoder=True,
    add_cross_attention=True,
)

model = PreParedModel(
    tokenizer=tokenizer,
    t5_model="Salesforce/codet5-base",
    bert_config=configuration,
    query_encoder="FacebookAI/roberta-base",
)

model = model.prepare()

training_args = Seq2SeqTrainingArguments(
    output_dir="./test",
    evaluation_strategy="steps",
    learning_rate=5e-4,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    weight_decay=0.01,
    save_total_limit=5,
    logging_steps=100,
    num_train_epochs=1,
    load_best_model_at_end=True,
)


trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"].select(range(3)),
    eval_dataset=tokenized_dataset["test"].select(range(3)),
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[
        EarlyStoppingCallback(early_stopping_patience=3),
    ],
)

trainer.train()
