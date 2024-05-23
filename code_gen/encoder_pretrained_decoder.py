import os
from transformers import (
    EncoderDecoderModel,
    BertConfig,
    BertModel,
    AutoModel,
    AutoTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    T5Model,
    AutoModelForSeq2SeqLM,
    PreTrainedModel,
    T5Config,
)
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
import json
from interface.config import Config
from Dataset import Dataset
from Metrics import Metrics
from models.custom_decoder_model import CustomDecoderModel


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
tokenizer = AutoTokenizer.from_pretrained("Salesforce/codet5-small")

dataset = Dataset(tokenizer=tokenizer)
tokenized_train_dataset, tokenized_val_dataset, tokenized_test_dataset = (
    dataset.load_dataset(for_project_model=False)
)
compute_metrics = Metrics(tokenizer=tokenizer)
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=data_collator_str)

encoder_config = BertConfig(
    vocab_size=len(tokenizer),
    hidden_size=768,
    num_hidden_layers=1,
    num_attention_heads=1,
)

encoder = BertModel(config=encoder_config)

t5: T5Model = AutoModelForSeq2SeqLM.from_pretrained("Salesforce/codet5-small")
t5.resize_token_embeddings(len(tokenizer))
decoder = CustomDecoderModel(t5.config, t5.get_decoder(), t5.lm_head)
encoder.resize_token_embeddings(len(tokenizer))


model = EncoderDecoderModel(
    encoder=encoder,
    decoder=decoder,
)

model.config.eos_token_id = tokenizer.sep_token_id
model.config.pad_token_id = tokenizer.pad_token_id
model.config.vocab_size = model.config.encoder.vocab_size
model.config.sep_token_id = tokenizer.sep_token_id
model.config.decoder_start_token_id = tokenizer.cls_token_id
model.generation_config.decoder_start_token_id = tokenizer.cls_token_id
model.gradient_checkpointing = True

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
    num_train_epochs=1,
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
model.save_pretrained(f"./saved_models/encoder_pretrained_decoder{12}")
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
