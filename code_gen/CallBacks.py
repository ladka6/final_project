from transformers import TrainerCallback
import torch


class PrintPredictionsCallback(TrainerCallback):
    def __init__(self, num_examples=5):
        self.num_examples = num_examples

    def on_evaluate(
        self, args, state, control, model, tokenizer, eval_dataloader, **kwargs
    ):
        model.eval()
        examples_processed = 0

        for examples in eval_dataloader:
            inputs = examples["input_ids"]
            labels = examples["labels"]

            if examples_processed >= self.num_examples:
                break

            inputs = inputs.to(args.device)
            labels = labels.to(args.device)

            with torch.no_grad():
                # Generate predictions
                generated_ids = model.generate(
                    inputs[:, 512:], max_length=50, num_beams=5, early_stopping=True
                )

            # Decode generated sequences and original inputs
            label_texts = tokenizer.batch_decode(labels, skip_special_tokens=True)
            prediction_texts = tokenizer.batch_decode(
                generated_ids, skip_special_tokens=True
            )

            print("Labels:", label_texts)
            print("Predictions:", prediction_texts)
            print("-" * 50)
            examples_processed += inputs.size(0)
        model.train()
