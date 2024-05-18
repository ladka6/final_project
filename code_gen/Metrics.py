from transformers import EvalPrediction
import evaluate
import numpy as np


class Metrics:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.accuracy_metric = evaluate.load("accuracy")

    def __call__(self, eval_preds: EvalPrediction):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]

        preds_argmax = preds.argmax(axis=-1)

        processed_preds = preds_argmax.flatten()
        processed_labels = labels.flatten()

        if len(processed_preds) < len(processed_labels):
            processed_preds = np.pad(
                processed_preds, (0, len(processed_labels) - len(processed_preds))
            )

        mask = np.where(processed_preds != self.tokenizer.pad_token_id, 1, 0)
        weights = 1 / mask.sum()
        sample_weight = np.where(mask != 0, weights, 0)

        accuracy = self.accuracy_metric.compute(
            predictions=processed_preds,
            references=processed_labels,
            sample_weight=sample_weight,
        )

        return {"accuracy": accuracy["accuracy"]}
