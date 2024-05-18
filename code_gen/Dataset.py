import torch
from datasets import load_dataset


class Dataset:
    def __init__(self, tokenizer, file_path) -> None:
        self.tokenizer = tokenizer
        self.file_path = file_path

    def load_dataset_from_csv(self, max_length=512):
        dataset = load_dataset("csv", data_files=self.file_path, split="train")
        # dataset = load_dataset("ladka6/code_dataset",split="train")
        dataset = dataset.filter(
            lambda example: all(value is not None for value in example.values())
        )
        dataset = dataset.train_test_split(test_size=0.2)

        def tokenize_function(examples):
            inputs = self.tokenizer(  # x
                examples["lang1"],
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )
            targets = self.tokenizer(  # 512
                examples["lang2"],
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )
            query = self.tokenizer(  # 512
                examples["query"],
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            attention_mask = inputs.attention_mask
            query_attention_mask = query.attention_mask
            input_ids = inputs.input_ids
            query_ids = query.input_ids

            B, L = input_ids.shape
            new_tensor = torch.empty((B, L * 2), dtype=torch.long)

            for i in range(B):
                new_tensor[i] = torch.cat([input_ids[i], query_ids[i]])

            mask_batch, mask_len = attention_mask.shape
            attention_mask_tensor = torch.empty(
                (mask_batch, mask_len * 2), dtype=torch.long
            )

            for i in range(mask_batch):
                attention_mask_tensor[i] = torch.cat(
                    [attention_mask[i], query_attention_mask[i]]
                )

            return {
                "input_ids": new_tensor,
                "attention_mask": attention_mask_tensor,
                "labels": targets.input_ids,
            }

        tokenized_datasets = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset["train"].column_names,
        )

        return tokenized_datasets
