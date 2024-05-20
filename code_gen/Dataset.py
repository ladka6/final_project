import torch
from datasets import load_dataset


class Dataset:
    def __init__(self, tokenizer, random_state=42) -> None:
        self.tokenizer = tokenizer
        self.random_state = random_state

    def load_dataset_from_csv(self, max_length=512):
        dataset = load_dataset("ladka6/code_dataset", split="train")
        dataset = dataset.filter(
            lambda example: all(value is not None for value in example.values())
        )

        # Split the dataset into training, validation, and test sets
        train_test_split = dataset.train_test_split(
            test_size=0.2, seed=self.random_state
        )
        train_val_split = train_test_split["train"].train_test_split(
            test_size=0.25, seed=self.random_state
        )

        train_dataset = train_val_split["train"]
        val_dataset = train_val_split["test"]
        test_dataset = train_test_split["test"]

        def tokenize_function(examples):
            inputs = self.tokenizer(
                examples["lang1"],
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )
            targets = self.tokenizer(
                examples["lang2"],
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )
            query = self.tokenizer(
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

        tokenized_train_dataset = train_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=train_dataset.column_names,
        )

        tokenized_val_dataset = val_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=val_dataset.column_names,
        )

        tokenized_test_dataset = test_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=test_dataset.column_names,
        )

        return tokenized_train_dataset, tokenized_val_dataset, tokenized_test_dataset
