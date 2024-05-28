from typing import Dict, List, Union, NamedTuple
from dataclasses import dataclass


@dataclass
class ModelConfig:
    num_layers: int
    num_attention_heads: int


@dataclass
class Config:
    tokenizer: str
    data_collator: str
    lr: float
    batch_size: int
    dataset_path: str
    encoder_model: str
    query_encoder_model: str
    model: List[Union[Dict[str, ModelConfig], int]]
