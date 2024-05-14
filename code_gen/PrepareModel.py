from transformers import (
    AutoModelForSeq2SeqLM,
    EncoderDecoderConfig,
    T5Model,
)
from transformers import BertConfig, BertLMHeadModel, BertModel, RobertaModel
from CodeGen import CodeGen


class PreParedModel:
    def __init__(
        self, tokenizer, t5_model: str, bert_config: BertConfig, query_encoder: str
    ) -> None:
        self.tokenizer = tokenizer
        self.t5: T5Model = AutoModelForSeq2SeqLM.from_pretrained(t5_model)
        self.encoder = self.t5.get_encoder()
        self.decoder = BertLMHeadModel(bert_config)
        self.query_encoder = RobertaModel.from_pretrained(query_encoder)

    def prepare(self) -> CodeGen:
        self.encoder.resize_token_embeddings(len(self.tokenizer))

        self.query_encoder.resize_token_embeddings(len(self.tokenizer))

        self.decoder.resize_token_embeddings(len(self.tokenizer))

        config = EncoderDecoderConfig.from_encoder_decoder_configs(
            self.encoder.config, self.decoder.config
        )

        self.query_encoder.config.eos_token_id = self.tokenizer.sep_token_id
        self.query_encoder.config.pad_token_id = self.tokenizer.pad_token_id
        self.query_encoder.config.vocab_size = config.encoder.vocab_size
        self.query_encoder.config.sep_token_id = self.tokenizer.sep_token_id
        self.query_encoder.config.decoder_start_token_id = self.tokenizer.cls_token_id

        model = CodeGen(
            config,
            encoder=self.encoder,
            decoder=self.decoder,
            query_encoder=self.query_encoder,
        )

        model.config.eos_token_id = self.tokenizer.sep_token_id
        model.config.pad_token_id = self.tokenizer.pad_token_id
        model.config.vocab_size = model.config.encoder.vocab_size
        model.config.sep_token_id = self.tokenizer.sep_token_id
        model.config.decoder_start_token_id = self.tokenizer.cls_token_id
        model.generation_config.decoder_start_token_id = self.tokenizer.cls_token_id

        return model
