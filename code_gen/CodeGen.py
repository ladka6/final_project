from transformers import (
    EncoderDecoderModel,
    PreTrainedModel,
)
from transformers.modeling_outputs import Seq2SeqLMOutput
from typing import Optional, Tuple, Union
import torch.nn as nn
import inspect
import torch
from BahdanauAttention import BahdanauAttention


class CodeGen(EncoderDecoderModel):
    def __init__(
        self,
        config,
        encoder: Optional[PreTrainedModel] = None,
        decoder: Optional[PreTrainedModel] = None,
        query_encoder: Optional[PreTrainedModel] = None,
        pad_token_id: Optional[int] = None,
        decoder_start_token_id: Optional[int] = None,
    ):
        super().__init__(config)
        if encoder is None:
            from transformers import AutoModel

            encoder = AutoModel.from_config(config.encoder)

        if decoder is None:
            from transformers import AutoModelForCausalLM

            decoder = AutoModelForCausalLM.from_config(config.decoder)

        self.query_encoder = query_encoder
        self.encoder = encoder
        self.decoder = decoder

        self.encoder.config = self.config.encoder
        self.decoder.config = self.config.decoder

        self.b_attention = BahdanauAttention(self.decoder.config.hidden_size)

        if (
            self.encoder.config.hidden_size != self.decoder.config.hidden_size
            and self.decoder.config.cross_attention_hidden_size is None
        ):
            self.enc_to_dec_proj = nn.Linear(
                self.encoder.config.hidden_size, self.decoder.config.hidden_size
            )

        if self.encoder.get_output_embeddings() is not None:
            raise ValueError(
                f"The encoder {self.encoder} should not have a LM Head. Please use a model without LM Head"
            )

        if self.query_encoder.get_output_embeddings() is not None:
            raise ValueError(
                f"The encoder {self.query_encoder} should not have a LM Head. Please use a model without LM Head"
            )

        decoder_signature = set(
            inspect.signature(self.decoder.forward).parameters.keys()
        )

        if "encoder_hidden_states" not in decoder_signature:
            raise ValueError(
                "The selected decoder is not prepared for the encoder hidden states to be passed. Please see the "
                "following discussion on GitHub: https://github.com/huggingface/transformers/issues/23350"
            )

        self.config.pad_token_id = pad_token_id
        self.config.decoder_start_token_id = decoder_start_token_id
        self.tie_weights()

    def tie_weights(self):
        if self.config.tie_encoder_decoder:
            decoder_base_model_prefix = self.decoder.base_model_prefix
            self._tie_encoder_decoder_weights(
                self.encoder,
                self.decoder._modules[decoder_base_model_prefix],
                self.decoder.base_model_prefix,
            )

    def shift_tokens_right(
        self, input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int
    ):
        """
        Shift input ids one token to the right.
        """
        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
        if decoder_start_token_id is None:
            raise ValueError(
                "Make sure to set the decoder_start_token_id attribute of the model's configuration."
            )
        shifted_input_ids[:, 0] = decoder_start_token_id

        if pad_token_id is None:
            raise ValueError(
                "Make sure to set the pad_token_id attribute of the model's configuration."
            )

        shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

        return shifted_input_ids

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        encoder_outputs: Optional[Tuple[torch.FloatTensor]] = None,
        past_key_values: Tuple[Tuple[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ):
        # This means model is in the training phase
        if encoder_outputs is None:  # Do this with dynamic padding
            query_ids = input_ids[:, 512:]
            input_ids = input_ids[:, :512]

            input_attention_mask = attention_mask[:, :512]
            query_attention_mask = attention_mask[:, 512:]

            encoder_outputs = self.encoder(
                input_ids=input_ids, attention_mask=input_attention_mask
            )

            encoder_hidden_states = encoder_outputs[0]

            query_outputs = self.query_encoder(
                input_ids=query_ids,
                attention_mask=query_attention_mask,  # Fix attention mask
            )

            query_hidden_states = query_outputs[0]

            cross_attn_out, _ = self.b_attention(
                query=encoder_hidden_states,
                keys=query_hidden_states,
            )

            if (labels is not None) and (
                decoder_input_ids is None and decoder_inputs_embeds is None
            ):
                decoder_input_ids = self.shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )
                if decoder_attention_mask is None:
                    decoder_attention_mask = (
                        decoder_input_ids.clone().detach() != self.config.pad_token_id
                    )

            decoder_outputs = self.decoder(
                input_ids=decoder_input_ids,
                attention_mask=decoder_attention_mask,
                encoder_hidden_states=cross_attn_out,
                # encoder_attention_mask=attention_mask,
            )

            loss = None
            if labels is not None:
                logits = decoder_outputs[0]
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(
                    logits.reshape(-1, self.decoder.config.vocab_size), labels.view(-1)
                )

            return Seq2SeqLMOutput(
                loss=loss,
                logits=decoder_outputs.logits,
            )
        # In the generation phase
        else:
            encoder_hidden_states = encoder_outputs[0]

            if (labels is not None) and (
                decoder_input_ids is None and decoder_inputs_embeds is None
            ):
                decoder_input_ids = self.shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )
                if decoder_attention_mask is None:
                    decoder_attention_mask = decoder_input_ids.new_tensor(
                        decoder_input_ids != self.config.pad_token_id
                    )

            decoder_outputs = self.decoder(
                input_ids=decoder_input_ids,
                attention_mask=decoder_attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=attention_mask,
            )

            loss = None
            if labels is not None:
                logits = decoder_outputs[0]
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(
                    logits.reshape(-1, self.decoder.config.vocab_size), labels.view(-1)
                )

            return Seq2SeqLMOutput(
                loss=loss,
                logits=decoder_outputs.logits,
            )
