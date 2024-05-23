from transformers import (
    PreTrainedModel,
    T5Config,
    RobertaConfig,
    BertConfig,
    T5Model,
    AutoModelForSeq2SeqLM,
    RobertaModel,
    BertLMHeadModel,
)
from transformers.modeling_outputs import Seq2SeqLMOutput
from typing import Optional, Tuple
import torch.nn as nn
import inspect
import torch
from code_gen.models.BahdanauAttention import BahdanauAttention
from code_gen.models.ProjectModelConfig import ProjectModelConfig
from safetensors.torch import load_file
import os


class ProjectModel(PreTrainedModel):
    config_class = ProjectModelConfig

    def __init__(self, config: ProjectModelConfig):
        super().__init__(config)
        if isinstance(config.encoder, dict):
            encoder_config = T5Config.from_dict(config.encoder)
        else:
            encoder_config = config.encoder

        if isinstance(config.query_encoder_config, dict):
            query_encoder_config = RobertaConfig.from_dict(config.query_encoder_config)
        else:
            query_encoder_config = config.query_encoder_config

        if isinstance(config.decoder, dict):
            decoder_config = BertConfig.from_dict(config.decoder)
        else:
            decoder_config = config.decoder

        t5: T5Model = AutoModelForSeq2SeqLM.from_config(encoder_config)
        self.encoder = t5.get_encoder()
        self.query_encoder = RobertaModel._from_config(query_encoder_config)
        self.b_attention = BahdanauAttention(decoder_config.hidden_size)
        self.decoder = BertLMHeadModel._from_config(decoder_config)

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

        # if self.query_encoder.get_output_embeddings() is not None:
        #     raise ValueError(
        #         f"The encoder {self.query_encoder} should not have a LM Head. Please use a model without LM Head"
        #     )

        decoder_signature = set(
            inspect.signature(self.decoder.forward).parameters.keys()
        )

        if "encoder_hidden_states" not in decoder_signature:
            raise ValueError(
                "The selected decoder is not prepared for the encoder hidden states to be passed. Please see the "
                "following discussion on GitHub: https://github.com/huggingface/transformers/issues/23350"
            )

        self.config.pad_token_id = config.pad_token_id
        self.config.decoder_start_token_id = config.decoder_start_token_id
        self.tie_weights()

    def tie_weights(self):
        if self.config.tie_encoder_decoder:
            decoder_base_model_prefix = self.decoder.base_model_prefix
            self._tie_encoder_decoder_weights(
                self.encoder,
                self.decoder._modules[decoder_base_model_prefix],
                self.decoder.base_model_prefix,
            )

    @classmethod
    def from_pretrained_with_state(cls, pretrained_model_name_or_path, config):
        # Instantiate the model with the given configuration
        model = cls(config)

        # Load the state dictionary
        state_dict = torch.load(
            os.path.join(pretrained_model_name_or_path, "pytorch_model.bin")
        )

        # Load the state dictionary into the model
        model.load_state_dict(state_dict)

        return model

    @classmethod
    def from_pretrained_safetensors(cls, pretrained_model_name_or_path, config):
        # Instantiate the model with the given configuration
        model = cls(config)

        # Load the weights using safetensors
        state_dict = load_file(f"{pretrained_model_name_or_path}/model.safetensors")

        # Load the weights into the model
        model.load_state_dict(state_dict, strict=False)

        return model

    def save_pretrained_with_state(self, save_directory):
        state_dict = self.state_dict()
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
        torch.save(state_dict, os.path.join(save_directory, "pytorch_model.bin"))
        self.config.save_pretrained(save_directory)

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

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        return super().from_pretrained(
            pretrained_model_name_or_path, *model_args, **kwargs
        )

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return self.shift_tokens_right(
            labels, self.config.pad_token_id, self.config.decoder_start_token_id
        )

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def get_input_embeddings(self):
        return self.encoder.get_input_embeddings()

    def get_output_embeddings(self):
        return self.decoder.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        return self.decoder.set_output_embeddings(new_embeddings)

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs,
    ):
        decoder_inputs = self.decoder.prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values
        )
        decoder_attention_mask = (
            decoder_inputs["attention_mask"]
            if "attention_mask" in decoder_inputs
            else None
        )

        input_dict = {
            "attention_mask": attention_mask,
            "decoder_attention_mask": decoder_attention_mask,
            "decoder_input_ids": decoder_inputs["input_ids"],
            "encoder_outputs": encoder_outputs,
            "past_key_values": decoder_inputs["past_key_values"],
            "use_cache": use_cache,
        }
        return input_dict

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
