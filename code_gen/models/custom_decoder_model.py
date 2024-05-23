from transformers import PreTrainedModel, T5Config
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions


class CustomDecoderModel(PreTrainedModel):
    config_class = T5Config

    def __init__(self, config, decoder, lm_head):
        super().__init__(config)
        # Load the pre-trained T5 model
        # self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        # Extract the decoder from the T5 model
        self.decoder = decoder
        # Extract the LM head from the T5 model
        self.lm_head = lm_head

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        use_cache=None,
        past_key_values=None,
        return_dict=None,
        **kwargs_decoder,
    ):
        # Use the decoder to get the hidden states
        decoder_outputs = self.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            return_dict=return_dict,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            past_key_values=past_key_values,
        )
        sequence_output = decoder_outputs.last_hidden_state

        lm_logits = self.lm_head(sequence_output)

        return CausalLMOutputWithCrossAttentions(
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            hidden_states=decoder_outputs.hidden_states,
            attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
        )
