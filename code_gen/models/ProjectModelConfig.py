from transformers import PretrainedConfig


class ProjectModelConfig(PretrainedConfig):
    model_type = "project_model_config"
    is_composition = True

    def __init__(self, encoder, decoder, query_encoder_config, **kwargs):
        super().__init__(**kwargs)
        self.query_encoder_config = query_encoder_config
        self.encoder = encoder
        self.decoder = decoder
        self.is_encoder_decoder = True
