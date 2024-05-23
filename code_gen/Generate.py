from code_gen.models.ProjectModelConfig import ProjectModelConfig
from code_gen.models.ProjectModel import ProjectModel

from transformers import AutoTokenizer


tokenizer = AutoTokenizer.from_pretrained("Salesforce/codet5-small")
config = ProjectModelConfig.from_pretrained("./saved_models/model4")

# model = ProjectModel.from_pretrained("./saved_models/model4", config=config)
model = ProjectModel.from_pretrained_safetensors("./saved_models/model4", config=config)
model.config.eos_token_id = tokenizer.sep_token_id
model.config.pad_token_id = tokenizer.pad_token_id
# model.config.vocab_size = model.config.encoder.vocab_size
model.config.sep_token_id = tokenizer.sep_token_id
model.config.decoder_start_token_id = tokenizer.cls_token_id
model.generation_config.decoder_start_token_id = tokenizer.cls_token_id
text = """
        public class Test {
            public static void main(String[] args){
                System.out.println("Hello World");
            }
        }
        """

inputs = tokenizer(
    text, return_tensors="pt", padding=True, truncation=True, max_length=512
)

input_ids = inputs.input_ids
input_ids = input_ids.to(model.device)
outputs = model.generate(input_ids, max_length=100)
out = tokenizer.decode(
    outputs[0],
    skip_special_tokens=True,
)
print(out)
