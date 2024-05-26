import os
from code_gen.models.ProjectModel import ProjectModel
from code_gen.models.ProjectModelConfig import ProjectModelConfig
from transformers import AutoTokenizer
from Dataset import Dataset
import code_bert_score
import evaluate
from codebleu import calc_codebleu
import pandas as pd

bleu = evaluate.load("bleu")


def calculate_scores(example, tokenizer: AutoTokenizer, model: ProjectModel):
    input_code = example["lang1"]
    ref_code = example["lang2"]
    # print("Ref Code: ", ref_code)

    inputs = tokenizer(
        input_code,
        padding="max_length",
        max_length=512,
        truncation=True,
        return_tensors="pt",
    )
    input_ids = inputs.input_ids.to(model.device)

    outputs = model.generate(input_ids=input_ids, max_length=200)
    generated_code = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # print("Generated Code: ", generated_code)

    pred_results = code_bert_score.score(
        cands=[generated_code], refs=[ref_code], lang="python"
    )

    codebleu_result = calc_codebleu([ref_code], [generated_code], lang="python")

    bleu_result = bleu.compute(predictions=[generated_code], references=[[ref_code]])

    results = {
        "CodeBLEU": codebleu_result,
        "BLEU": bleu_result,
        "CodeBERTScore": {
            "precision": pred_results[0].item(),
            "recall": pred_results[1].item(),
            "f1": pred_results[2].item(),
            "hash": pred_results[3].item(),
        },
    }

    return results


def main():
    MODELS_PATH = os.path.join(os.getcwd(), "saved_models")
    saved_models = os.listdir(MODELS_PATH)
    tokenizer = AutoTokenizer.from_pretrained("Salesforce/codet5-small")
    dataset = Dataset(tokenizer=tokenizer)
    _, _, test_data = dataset.load_dataset_from_csv()

    all_results = []

    for model_name in saved_models:
        model_path = os.path.join(MODELS_PATH, model_name)
        print(f"Loading model from {model_path}")

        config = ProjectModelConfig.from_pretrained(model_path)
        model = ProjectModel.from_pretrained_safetensors(model_path, config=config)

        model.config.eos_token_id = tokenizer.sep_token_id
        model.config.pad_token_id = tokenizer.pad_token_id
        model.config.vocab_size = model.config.encoder["vocab_size"]
        model.config.sep_token_id = tokenizer.sep_token_id
        model.config.decoder_start_token_id = tokenizer.cls_token_id
        model.generation_config.decoder_start_token_id = tokenizer.cls_token_id

        for example in test_data:
            result = calculate_scores(example, tokenizer=tokenizer, model=model)
            result["Model"] = model_name
            all_results.append(result)

    df_results = pd.DataFrame(all_results)
    for col in ["CodeBLEU", "BLEU", "CodeBERTScore"]:
        df_temp = pd.json_normalize(df_results[col])
        df_temp.columns = [f"{col}.{subcol}" for subcol in df_temp.columns]
        df_results = df_results.drop(columns=[col]).join(df_temp)

    print(df_results.to_string(index=False))

    model_dir = os.path.join("evaluation", model_name)
    os.makedirs(model_dir, exist_ok=True)

    csv_filename = os.path.join(model_dir, f"{model_name}.csv")
    df_results.to_csv(csv_filename, index=False)
    print("Done!")


if __name__ == "__main__":
    main()
