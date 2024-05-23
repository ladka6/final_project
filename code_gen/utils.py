def generate_for_testing(model, tokenizer):
    print("\nGenerating code for testing the model out\n")
    for _ in range(3):
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

        outputs = model.generate(inputs=input_ids)
        out = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(out)
