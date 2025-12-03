from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


class LocalDistilGPT2:
    def __init__(self, model_path: str):
        """
        init and load local distilgpt2 model
        """
        print(f"Loading model from: {model_path}")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path).to(self.device)
        self.model.eval()
        print(f"Model loaded on {self.device}")

    def generate(self, prompt: str, max_new_tokens: int = 100, temperature: float = 0.7):
        """
        generate answer based on prompt
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response


if __name__ == "__main__":
    # download model to local
    model_name = "distilgpt2"
    AutoTokenizer.from_pretrained(model_name).save_pretrained("../models/distilgpt2")
    AutoModelForCausalLM.from_pretrained(model_name).save_pretrained("../models/distilgpt2")

    # test
    gpt2 = LocalDistilGPT2("../models/distilgpt2")
    prompt = "Q: What is reinforcement learning?\nA:"
    gpt_response = gpt2.generate(prompt)
    print(gpt_response)
