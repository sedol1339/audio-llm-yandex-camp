# llm.py
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_NAME = "yandex/YandexGPT-5-Lite-8B-instruct"



class TextGenerator:
    """
    An agent that generates realistic sentences using the Google Gemma model.
    """
    def __init__(self, model_id=MODEL_NAME):
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            device_map="cuda",
            torch_dtype="auto",
        )

    def generate_sentence(self, word: str) -> str:
        """
        Generates a single sentence for the given word.
        """
        # The prompt format required by the Gemma model.
        prompt_template = (
            "Напиши одно короткое, реалистичное предложение на русском языке, которое естественным образом включает слово '{word}'. "
            "Предложение должно звучать так, как будто его сказал обычный человек. Не пиши ничего кроме этого предложения."
        )
        prompt = prompt_template.format(word=word)

        messages = [{"role": "user", "content": prompt}]
        input_ids = self.tokenizer.apply_chat_template(
            messages, tokenize=True, return_tensors="pt"
        ).to("cuda")

        outputs = self.model.generate(input_ids, max_new_tokens=1024)
        return self.tokenizer.decode(outputs[0][input_ids.size(1) :], skip_special_tokens=True)
