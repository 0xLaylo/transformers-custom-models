"""
Example: Using custom models with Transformers
"""

from transformers import AutoTokenizer, AutoModelForCausalLM

def load_custom_model(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return tokenizer, model

def generate_text(tokenizer, model, prompt: str, max_length: int = 100):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=max_length)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

