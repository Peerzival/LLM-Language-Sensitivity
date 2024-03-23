import torch
from lime.lime_text import LimeTextExplainer
from transformers import GPT2Tokenizer
import bigbench.models.huggingface_models as huggingface_models

def prob(inputs):
  text = [tokenizer.decode(tokenizer.encode(input)) for input in inputs]
  return torch.tensor([model.generate_text(t) for t in text])


model = huggingface_models.BIGBenchHFModel('gpt2-large')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2-large')

class_names = {1: "Make a new plan", 2: "Go home and see Riley", 3: "Find somewhere to go"}
input_text = "Tracy didn't go home that evening and resisted Riley's attacks. What does Tracy need to do before this?"
explainer = LimeTextExplainer(class_names=class_names)
explainer.explain_instance(input_text, prob)
explainer.show_in_notebook(text=True)