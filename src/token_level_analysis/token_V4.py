import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import torch
import numpy as np
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from scipy.special import softmax
from lime.lime_text import LimeTextExplainer

# Check if a GPU is available and if not, use a CPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Load the model and tokenizer
model_name = 'gpt2-large'
model = GPT2LMHeadModel.from_pretrained(model_name).to(device)  # Move the model to the GPU
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# Tokenize the input sentence
input_text = "Tracy didn't go home that evening and resisted Riley's attacks. What does Tracy need to do before this?"
input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)  # Move the data to the GPU

# Tokenize the output sentence
output_text = "Find somewhere to go"
output_ids = tokenizer.encode(output_text, return_tensors='pt').to(device)  # Move the data to the GPU

# Calculate the probability of the output sentence
prob = 1.0
for i in range(len(output_ids[0])):
    # Get the logits from the model
    with torch.no_grad():
        logits = model(input_ids)[0]

    # Convert logits to probabilities
    probabilities = softmax(logits.cpu().numpy(), axis=-1)

    # Multiply the probability of the next token in the output sentence
    prob *= probabilities[0, -1, output_ids[0, i].item()]  # Use .item() to get a Python number from a tensor

    # Add the next token to the input
    input_ids = torch.cat([input_ids, output_ids[0, i:i+1].unsqueeze(0)], dim=-1)

print(f'Probability of "{output_text}" is {prob}')

# The rest of the code...

# Create a LIME text explainer
explainer = LimeTextExplainer(class_names=["Make a new plan", "Go home and see Riley", "Find somewhere to go"])

# Define a prediction function for the explainer
# def predict_proba(texts, batch_size=64):
#     # Initialize an empty list to store the probabilities
#     probabilities = []

#     # Process the data in batches
#     for i in range(0, len(texts), batch_size):
#         # Get the current batch of texts
#         batch_texts = texts[i:i+batch_size]

#         # Tokenize the batch of texts
#         input_ids = tokenizer(batch_texts, return_tensors='pt', truncation=True, padding=True)['input_ids'].to(device)  # Move the data to the GPU

#         # Get the logits from the model
#         with torch.no_grad():
#             logits = model(input_ids)[0]

#         # Convert logits to probabilities
#         batch_probabilities = softmax(logits.cpu().numpy(), axis=-1)

#         # Flatten the batch probabilities
#         batch_probabilities = batch_probabilities.reshape(-1)

#         # Add the probabilities of the current batch to the list
#         probabilities.extend(batch_probabilities)

#     # Convert the list of probabilities to a numpy array
#     probabilities = np.array(probabilities)

#     # Return the probabilities of the last token
#     return probabilities[:, -1, :]

def predict_proba(texts, batch_size=32):  # Reduced batch size
    probabilities = []

    # Process the data in batches
    for i in range(0, len(texts), batch_size):
        # Get the current batch of texts
        batch_texts = texts[i:i+batch_size]

        # Tokenize the batch of texts
        input_ids = tokenizer(batch_texts, return_tensors='pt', truncation=True, padding=True)['input_ids'].to(device)  # Move the data to the GPU

        # Get the logits from the model
        with torch.no_grad():
            logits = model(input_ids)[0]

        # Convert logits to probabilities
        batch_probabilities = softmax(logits.cpu().numpy(), axis=-1)

        # Flatten the batch probabilities
        batch_probabilities = batch_probabilities.reshape(-1)

        # Add the probabilities of the current batch to the list
        probabilities.extend(batch_probabilities)

        # Delete batch_probabilities to free up memory
        del batch_probabilities

    # Convert the list of probabilities to a numpy array and reshape it to a 2D array
    probabilities = np.array(probabilities).reshape(len(texts), -1)

    return probabilities

# Generate an explanation
explanation = explainer.explain_instance(input_text, predict_proba, num_features=10)

# Show the explanation
# explanation.show_in_notebook(text=input_text, show_predicted_value=True)
explanation.save_to_file('explanation.html')
# explanation.as_list()