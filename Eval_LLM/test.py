import json


# Open and load the JSON file
with open('/Users/max/Desktop/Eval_LLM/Common-sense/Social_IQa/Social_IQa.json', 'r') as file:
    data = json.load(file)

# Get the first example's input
first_input = data['examples'][0]['input']

# Print the first input
print(first_input)