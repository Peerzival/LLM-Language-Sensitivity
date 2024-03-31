import json
import bigbench.models.huggingface_models as huggingface_models
from captum.attr import Lime
import torch
from transformers import GPT2Tokenizer
import matplotlib.pyplot as plt

def predict(inputs):
    text = [tokenizer.decode(input) for input in inputs]
    return torch.tensor([model.cond_log_prob(t, "Find somewhere to go") for t in text])


input = """Tracy didn't go home that evening and resisted Riley's attacks. What does Tracy need to do before this?
Choice: Make a new plan
Choice: Go home and see Riley
Choice: Find somewhere to go"""

model = huggingface_models.BIGBenchHFModel('gpt2-large')

# Tokenize the input
tokenizer = GPT2Tokenizer.from_pretrained('gpt2-large')
input_ids = tokenizer.encode(input, return_tensors='pt')


lime = Lime(predict)

explanation = lime.attribute(inputs=input_ids.squeeze(), target=0, show_progress=True)


tokens = [tokenizer.decode(input_ids.squeeze()[i]) for i in range(len(input_ids.squeeze()))]
importance_scores = explanation.detach().numpy()

# # Create a bar chart
plt.figure(figsize=(10, 5))
plt.bar(tokens, importance_scores)
plt.xlabel('Tokens')
plt.ylabel('Importance')
plt.title('Token Importance')
plt.xticks(rotation=90)
plt.show()

# Print the explanation
# for i in range(len(input_ids.squeeze())):
#     print(
#         f"Token: {tokenizer.decode(input_ids.squeeze()[i])}, Importance: {explanation[i]}")


"""
Kendall drank everyday until their friends mentioned it was becoming a problem. What will Kendall want to do next? -> Go to a meeting
Token: K, Importance: 1.0211186408996582
Token: end, Importance: -2.0828235149383545
Token: all, Importance: -2.464488983154297
Token:  drank, Importance: -3.2338154315948486
Token:  everyday, Importance: -2.1999223232269287
Token:  until, Importance: -1.3461079597473145
Token:  their, Importance: -2.076770305633545
Token:  friends, Importance: -1.4823126792907715
Token:  mentioned, Importance: -1.4004733562469482
Token:  it, Importance: -2.216550588607788
Token:  was, Importance: -2.125145673751831
Token:  becoming, Importance: -3.4679512977600098
Token:  a, Importance: -0.8646241426467896
Token:  problem, Importance: -2.0892341136932373
Token: ., Importance: 0.003114504972472787
Token:  What, Importance: -2.106492280960083
Token:  will, Importance: -3.4151952266693115
Token:  Kendall, Importance: -1.6906898021697998
Token:  want, Importance: -1.6177031993865967
Token:  to, Importance: -1.7657827138900757
Token:  do, Importance: -2.0665459632873535
Token:  next, Importance: -0.18573890626430511
Token: ?, Importance: -1.3680957555770874
"""
"""
Tracy didn't go home that evening and resisted Riley's attacks. What does Tracy need to do before this?
Token: Tr, Importance: -2.664870023727417
Token: acy, Importance: -1.4026352167129517
Token:  didn, Importance: -7.147127628326416
Token: 't, Importance: -2.0149736404418945
Token:  go, Importance: -2.294536590576172
Token:  home, Importance: -1.0583951473236084
Token:  that, Importance: -1.5632743835449219
Token:  evening, Importance: -2.836550712585449
Token:  and, Importance: -1.0213130712509155
Token:  resisted, Importance: -1.293981909751892
Token:  Riley, Importance: 0.2464285045862198
Token: 's, Importance: -0.9724643230438232
Token:  attacks, Importance: -1.4826838970184326
Token: ., Importance: -0.014925955794751644
Token:  What, Importance: -2.0056350231170654
Token:  does, Importance: -2.623699903488159
Token:  Tracy, Importance: -0.6442837119102478
Token:  need, Importance: -1.8917872905731201
Token:  to, Importance: -1.4388587474822998
Token:  do, Importance: -2.129274845123291
Token:  before, Importance: -0.44853487610816956
Token:  this, Importance: -0.9855194091796875
Token: ?, Importance: -1.2732517719268799
"""
