#Example Use:
#python mirostat.py --num_tokens 200 --tau 3.0 --context "/context.txt"
import argparse
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import math


def estimate_s(prob):
  result = 0
  num = 0
  den = 0
  for i in range(100):
    b = prob[i]/prob[i+1]
    t = (i+2)/(i+1)
    num += math.log(b)*math.log(t)
    den += math.log(t)**2
  return num/den


def compute_k(n,s,tau):
    eps = s-1
    k = ((eps*(2**(tau)))/(1-n**(-eps)))**(1/s)
    k = round(k)
    return k


parser = argparse.ArgumentParser(description='Mirostat sampling')
parser.add_argument('--num_tokens', type=int, help='Number of tokens to be generated', default=200)
parser.add_argument('--tau', type=float, help='Target average surprise value', default=3.0)
parser.add_argument('--context', type=str, help='Provide the address to the file containing the context', default="/content/Context_Shannon.txt")

args = parser.parse_args()
print("Mirostat sampling!")

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained('gpt2')

target_surprise = args.tau
max_surprise = 2*target_surprise
error_surprise = 0
running_tot_surprise = 0
learning_rate = 1
num_tokens = args.num_tokens
n=50000 #Vocabulary size

file_string = args.context
f = open(file_string, "r")
context_text = f.read()
context = torch.tensor([tokenizer.encode(context_text)])
generated = []
prev = context
past = None

indices_surprise = []

######################################################
model.eval()

# If you have a GPU, put everything on cuda
# context = context.to('cuda')
# model.to('cuda')

with torch.no_grad():

    for i in range(num_tokens):
        forward = model(input_ids=context, past_key_values=past, return_dict=True)
        logits = forward.logits[0, -1, :]
        past = forward.past_key_values

        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        prob_original = torch.softmax(sorted_logits, dim=-1).tolist()

        # Estimate s
        s = estimate_s(prob_original)
        # Compute k
        k = compute_k(n,s,max_surprise)+1

        sorted_logits = sorted_logits[0:k]
        sorted_indices = sorted_indices[0:k]

        prob_topk = torch.softmax(sorted_logits, dim = 0)
        prev_i = torch.multinomial(prob_topk, num_samples=1, replacement=True)
        index_surprise = math.log2(1/prob_original[prev_i])
        indices_surprise.append(index_surprise)

        running_tot_surprise += index_surprise
        prev = sorted_indices[prev_i]
        generated += prev.tolist()
        context = torch.tensor([prev.tolist()])  # add ".to('cuda')" if you have a GPU

        # adjust max_surprise
        error_surprise = index_surprise - target_surprise
        max_surprise -= learning_rate*error_surprise

print("Total surprise value:", sum(indices_surprise))
print("Average surprise value:", sum(indices_surprise)/num_tokens)
print("Context text:",context_text)
print("Generated text:",tokenizer.decode(generated))
