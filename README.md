This is the code for mirostat sampling algorithm proposed in our paper-"Mirostat: A Perplexity-Controlled Neural Text Decoding Algorithm". The paper is available here: https://arxiv.org/abs/2007.14966
Mirostat is a neural text decoding algorithm that generates texts of predetermined perplexity value.

Installation requirement:
- pip install transformers==v2.8.0
- pip install torch torchvision

Example Use:

python mirostat.py --num_tokens 200 --tau 3.0 --context "/context.txt"

where num_tokens reperesent the number of tokens to be generated, tau reperesent the average surprise value (i.e. log of perplexity), and context.txt is a text file containing the context.
