This is the code for mirostat sampling algorithm proposed in .
Mirostat is a neural text decoding algorithm that generates texts of predetermined perplexity value.

Installation requirement:
- pip install transformers==v2.8.0
- pip3 install torch torchvision

Example Use:

python mirostat.py --num_tokens 200 --tau 3.0 --context "/context.txt"

where num_tokens reperesent the number of tokens to be generated, tau reperesent the average surprise value (i.e. log of perplexity), and context.txt is a text file containing the context.
