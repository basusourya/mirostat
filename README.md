Code for mirostat sampling algorithm proposed in our paper "Mirostat: A Perplexity-Controlled Neural Text Decoding Algorithm" in ICLR 2021. The paper is available here: https://arxiv.org/abs/2007.14966.

**Tl;dr**: We provide a new text decoding algorithm that directly controls generated text statistics and hence generates more human-like texts using large language models like GPT-2, CTRL, etc.

Installation requirement:
(Tested with version 4.16.2)

`pip install transformers`

Example Use:

`python mirostat.py --num_tokens 200 --tau 3.0 --context "/context.txt"`

where num_tokens reperesent the number of tokens to be generated, tau reperesent the average surprise value (i.e. log of perplexity), and context.txt is a text file containing the context.

If you find the code useful in your work, please cite it as:
```
@inproceedings{
BasuRKV2021,
title={MIROSTAT: A NEURAL TEXT DECODING ALGORITHM THAT DIRECTLY CONTROLS PERPLEXITY},
author={Sourya Basu and Govardana Sachitanandam Ramachandran and Nitish Shirish Keskar and Lav R. Varshney},
booktitle={International Conference on Learning Representations},
year={2021},
url={https://openreview.net/forum?id=W1G1JZEIy5_}
}
```
