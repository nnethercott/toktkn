# 🪙 toktokenizer
toktokenizer is a [BPE](https://en.wikipedia.org/wiki/Byte_pair_encoding) tokenizer implemented in rust and exposed in python using [pyo3](https://github.com/PyO3/pyo3) bindings. 

```python 
import toktokenizer as tok
bpe = tok.BPETokenizer.from_pretrained("wikibpe.json")

assert bpe.decode(bpe.encode("rust is pretty fun 🦀"))
```

Install `toktokenizer` from PyPI with the following
```
pip install toktokenizer
```
**Note:** if you want to build from source make sure rust is installed! 

The only class `toktokenizer` exposes is `BPETokenizer`. The class itself is pretty minimalistic, with all major methods being showed below:

```python
from toktokenizer import BPETokenizer

bpe = tok.BPETokenizer()

# train a byte-pair tokenizer on some corpus 
train_corpus = "this is some training data. any dumped string will do!"
vocab_size = 8
bpe.train(train_corpus, vocab_size)

# save tokenizer state
bpe.save_encoder("8word.json")

# load tokenizer from dumped file 
bpe.load_encoder("8word.json")

# encode and decode
input_ids = bpe.encode("some data")
decoded = bpe.decode(input_ids)
```

# Performance 
not the greatest but also not the worst!

![alt text](performance.png)

Performance measured on 2.5MB from the [wikitext](https://huggingface.co/datasets/wikitext) test split using openai's [tiktoken gpt2 tokenizer](https://github.com/openai/tiktoken) with `tiktoken==0.6.0` and the [implementation from 🤗 tokenizers](https://huggingface.co/openai-community/gpt2) at `tokenizers==0.19.1`
