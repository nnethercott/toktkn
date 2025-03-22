# 🪙 toktkn

toktkn is a [BPE](https://en.wikipedia.org/wiki/Byte_pair_encoding) tokenizer implemented in rust and exposed in python using [pyo3](https://github.com/PyO3/pyo3) bindings.

```python
from toktkn import BPETokenizer, TokenizerConfig

# create new tokenizer
config = TokenizerConfig(vocab_size: 10)
bpe = BPETokenizer(config)

# build encoding rules on some corpus
bpe.train("some really interesting training data here...")
text = "rust is pretty fun 🦀"

assert bpe.decode(bpe.encode(text)) == text

# serialize to disk
bpe.save_pretrained("tokenizer.json")
del(bpe)
bpe = BPETokenizer.from_pretrained("tokenizer.json")
assert(len(bpe)==10)
```

# Install 

Install `toktkn` from PyPI with the following

```
pip install toktkn
```

**Note:** if you want to build from source make sure cargo is installed!


# Performance

slightly faster than openai & a lot quicker than 🤗!

![alt text](performance.png)

Performance measured on 2.5MB from the [wikitext](https://huggingface.co/datasets/wikitext) test split using openai's [tiktoken gpt2 tokenizer](https://github.com/openai/tiktoken) with `tiktoken==0.6.0` and the [implementation from 🤗 tokenizers](https://huggingface.co/openai-community/gpt2) at `tokenizers==0.19.1`
