# 🪙 toktokenizer
toktokenizer is a [BPE](https://en.wikipedia.org/wiki/Byte_pair_encoding) tokenizer implemented in rust and exposed in python using [pyo3](https://github.com/PyO3/pyo3) bindings.

```python 
import toktokenizer as tok
bpe = tok.BPETokenizer.from_pretrained("wikibpe.json")
assert bpe.decode(bpe.encode("rust is pretty fun 🦀"))
```

Install `toktokenizer` from PyPI or from source 
```
pip install toktokenizer
```

# Performance 

tok: 16.18MB/s
tokenizers: 4.89MB/s
tiktoken: 22.98MB/s


