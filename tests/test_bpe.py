import os
import random

import numpy as np
import pytest
import toktokenizer as tok

ENCODER_PATH = "./tiny_shakespeare_5k.json"
DATA_PATH = "/Users/nathanielnethercott/ml/tokenizers/wikidump_test.txt"


@pytest.fixture
def bpe():
    return tok.BPETokenizer.from_pretrained(ENCODER_PATH)


@pytest.fixture
def text():
    file = DATA_PATH

    with open(file, "r") as f:
        text = f.read()

    return text[:256] #ad hoc 


class TestBPETokenizer:
    def test_encode_decode(self, bpe, text):
        assert bpe.decode(bpe.encode(text)) == text

    def test_compression(self, bpe, text):
        encoded = bpe.encode(text)
        assert len(encoded) < len(text)

    def test_read_write(self, tmp_path, bpe):
        dir = tmp_path / "encoder"
        dir.mkdir()
        path = str(dir)

        bpe.save_encoder(f"{path}/encoder.json")
        assert os.path.exists(f"{path}/encoder.json")
    
    def test_random_encode_decode(self, bpe):
        random_tokens = np.random.randint(low=0, high = len(bpe)-1, size = 128).tolist()

        # make sure doesn't fail (like 128-256 token id gap in earlier versions)
        bpe.decode(random_tokens)
        assert True
