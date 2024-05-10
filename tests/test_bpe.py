import pytest
import tok
import random
import os

ENCODER_PATH = "/Users/nathanielnethercott/ml/tokenizers/tok/wikibpe.json"
DATA_PATH = "/Users/nathanielnethercott/ml/tokenizers/wikidump_test.txt"


@pytest.fixture
def bpe():
    return tok.BPETokenizer.from_pretrained(ENCODER_PATH)


@pytest.fixture
def text():
    file = DATA_PATH

    with open(file, "r") as f:
        text = f.read()

    size = len(text)
    offset = random.randint(0, size // 2)
    return text[offset : offset + len(text) // 4]


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
