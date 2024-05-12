import toktokenizer as tok
import requests
import time

tiny_shakespeare = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
train_data = requests.get(tiny_shakespeare).content.decode("utf-8")

train_data = train_data[: len(train_data) // 4]
eval_data = train_data[len(train_data) // 4 : len(train_data) // 2]

# train bpe tokenizer on some shakespeare
bpe = tok.BPETokenizer()

# should take ~1 min
bpe.train(
    train_data,
    vocab_size=500,
)

# saving and loading trained encoders
bpe.save_encoder("tiny_shakespeare_pairs.json")
bpe = tok.BPETokenizer()
bpe.load_encoder("tiny_shakespeare_pairs.json")

# encode and decode
print("here's a little sentence before encoding")
print(
    f'same sentence after encoding: {bpe.encode("here's a little sentence before encoding")}'
)

# measure token throughput
now = time.time()
input_ids = bpe.encode(eval_data)
elapsed = time.time() - now

print(f"throughput: {len(eval_data)/(elapsed*1e6):.2f}MB/s")
print(f"compression ratio: {len(input_ids)/len(eval_data)}")
