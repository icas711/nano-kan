"""
Download and tokenize the Shakespeare dataset using tiktoken GPT-2 BPE.
Produces train.bin and val.bin in the same directory.
"""

import os
import urllib.request

import numpy as np
import tiktoken

DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    input_path = os.path.join(SCRIPT_DIR, "input.txt")

    # Download if needed
    if not os.path.exists(input_path):
        print(f"Downloading Shakespeare from {DATA_URL} ...")
        urllib.request.urlretrieve(DATA_URL, input_path)
        print(f"Saved to {input_path}")

    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()
    print(f"Dataset size: {len(text):,} characters")

    # Tokenize
    enc = tiktoken.get_encoding("gpt2")
    tokens = enc.encode_ordinary(text)
    tokens = np.array(tokens, dtype=np.uint16)
    print(f"Tokenized: {len(tokens):,} tokens")

    # Split: 90% train, 10% val
    n = len(tokens)
    split = int(n * 0.9)
    train_tokens = tokens[:split]
    val_tokens = tokens[split:]

    train_path = os.path.join(SCRIPT_DIR, "train.bin")
    val_path = os.path.join(SCRIPT_DIR, "val.bin")
    train_tokens.tofile(train_path)
    val_tokens.tofile(val_path)

    print(f"Train: {len(train_tokens):,} tokens → {train_path}")
    print(f"Val:   {len(val_tokens):,} tokens → {val_path}")


if __name__ == "__main__":
    main()
