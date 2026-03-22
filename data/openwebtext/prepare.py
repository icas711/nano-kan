"""
Download and tokenize the OpenWebText dataset using tiktoken GPT-2 BPE.
Produces train.bin and val.bin (uint16 numpy arrays of token IDs).

Requires: pip install datasets tiktoken
This downloads ~55GB and processes ~9B tokens — may take 1-2 hours.
"""

import os
from pathlib import Path

import numpy as np
import tiktoken
from datasets import load_dataset

SCRIPT_DIR = Path(__file__).parent.resolve()
NUM_PROC = 8  # parallel tokenization workers


def main():
    enc = tiktoken.get_encoding("gpt2")

    print("Loading OpenWebText from HuggingFace...")
    dataset = load_dataset("openwebtext", trust_remote_code=True)

    # OpenWebText has only 'train' split — we split 0.5% for val
    split_dataset = dataset["train"].train_test_split(
        test_size=0.0005, seed=2357, shuffle=True
    )
    split_dataset["val"] = split_dataset.pop("test")
    print(f"Train: {len(split_dataset['train']):,} docs | Val: {len(split_dataset['val']):,} docs")

    def tokenize(example):
        ids = enc.encode_ordinary(example["text"])
        ids.append(enc.eot_token)  # end-of-text delimiter
        return {"ids": ids, "len": len(ids)}

    print("Tokenizing (this may take a while)...")
    tokenized = split_dataset.map(
        tokenize,
        remove_columns=["text"],
        num_proc=NUM_PROC,
        desc="Tokenizing",
    )

    # Write to binary files
    for split, dset in tokenized.items():
        total_len = sum(dset["len"])
        print(f"{split}: {total_len:,} tokens")

        out_path = SCRIPT_DIR / f"{split}.bin"
        arr = np.memmap(str(out_path), dtype=np.uint16, mode="w+", shape=(total_len,))

        idx = 0
        for example in dset:
            ids = np.array(example["ids"], dtype=np.uint16)
            arr[idx : idx + len(ids)] = ids
            idx += len(ids)
        arr.flush()
        print(f"  → {out_path} ({out_path.stat().st_size / 1e9:.2f} GB)")

    print("Done!")


if __name__ == "__main__":
    main()
