"""
Generate text from a trained nano-KAN checkpoint.
"""

import argparse

import torch
import tiktoken

from model import GPT, GPTConfig


def main():
    p = argparse.ArgumentParser(description="Generate from nano-KAN")
    p.add_argument("--checkpoint", type=str, default="out/ckpt.pt", help="Path to checkpoint")
    p.add_argument("--prompt", type=str, default="To be, or not to be,", help="Starting prompt")
    p.add_argument("--max_tokens", type=int, default=500, help="Number of tokens to generate")
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_k", type=int, default=200)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    config: GPTConfig = ckpt["config"]
    model = GPT(config)
    model.load_state_dict(ckpt["model"])
    model.to(args.device)
    model.eval()

    # Encode prompt
    enc = tiktoken.get_encoding("gpt2")
    tokens = enc.encode(args.prompt, allowed_special=set())
    x = torch.tensor(tokens, dtype=torch.long, device=args.device).unsqueeze(0)

    # Generate
    with torch.no_grad():
        y = model.generate(x, max_new_tokens=args.max_tokens, temperature=args.temperature, top_k=args.top_k)

    print(enc.decode(y[0].tolist()))


if __name__ == "__main__":
    main()
