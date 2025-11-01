#!/usr/bin/env python3
"""
Check molt5_best.pt checkpoint structure to get vocab size and tokenizer info
"""

import torch
import os

checkpoint_path = "models/molt5_best.pt"

if not os.path.exists(checkpoint_path):
    print(f"❌ Checkpoint not found: {checkpoint_path}")
    exit(1)

print(f"Loading checkpoint: {checkpoint_path}")
ckpt = torch.load(checkpoint_path, map_location='cpu')

print("\n" + "="*60)
print("Checkpoint Keys:")
print("="*60)
if isinstance(ckpt, dict):
    for key in ckpt.keys():
        print(f"  - {key}")
        if key == "model_state":
            state_dict = ckpt[key]
            if isinstance(state_dict, dict):
                print(f"    State dict has {len(state_dict)} keys")
                # Find embedding layer to get vocab size
                for k in state_dict.keys():
                    if "tok_emb.weight" in k or "encoder.tok_emb.weight" in k:
                        vocab_size = state_dict[k].shape[0]
                        print(f"\n✅ Found vocab_size from {k}: {vocab_size}")
                        print(f"   Embedding shape: {state_dict[k].shape}")
        elif key == "args":
            args = ckpt[key]
            print(f"    Args keys: {list(args.keys()) if isinstance(args, dict) else 'N/A'}")
        elif "tokenizer" in key.lower():
            print(f"    Tokenizer info: {type(ckpt[key])}")
            if hasattr(ckpt[key], 'tokens'):
                print(f"    Tokenizer vocab size: {len(ckpt[key].tokens)}")
                print(f"    First 20 tokens: {ckpt[key].tokens[:20]}")
else:
    # Direct state dict
    print("Checkpoint is direct state_dict")
    state_dict = ckpt
    # Find embedding layer
    for k in state_dict.keys():
        if "tok_emb.weight" in k or "encoder.tok_emb.weight" in k:
            vocab_size = state_dict[k].shape[0]
            print(f"\n✅ Found vocab_size from {k}: {vocab_size}")
            print(f"   Embedding shape: {state_dict[k].shape}")

print("\n" + "="*60)
print("Sample State Dict Keys:")
print("="*60)
if isinstance(ckpt, dict) and "model_state" in ckpt:
    sample_keys = list(ckpt["model_state"].keys())[:10]
elif isinstance(ckpt, dict):
    sample_keys = list(ckpt.keys())[:10]
else:
    sample_keys = list(ckpt.keys())[:10]

for key in sample_keys:
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        shape = ckpt["model_state"][key].shape if hasattr(ckpt["model_state"][key], 'shape') else 'N/A'
    else:
        shape = ckpt[key].shape if hasattr(ckpt[key], 'shape') else 'N/A'
    print(f"  {key}: {shape}")

