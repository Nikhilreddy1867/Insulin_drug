#!/usr/bin/env python3
"""
Helper script to reconstruct the exact tokenizer used during training.
This should match the CharTokenizer logic from the training script.

The model was trained with vocab_size=65, so we need to reconstruct
the exact 65 tokens that were used.
"""

import torch

# Load checkpoint to inspect
checkpoint_path = "models/molt5_best.pt"
ckpt = torch.load(checkpoint_path, map_location='cpu')

print("="*60)
print("Tokenizer Reconstruction Helper")
print("="*60)

# Get vocab size from embedding
if isinstance(ckpt, dict):
    state_dict = ckpt.get("model_state", ckpt)
else:
    state_dict = ckpt

vocab_size = None
for key in state_dict.keys():
    if "encoder.tok_emb.weight" in key:
        vocab_size = state_dict[key].shape[0]
        print(f"Found vocab_size: {vocab_size}")
        break

if vocab_size is None:
    print("Could not determine vocab_size")
    exit(1)

print(f"\nVocab size: {vocab_size}")
print("\nTo fix SMILES generation, you need to:")
print("1. Reconstruct the exact tokenizer used during training")
print("2. The training script should have saved the tokenizer.tokens list")
print("3. Or you can reconstruct it from the training data")

print("\n" + "="*60)
print("Reconstruction steps:")
print("="*60)
print("1. Find the training data files (train.txt, validation.txt, test.txt)")
print("2. Extract all unique characters from both descriptions and SMILES")
print("3. Build the tokenizer exactly as in training:")
print("   - Special tokens: ['<pad>', '<s>', '</s>', '<unk>']")
print("   - Then sorted unique characters from corpus")
print("   - Total should be exactly 65 tokens")
print("\n4. Save the tokenizer.tokens list to the checkpoint or a separate file")
print("\n5. Update combined_server.py to load the exact tokenizer")

print("\n" + "="*60)
print("Quick fix - Update your training script to save tokenizer:")
print("="*60)
print("""
# In your training script, after creating tokenizer:
# Save tokenizer tokens to checkpoint
checkpoint = {
    'model_state': model.state_dict(),
    'args': {
        'd_model': D_MODEL,
        'd_ff': D_FF,
        'n_layers': NUM_LAYERS,
        'n_heads': NUM_HEADS,
        'max_len': MAX_LEN,
    },
    'tokenizer_tokens': tokenizer.tokens,  # Add this line
}
torch.save(checkpoint, 'molt5_best.pt')
""")

