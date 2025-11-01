#!/usr/bin/env python3
"""
Diagnostic script to check if SMILES model is loading correctly
and whether it's using saved weights or random weights.
"""

import os
import torch
import torch.nn as nn
import math

# MiniT5 architecture from training script
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2048):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe)

    def forward(self, x):
        L = x.size(1)
        return x + self.pe[:L]

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.nh = n_heads
        self.dk = d_model // n_heads
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, q_in, k_in, v_in, mask=None):
        B, Lq, D = q_in.shape
        Lk = k_in.shape[1]
        q = self.q(q_in).view(B, Lq, self.nh, self.dk).transpose(1, 2)
        k = self.k(k_in).view(B, Lk, self.nh, self.dk).transpose(1, 2)
        v = self.v(v_in).view(B, Lk, self.nh, self.dk).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.dk)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e4)
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, Lq, D)
        return self.out(out)

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.fc2(self.dropout(torch.relu(self.fc1(x))))

class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, src_mask=None):
        _x = x
        x = self.norm1(x)
        x = _x + self.dropout(self.self_attn(x, x, x, mask=src_mask))
        _x2 = x
        x = self.norm2(x)
        x = _x2 + self.dropout(self.ff(x))
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads)
        self.cross_attn = MultiHeadAttention(d_model, n_heads)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, tgt_mask=None, memory_mask=None):
        _x = x
        x = self.norm1(x)
        x = _x + self.dropout(self.self_attn(x, x, x, mask=tgt_mask))
        _x2 = x
        x = self.norm2(x)
        x = _x2 + self.dropout(self.cross_attn(x, enc_output, enc_output, mask=memory_mask))
        _x3 = x
        x = self.norm3(x)
        x = _x3 + self.dropout(self.ff(x))
        return x

class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, n_heads, d_ff, max_len):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos = PositionalEncoding(d_model, max_len=max_len)
        self.layers = nn.ModuleList([EncoderLayer(d_model, n_heads, d_ff) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, src, src_mask=None):
        x = self.tok_emb(src)
        x = self.pos(x)
        for l in self.layers:
            x = l(x, src_mask)
        return self.norm(x)

class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, n_heads, d_ff, max_len):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos = PositionalEncoding(d_model, max_len=max_len)
        self.layers = nn.ModuleList([DecoderLayer(d_model, n_heads, d_ff) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.out = nn.Linear(d_model, vocab_size)

    def forward(self, tgt, enc_out, tgt_mask=None, memory_mask=None):
        x = self.tok_emb(tgt)
        x = self.pos(x)
        for l in self.layers:
            x = l(x, enc_out, tgt_mask=tgt_mask, memory_mask=memory_mask)
        x = self.norm(x)
        return self.out(x)

class MiniT5(nn.Module):
    def __init__(self, vocab_size, d_model=512, d_ff=1024, n_layers=6, n_heads=8, max_len=512):
        super().__init__()
        self.encoder = Encoder(vocab_size, d_model, n_layers, n_heads, d_ff, max_len)
        self.decoder = Decoder(vocab_size, d_model, n_layers, n_heads, d_ff, max_len)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None):
        enc = self.encoder(src, src_mask)
        out = self.decoder(tgt, enc, tgt_mask=tgt_mask, memory_mask=memory_mask)
        return out

def check_model_weights(model_path, device='cpu'):
    """Check if model is loading saved weights or using random weights"""
    print(f"\n{'='*60}")
    print(f"Checking model: {model_path}")
    print(f"{'='*60}")
    
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return False
    
    try:
        # Load checkpoint
        ckpt = torch.load(model_path, map_location=device)
        print(f"✅ Checkpoint loaded successfully")
        
        # Get state dict
        if isinstance(ckpt, dict):
            if 'model_state' in ckpt:
                state_dict = ckpt['model_state']
                print(f"✅ Found 'model_state' in checkpoint")
            else:
                state_dict = ckpt
                print(f"⚠️  Checkpoint is dict but no 'model_state' key")
        else:
            state_dict = ckpt
            print(f"✅ Checkpoint is state_dict")
        
        # Check architecture by examining keys
        first_keys = list(state_dict.keys())[:5]
        print(f"\nFirst 5 keys in checkpoint:")
        for key in first_keys:
            print(f"  - {key}")
        
        # Check if it's MiniT5 (encoder/decoder) or ProteinLM (single transformer)
        is_encoder_decoder = any('encoder' in k.lower() or 'decoder' in k.lower() for k in state_dict.keys())
        print(f"\nArchitecture detected: {'Encoder-Decoder (MiniT5)' if is_encoder_decoder else 'Single Transformer (ProteinLM)'}")
        
        # Get some weight values to check if they're random or trained
        print(f"\nSample weight statistics:")
        sample_weights = []
        for key, tensor in list(state_dict.items())[:3]:
            if tensor.numel() > 0:
                sample_weights.append(tensor.flatten()[:10].tolist())
                print(f"  {key}:")
                print(f"    Shape: {tensor.shape}")
                print(f"    Mean: {tensor.mean().item():.6f}")
                print(f"    Std: {tensor.std().item():.6f}")
                print(f"    Min: {tensor.min().item():.6f}")
                print(f"    Max: {tensor.max().item():.6f}")
                print(f"    First 5 values: {tensor.flatten()[:5].tolist()}")
        
        # Try to instantiate model and load weights
        print(f"\n{'='*60}")
        print("Attempting to load model...")
        print(f"{'='*60}")
        
        # Try to infer vocab size from embedding layer
        emb_key = None
        for key in state_dict.keys():
            if 'tok_emb' in key or 'embedding' in key.lower():
                emb_key = key
                break
        
        if emb_key:
            vocab_size = state_dict[emb_key].shape[0]
            print(f"✅ Detected vocab_size: {vocab_size}")
        else:
            vocab_size = 100  # Default guess
            print(f"⚠️  Could not detect vocab_size, using default: {vocab_size}")
        
        # Try loading as MiniT5
        if is_encoder_decoder:
            print("\n🔄 Trying to load as MiniT5 (encoder-decoder)...")
            model = MiniT5(vocab_size=vocab_size)
            model.eval()
            
            # Get initial weights (random)
            initial_weights = {}
            for name, param in model.named_parameters():
                initial_weights[name] = param.data.clone()
            
            # Try loading state dict
            result = model.load_state_dict(state_dict, strict=False)
            
            if result.missing_keys:
                print(f"⚠️  Missing keys: {len(result.missing_keys)}")
                print(f"   First 5: {result.missing_keys[:5]}")
            if result.unexpected_keys:
                print(f"⚠️  Unexpected keys: {len(result.unexpected_keys)}")
                print(f"   First 5: {result.unexpected_keys[:5]}")
            
            # Check if weights changed
            weights_changed = False
            for name, param in model.named_parameters():
                if name in initial_weights:
                    if not torch.equal(param.data, initial_weights[name]):
                        weights_changed = True
                        break
            
            if weights_changed:
                print(f"\n✅ ✅ ✅ SUCCESS: Model loaded saved weights (weights changed from random)")
            else:
                print(f"\n❌ ❌ ❌ ERROR: Model is using RANDOM weights (weights unchanged)")
            
            return weights_changed
        else:
            print("\n⚠️  Architecture appears to be ProteinLM, not MiniT5")
            print("   Current Flask code uses ProteinLM architecture")
            return True
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Check both possible model paths
    models_to_check = [
        "models/Protein_to_Smile.pt",
        "models/molt5_best.pt",
    ]
    
    for model_path in models_to_check:
        if os.path.exists(model_path):
            check_model_weights(model_path, device)
            print("\n")
        else:
            print(f"⚠️  Model not found: {model_path}")

