"""
FastAPI service for Seq2Drug Fusion model inference
"""
import os
import json
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

# Model paths (relative to this file's directory)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(os.path.dirname(BASE_DIR), "backend", "models")

# Helper to find file with case variations
def find_model_file(base_name, dir_path=MODEL_DIR):
    variations = [base_name, base_name.lower(), base_name.upper(), base_name.capitalize()]
    variations += [f.replace('t5', 'T5') if 't5' in f.lower() else f for f in variations]
    for var in variations:
        path = os.path.join(dir_path, var)
        if os.path.isfile(path):
            return path
    return os.path.join(dir_path, base_name)  # fallback

PROGEN_CKP = find_model_file("progen.pt")
MOLT5_CKP = find_model_file("molt5.pt")
FUSION_CKP = find_model_file("fusion_best.pt")
TOKENIZER_JSON = os.path.join(MODEL_DIR, "tokenizer.json")

# Device selection: CUDA > MPS (Metal/M1) > CPU
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    DEVICE_TYPE = "GPU (CUDA)"
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    DEVICE_TYPE = "GPU (Metal/M1)"
    # Optimize for M1 GPU
    try:
        if hasattr(torch.mps, 'empty_cache'):
            torch.mps.empty_cache()
        # Enable optimizations
        torch.backends.mps.allow_tf32 = True
        print(f"[FastAPI] M1 GPU optimizations enabled")
    except Exception as e:
        print(f"[FastAPI] MPS optimization warning: {e}")
else:
    DEVICE = torch.device("cpu")
    DEVICE_TYPE = "CPU"

print(f"[FastAPI] Using device: {DEVICE} ({DEVICE_TYPE})")

MAX_SEQ_LEN = 128
MAX_SMILES_LEN = 256

app = FastAPI(title="Seq2Drug Fusion API", version="1.0.0")

# Optional RDKit
try:
    from rdkit import Chem
    rdkit_ok = True
except Exception:
    rdkit_ok = False

# ============ Tokenizer ============

class CharTokenizer:
    def __init__(self, tokens, special=("<pad>", "<s>", "</s>", "<unk>")):
        self.PAD, self.BOS, self.EOS, self.UNK = special
        self.tokens = [self.PAD, self.BOS, self.EOS, self.UNK] + tokens
        self.stoi = {s: i for i, s in enumerate(self.tokens)}
        self.itos = {i: s for s, i in self.stoi.items()}

    def encode(self, text, add_special=True, max_len=None):
        ids = [self.stoi.get(ch, self.stoi[self.UNK]) for ch in text]
        if add_special:
            ids = [self.stoi[self.BOS]] + ids + [self.stoi[self.EOS]]
        if max_len:
            ids = ids[:max_len]
        return ids

    def decode(self, ids):
        return "".join([
            self.itos.get(i, self.UNK)
            for i in ids
            if i not in {self.stoi[self.PAD], self.stoi[self.BOS], self.stoi[self.EOS]}
        ])

    def __len__(self):
        return len(self.tokens)

def load_tokenizer(tokenizer_json_path: str) -> CharTokenizer:
    with open(tokenizer_json_path, "r") as f:
        data = json.load(f)
    special = tuple(data.get("special", ["<pad>", "<s>", "</s>", "<unk>"]))
    tokens = data["tokens"]
    return CharTokenizer(tokens=tokens, special=special)

# ============ Models ============

class TransformerBlock(nn.Module):
    def __init__(self, d_model, nhead, d_ff=1024, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.ln1, self.ln2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff), nn.ReLU(), nn.Linear(d_ff, d_model))

    def forward(self, x, mask=None):
        res, _ = self.attn(x, x, x, key_padding_mask=(mask == 0) if mask is not None else None)
        x = self.ln1(x + res)
        return self.ln2(x + self.ff(x))

class ProteinLM(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=8, num_layers=6, d_ff=1024, max_len=512):
        super().__init__()
        self.token_emb, self.pos_emb = nn.Embedding(vocab_size, d_model), nn.Embedding(max_len, d_model)
        self.layers = nn.ModuleList([TransformerBlock(d_model, nhead, d_ff) for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(d_model)

    def embed(self, input_ids):
        B, T = input_ids.size()
        x = self.token_emb(input_ids) + self.pos_emb(torch.arange(0, T, device=input_ids.device).unsqueeze(0))
        for layer in self.layers:
            x = layer(x)
        return self.ln_f(x)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2048):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:x.size(1)]

class MultiHeadAttentionCustom(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.nh, self.dk = n_heads, d_model // n_heads
        self.q, self.k, self.v = nn.Linear(d_model, d_model), nn.Linear(d_model, d_model), nn.Linear(d_model, d_model)
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
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, Lq, D)
        return self.out(out)

class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.self_attn = MultiHeadAttentionCustom(d_model, n_heads)
        self.cross_attn = MultiHeadAttentionCustom(d_model, n_heads)
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff), nn.ReLU(), nn.Linear(d_ff, d_model))
        self.norm1, self.norm2, self.norm3 = nn.LayerNorm(d_model), nn.LayerNorm(d_model), nn.LayerNorm(d_model)

    def forward(self, x, enc_output, tgt_mask=None):
        x = x + self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x), tgt_mask)
        x = x + self.cross_attn(self.norm2(x), enc_output, enc_output)
        return x + self.ff(self.norm3(x))

class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, n_heads, d_ff, max_len):
        super().__init__()
        self.tok_emb, self.pos = nn.Embedding(vocab_size, d_model, padding_idx=0), PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([DecoderLayer(d_model, n_heads, d_ff) for _ in range(n_layers)])
        self.norm, self.out = nn.LayerNorm(d_model), nn.Linear(d_model, vocab_size)

    def forward(self, tgt, enc_out, tgt_mask=None):
        x = self.pos(self.tok_emb(tgt))
        for l in self.layers:
            x = l(x, enc_out, tgt_mask)
        return self.out(self.norm(x))

class MiniT5(nn.Module):
    def __init__(self, vocab_size, d_model=512, d_ff=1024, n_layers=6, n_heads=8, max_len=512):
        super().__init__()
        self.decoder = Decoder(vocab_size, d_model, n_layers, n_heads, d_ff, max_len)

    def decoder_forward(self, tgt, enc_out, tgt_mask=None):
        return self.decoder(tgt, enc_out, tgt_mask)

    @torch.no_grad()
    def generate_from_enc(self, enc_out, tokenizer, max_len=128, temperature=1.0, top_k=0, top_p=0.0, repetition_penalty=1.0):
        """
        Generate with sampling for novel outputs.
        - temperature: > 1.0 = more creative/random, < 1.0 = more conservative, 1.0 = balanced
        - top_k: sample from top K tokens (0 = disabled)
        - top_p: nucleus sampling threshold (0.0 = disabled)
        - repetition_penalty: > 1.0 penalizes repeated tokens (encourages diversity)
        """
        B = enc_out.size(0)
        ys = torch.tensor([[tokenizer.stoi[tokenizer.BOS]]] * B, device=enc_out.device)
        
        for _ in range(max_len - 1):
            tgt_mask = torch.tril(torch.ones((ys.size(1), ys.size(1)), device=enc_out.device)).unsqueeze(0).unsqueeze(0)
            logits = self.decoder(ys, enc_out, tgt_mask)
            logits = logits[:, -1, :] / max(temperature, 1e-8)
            
            # Apply repetition penalty to discourage exact repeats (helps avoid memorization)
            if repetition_penalty > 1.0 and ys.size(1) > 1:
                # Count occurrences of each token in the generated sequence so far (vectorized)
                # Create a matrix where each column represents a token ID
                token_ids = torch.arange(logits.size(-1), device=logits.device).unsqueeze(0).unsqueeze(0)
                # Count occurrences: (batch_size, seq_len, vocab_size) -> (batch_size, vocab_size)
                token_counts = (ys.unsqueeze(-1) == token_ids).sum(dim=1).float()
                # Apply penalty: divide logits by (1 + penalty * count) for each token
                penalty_factor = 1.0 + (repetition_penalty - 1.0) * token_counts
                logits = logits / penalty_factor
            
            # Apply top-k filtering (ensure k doesn't exceed vocab size)
            if top_k > 0:
                vocab_size = logits.size(-1)
                k = min(top_k, vocab_size)  # Clamp to vocab size
                top_k_values, top_k_indices = torch.topk(logits, k, dim=-1)
                # Create a mask to zero out tokens outside top-k
                logits_filtered = torch.full_like(logits, float('-inf'))
                logits_filtered.scatter_(-1, top_k_indices, top_k_values)
                logits = logits_filtered
            
            # Apply top-p (nucleus) sampling
            if top_p > 0.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                # Create mask for tokens outside nucleus
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                # Create indices to remove
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float('-inf')
            
            # Handle edge case: if all logits are -inf, reset to uniform distribution
            if torch.isinf(logits).all():
                logits = torch.zeros_like(logits)
            
            # Sample from probability distribution
            probs = F.softmax(logits, dim=-1)
            # Ensure probabilities are valid (not all NaN/inf)
            if torch.isnan(probs).any() or torch.isinf(probs).any():
                probs = torch.softmax(torch.randn_like(logits) * 0.1, dim=-1)  # Fallback to random
            
            next_token = torch.multinomial(probs, num_samples=1)
            
            ys = torch.cat([ys, next_token], dim=1)
            if (next_token == tokenizer.stoi[tokenizer.EOS]).all():
                break
        
        return ys

class FusionModel(nn.Module):
    def __init__(self, progen_model, molt5_model, proj_in=256, proj_out=512):
        super().__init__()
        self.progen, self.molt5 = progen_model, molt5_model
        self.proj = nn.Sequential(
            nn.Linear(proj_in, proj_out), nn.LayerNorm(proj_out), nn.ReLU(), nn.Dropout(0.1)
        )

# ============ Global Models ============

_tokenizer = None
_models = None

def load_models():
    global _tokenizer, _models
    if _tokenizer is None:
        if not os.path.isfile(TOKENIZER_JSON):
            raise FileNotFoundError(f"Tokenizer not found: {TOKENIZER_JSON}")
        _tokenizer = load_tokenizer(TOKENIZER_JSON)
    
    if _models is None:
        PROGEN_DMODEL, MOLT5_DMODEL = 256, 512
        progen = ProteinLM(vocab_size=len(_tokenizer), d_model=PROGEN_DMODEL).to(DEVICE)
        molt5 = MiniT5(vocab_size=len(_tokenizer), d_model=MOLT5_DMODEL).to(DEVICE)
        fusion = FusionModel(progen, molt5, proj_in=PROGEN_DMODEL, proj_out=MOLT5_DMODEL).to(DEVICE)
        
        # Load checkpoints (non-strict)
        try:
            progen.load_state_dict(torch.load(PROGEN_CKP, map_location=DEVICE), strict=False)
        except Exception as e:
            print(f"Warning: ProGen checkpoint load issue: {e}")
        try:
            molt5.load_state_dict(torch.load(MOLT5_CKP, map_location=DEVICE), strict=False)
        except Exception as e:
            print(f"Warning: MolT5 checkpoint load issue: {e}")
        try:
            fusion.load_state_dict(torch.load(FUSION_CKP, map_location=DEVICE), strict=False)
        except Exception as e:
            print(f"Warning: Fusion checkpoint load issue: {e}")
        
        for p in progen.parameters():
            p.requires_grad = False
        
        _models = (progen, molt5, fusion)
    
    return _tokenizer, _models

# Load models on startup
print("🔄 Loading models...")
try:
    load_models()
    print("✅ Models loaded successfully")
except Exception as e:
    print(f"❌ Error loading models: {e}")
    raise

# ============ API Endpoints ============

class PredictRequest(BaseModel):
    sequence: str
    max_length: Optional[int] = 256
    temperature: Optional[float] = 1.5  # Default 1.5 for balanced creativity/validity (reduces memorization)
    top_k: Optional[int] = 50  # Default 50 for diversity (0 = disabled)
    top_p: Optional[float] = 0.9  # Default 0.9 for nucleus sampling (0.0 = disabled)
    repetition_penalty: Optional[float] = 1.15  # Default 1.15 penalizes repeats (encourages novelty)

@app.get("/")
def root():
    return {
        "message": "Seq2Drug Fusion API",
        "status": "running",
        "device": str(DEVICE),
        "device_type": DEVICE_TYPE
    }

@app.get("/health")
def health():
    return {
        "status": "ok",
        "models_loaded": _models is not None,
        "device": str(DEVICE),
        "device_type": DEVICE_TYPE
    }

def validate_smiles(smiles: str) -> tuple[bool, bool]:
    """
    Validate SMILES string using RDKit.
    Returns: (is_valid, rdkit_available)
    """
    if not rdkit_ok:
        return None, False
    
    try:
        # Check if SMILES contains invalid characters
        invalid_chars = ['<unk>', '[', ']', '*', 'Ml', 'RW', 'VW', 'NCW', 'ccc(Ml)']
        has_invalid = any(char in smiles for char in invalid_chars)
        
        if has_invalid:
            return False, True
        
        # Try to parse with RDKit
        mol = Chem.MolFromSmiles(smiles)
        is_valid = mol is not None
        
        # Additional validation: check if molecule can be sanitized
        if is_valid:
            try:
                Chem.SanitizeMol(mol)
                return True, True
            except Exception:
                return False, True
        
        return False, True
    except Exception:
        return False, True

@app.post("/predict")
def predict(request: PredictRequest):
    try:
        if not request.sequence or len(request.sequence.strip()) == 0:
            raise HTTPException(status_code=400, detail="Empty sequence provided")
        
        tokenizer, models = load_models()
        progen, molt5, fusion = models
        
        sequence = request.sequence.strip()
        max_len = min(request.max_length or 256, MAX_SMILES_LEN)
        
        # Encode sequence once (reused for retries)
        seq_ids = torch.tensor([tokenizer.encode(sequence, True, MAX_SEQ_LEN)], device=DEVICE)
        
        # Base sampling parameters
        base_temperature = max(0.5, min(request.temperature or 1.5, 2.5))
        base_top_k = max(0, request.top_k if request.top_k is not None else 50)
        base_top_p = max(0.0, min(request.top_p if request.top_p is not None else 0.9, 1.0))
        base_rep_penalty = max(1.0, min(request.repetition_penalty if request.repetition_penalty is not None else 1.15, 2.0))
        
        # Generate with retry logic for invalid SMILES
        max_retries = 5
        smiles = None
        isValid = None
        retries = 0
        final_temp = base_temperature
        final_params = {}
        
        with torch.no_grad():
            seq_h = progen.embed(seq_ids)
            pooled = seq_h.mean(dim=1)
            enc_emb = fusion.proj(pooled).unsqueeze(1)
            
            for attempt in range(max_retries):
                # Adjust parameters for retries (slightly lower temperature for stability)
                if attempt == 0:
                    temperature = base_temperature
                    top_k = base_top_k
                    top_p = base_top_p
                    repetition_penalty = base_rep_penalty
                else:
                    # Gradually reduce temperature and increase top_k for more conservative generation
                    temperature = max(0.8, base_temperature - 0.2 * attempt)
                    top_k = min(base_top_k + 10 * attempt, 100)
                    top_p = min(base_top_p + 0.05 * attempt, 0.95)
                    repetition_penalty = max(1.05, base_rep_penalty - 0.05 * attempt)
                
                # Generate SMILES
                out_ids = molt5.generate_from_enc(
                    enc_emb, tokenizer, max_len=max_len,
                    temperature=temperature, top_k=top_k, top_p=top_p,
                    repetition_penalty=repetition_penalty
                )
                
                smiles = tokenizer.decode(out_ids[0].cpu().tolist())
                
                # Validate with RDKit
                is_valid, rdkit_available = validate_smiles(smiles)
                
                if is_valid:
                    # Found valid SMILES!
                    isValid = True
                    final_temp = temperature
                    final_params = {
                        "temperature": temperature,
                        "top_k": top_k,
                        "top_p": top_p,
                        "repetition_penalty": repetition_penalty,
                        "attempts": attempt + 1
                    }
                    break
                elif rdkit_ok:
                    # Invalid SMILES, will retry
                    isValid = False
                    retries = attempt + 1
                    if attempt < max_retries - 1:
                        continue  # Retry with adjusted parameters
                    else:
                        # Last attempt failed, use original parameters for response
                        final_temp = base_temperature
                        final_params = {
                            "temperature": base_temperature,
                            "top_k": base_top_k,
                            "top_p": base_top_p,
                            "repetition_penalty": base_rep_penalty,
                            "attempts": max_retries,
                            "note": "Failed to generate valid SMILES after retries"
                        }
                else:
                    # RDKit not available, can't validate
                    isValid = None
                    final_temp = base_temperature
                    final_params = {
                        "temperature": base_temperature,
                        "top_k": base_top_k,
                        "top_p": base_top_p,
                        "repetition_penalty": base_rep_penalty,
                        "attempts": 1,
                        "note": "RDKit not available for validation"
                    }
                    break
        
        return {
            "sequence": sequence,
            "smiles": smiles,
            "isValid": isValid,
            "device": str(DEVICE),
            "deviceType": DEVICE_TYPE,
            "sampling": final_params,
            "rdkit_available": rdkit_ok
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

