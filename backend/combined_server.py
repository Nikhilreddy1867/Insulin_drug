#!/usr/bin/env python3
"""
Combined Backend Server

Sections:
- AUTH: registration, login, sessions, MongoDB
- MODEL: tokenizer/ProteinLM -> PCA -> MLP predictions
- OAUTH: Google OAuth login flow
"""

import os
import requests
import math
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib
from dotenv import load_dotenv
from flask import Flask, request, jsonify, session, redirect
from flask_cors import CORS
from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError
import bcrypt
import logging
from datetime import datetime, timedelta
import secrets
import random
from twilio.rest import Client as TwilioClient
from difflib import SequenceMatcher
from scipy.spatial.distance import hamming
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', secrets.token_hex(32))
# Allow frontend origin with credentials
CORS(app, supports_credentials=True, resources={r"/*": {"origins": ["http://localhost:5173", "http://127.0.0.1:5173"]}})

# MongoDB configuration
MONGO_URI = os.environ.get('MONGO_URI', 'mongodb://localhost:27017/')
client = MongoClient(MONGO_URI)
db = client['protein_prediction_db']
users_collection = db['users']
otps_collection = db['otps']
try:
    users_collection.create_index('username', unique=True)
except Exception as e:
    logger.warning(f"Index creation warning: {e}")

# Session configuration
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(hours=24)

def hash_password(password):
    """Hash a password using bcrypt"""
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def verify_password(password, hashed):
    """Verify a password against its hash"""
    return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))

# ==============
# OTP UTILITIES
# ==============

TWILIO_ACCOUNT_SID = os.environ.get('TWILIO_ACCOUNT_SID')
TWILIO_AUTH_TOKEN = os.environ.get('TWILIO_AUTH_TOKEN')
TWILIO_PHONE_NUMBER = os.environ.get('TWILIO_PHONE_NUMBER')
DEFAULT_COUNTRY_CODE = os.environ.get('DEFAULT_COUNTRY_CODE', '+91')
DEV_FAKE_OTP = os.environ.get('DEV_FAKE_OTP', 'false').lower() == 'true'

def normalize_phone(mobile: str) -> str:
    m = mobile.strip().replace(' ', '')
    if m.startswith('+'):
        return m
    if m.startswith('0'):
        m = m.lstrip('0')
    return f"{DEFAULT_COUNTRY_CODE}{m}"

def generate_otp(length: int = 6) -> str:
    return ''.join(str(random.randint(0, 9)) for _ in range(length))

def send_sms_otp(phone: str, code: str) -> bool:
    if DEV_FAKE_OTP or not (TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN and TWILIO_PHONE_NUMBER):
        logger.info(f"[DEV] OTP for {phone}: {code}")
        return True
    try:
        client = TwilioClient(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        client.messages.create(
            body=f"Your verification code is {code}",
            from_=TWILIO_PHONE_NUMBER,
            to=phone,
        )
        return True
    except Exception as e:
        logger.error(f"Twilio send error: {e}")
        return False

# =============================
# MODEL SECTION: Inference setup
# =============================

# Config
MAX_TOKEN_LENGTH = 512
WINDOW_LEN = 50
HIDDEN_DIM_1 = 512
HIDDEN_DIM_2 = 256
DROPOUT = 0.4

# Device selection: CUDA > MPS (Metal/M1) > CPU
if torch.cuda.is_available():
    device = torch.device("cuda")
    device_type = "GPU (CUDA)"
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    device_type = "GPU (Metal/M1)"
else:
    device = torch.device("cpu")
    device_type = "CPU"

logger.info(f"[Model] Using device: {device} ({device_type})")

# Optimize for M1 GPU
if device_type == "GPU (Metal/M1)":
    # Enable optimizations for MPS backend
    torch.mps.empty_cache()  # Clear cache at startup
    logger.info("[Model] M1 GPU optimizations enabled")

# Globals
custom_protein_lm = None
mlp_model = None
label_encoder = None
pca_model = None
sequence_generator = None
protein_to_smile = None
smiles_tokenizer = None  # CharTokenizer for MiniT5
# Fusion model globals
progen_model = None  # ProteinLM for embeddings (from final_ckpt.pt)
molt5_fusion = None  # MiniT5 decoder for fusion
fusion_model = None  # FusionModel that combines progen + molt5
fusion_tokenizer = None  # CharTokenizer loaded from JSON

# Custom tokenizer setup (same as model_server)
AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")
SPECIAL_TOKENS = ["<PAD>", "<SOS>", "<EOS>", "<UNK>"]
VOCAB = SPECIAL_TOKENS + AMINO_ACIDS
token2idx = {tok: idx for idx, tok in enumerate(VOCAB)}
idx2token = {idx: tok for tok, idx in token2idx.items()}
PAD_ID = token2idx["<PAD>"]
SOS_ID = token2idx["<SOS>"]
EOS_ID = token2idx["<EOS>"]
UNK_ID = token2idx["<UNK>"]

def tokenize(seq: str):
    ids = [SOS_ID]
    for ch in seq.strip().upper():
        ids.append(token2idx.get(ch, token2idx["<UNK>"]))
    ids.append(EOS_ID)
    return ids

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_head, attn_dropout=0.1):
        super().__init__()
        assert d_model % n_head == 0
        self.n_head = n_head
        self.d_head = d_model // n_head
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(attn_dropout)
    def forward(self, x, mask=None):
        B, T, C = x.size()
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        att = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)
        causal_mask = torch.tril(torch.ones(T, T, device=x.device)).unsqueeze(0).unsqueeze(0)
        att = att.masked_fill(causal_mask == 0, float("-inf"))
        if mask is not None:
            mask2 = mask.unsqueeze(1).unsqueeze(2)
            att = att.masked_fill(mask2 == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)
        out = torch.matmul(att, v)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
    def forward(self, x):
        return self.net(x)

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_head, d_ff, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_head, attn_dropout=dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff, dropout)
    def forward(self, x, mask):
        x = x + self.attn(self.ln1(x), mask)
        x = x + self.ff(self.ln2(x))
        return x

# Fusion Model TransformerBlock - uses nn.MultiheadAttention (matches working code)
class FusionTransformerBlock(nn.Module):
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
	def __init__(self, vocab_size=len(VOCAB), d_model=256, nhead=8, num_layers=6, d_ff=1024, max_len=1024, dropout=0.1, with_head=False):
		super().__init__()
		self.token_emb = nn.Embedding(vocab_size, d_model)
		self.pos_emb = nn.Embedding(max_len, d_model)
		self.layers = nn.ModuleList([TransformerBlock(d_model, nhead, d_ff, dropout) for _ in range(num_layers)])
		self.ln_f = nn.LayerNorm(d_model)
		self.max_len = max_len
		self.d_model = d_model
		self.with_head = with_head
		if with_head:
			self.head = nn.Linear(d_model, vocab_size, bias=False)
	def forward(self, input_ids, attention_mask=None):
		B, T = input_ids.size()
		positions = torch.arange(0, T, device=input_ids.device).unsqueeze(0)
		x = self.token_emb(input_ids) + self.pos_emb(positions)
		for layer in self.layers:
			x = layer(x, attention_mask)
		x = self.ln_f(x)
		if self.with_head:
			return self.head(x)  # (B, T, vocab_size) - logits for next token prediction
		return x  # (B, T, d_model) - embeddings
	
	def embed(self, input_ids):
		"""Extract embeddings for fusion model"""
		B, T = input_ids.size()
		positions = torch.arange(0, T, device=input_ids.device).unsqueeze(0)
		x = self.token_emb(input_ids) + self.pos_emb(positions)
		for layer in self.layers:
			x = layer(x, None)
		return self.ln_f(x)

# Fusion Model ProteinLM - uses FusionTransformerBlock (matches working code)
class FusionProteinLM(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=8, num_layers=6, d_ff=1024, max_len=512):
        super().__init__()
        self.token_emb, self.pos_emb = nn.Embedding(vocab_size, d_model), nn.Embedding(max_len, d_model)
        self.layers = nn.ModuleList([FusionTransformerBlock(d_model, nhead, d_ff) for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(d_model)

    def embed(self, input_ids):
        B, T = input_ids.size()
        x = self.token_emb(input_ids) + self.pos_emb(torch.arange(0, T, device=input_ids.device).unsqueeze(0))
        for layer in self.layers:
            x = layer(x)
        return self.ln_f(x)

# ==============
# MiniT5 Architecture for SMILES Generation
# ==============

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

class MiniT5FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.fc2(self.dropout(F.relu(self.fc1(x))))

class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads)
        self.ff = MiniT5FeedForward(d_model, d_ff, dropout)
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
        self.ff = MiniT5FeedForward(d_model, d_ff, dropout)
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

    @torch.no_grad()
    def generate(self, src, tokenizer, max_len=256):
        self.eval()
        enc = self.encoder(src)
        B = src.size(0)
        bos_id = tokenizer.stoi[tokenizer.BOS]
        ys = torch.ones(B, 1, dtype=torch.long, device=src.device) * bos_id
        eos_id = tokenizer.stoi[tokenizer.EOS]
        
        for _ in range(max_len - 1):
            L = ys.size(1)
            tgt_mask = torch.tril(torch.ones((L, L), device=src.device)).unsqueeze(0).unsqueeze(0)
            logits = self.decoder(ys, enc, tgt_mask=tgt_mask)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            ys = torch.cat([ys, next_token], dim=1)
            if (next_token == eos_id).all():
                break
        return ys

# CharTokenizer for MiniT5 (matching training script)
class CharTokenizer:
    def __init__(self, corpus=None, tokens=None, special=("<pad>", "<s>", "</s>", "<unk>")):
        if tokens is not None:
            # Initialize from provided tokens
            self.PAD, self.BOS, self.EOS, self.UNK = special
            self.tokens = [self.PAD, self.BOS, self.EOS, self.UNK] + tokens
        else:
            # Initialize from corpus (backward compatibility)
            if corpus is None:
                corpus = []
            chars = sorted(set("".join(corpus))) if corpus else []
            self.PAD, self.BOS, self.EOS, self.UNK = special
            self.tokens = [self.PAD, self.BOS, self.EOS, self.UNK] + chars
        self.stoi = {s: i for i, s in enumerate(self.tokens)}
        self.itos = {i: s for s, i in self.stoi.items()}

    def encode(self, text: str, add_special: bool = True, max_len: int = None) -> list:
        ids = [self.stoi.get(ch, self.stoi[self.UNK]) for ch in text]
        if add_special:
            ids = [self.stoi[self.BOS]] + ids + [self.stoi[self.EOS]]
        if max_len is not None:
            ids = ids[:max_len]
        return ids

    def decode(self, ids: list) -> str:
        pieces = []
        for i in ids:
            if i == self.stoi[self.PAD]:
                continue
            if i == self.stoi[self.BOS] or i == self.stoi[self.EOS]:
                continue
            pieces.append(self.itos.get(i, self.UNK))
        return "".join(pieces)

    def __len__(self):
        return len(self.tokens)

# ==============
# Fusion Model Architecture for SMILES Generation
# ==============

class MultiHeadAttentionCustom(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.nh, self.dk = n_heads, d_model // n_heads
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
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, Lq, D)
        return self.out(out)

class FusionDecoderLayer(nn.Module):
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

class FusionDecoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, n_heads, d_ff, max_len):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([FusionDecoderLayer(d_model, n_heads, d_ff) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.out = nn.Linear(d_model, vocab_size)

    def forward(self, tgt, enc_out, tgt_mask=None):
        x = self.pos(self.tok_emb(tgt))
        for l in self.layers:
            x = l(x, enc_out, tgt_mask)
        return self.out(self.norm(x))

class FusionMiniT5(nn.Module):
    """MiniT5 decoder-only for fusion model"""
    def __init__(self, vocab_size, d_model=512, d_ff=1024, n_layers=6, n_heads=8, max_len=512):
        super().__init__()
        self.decoder = FusionDecoder(vocab_size, d_model, n_layers, n_heads, d_ff, max_len)

    def decoder_forward(self, tgt, enc_out, tgt_mask=None):
        return self.decoder(tgt, enc_out, tgt_mask)

    @torch.no_grad()
    def generate_from_enc(self, enc_out, tokenizer, max_len=128, temperature=1.5, top_k=50, top_p=0.9, repetition_penalty=1.15):
        """Generate with sampling for novel outputs"""
        B = enc_out.size(0)
        ys = torch.tensor([[tokenizer.stoi[tokenizer.BOS]]] * B, device=enc_out.device)
        eos_id = tokenizer.stoi[tokenizer.EOS]
        
        for _ in range(max_len - 1):
            tgt_mask = torch.tril(torch.ones((ys.size(1), ys.size(1)), device=enc_out.device)).unsqueeze(0).unsqueeze(0)
            logits = self.decoder(ys, enc_out, tgt_mask)
            logits = logits[:, -1, :] / max(temperature, 1e-8)
            
            # Apply repetition penalty
            if repetition_penalty > 1.0 and ys.size(1) > 1:
                token_ids = torch.arange(logits.size(-1), device=logits.device).unsqueeze(0).unsqueeze(0)
                token_counts = (ys.unsqueeze(-1) == token_ids).sum(dim=1).float()
                penalty_factor = 1.0 + (repetition_penalty - 1.0) * token_counts
                logits = logits / penalty_factor
            
            # Apply top-k filtering
            if top_k > 0:
                vocab_size = logits.size(-1)
                k = min(top_k, vocab_size)
                top_k_values, top_k_indices = torch.topk(logits, k, dim=-1)
                logits_filtered = torch.full_like(logits, float('-inf'))
                logits_filtered.scatter_(-1, top_k_indices, top_k_values)
                logits = logits_filtered
            
            # Apply top-p (nucleus) sampling
            if top_p > 0.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float('-inf')
            
            # Handle edge case: if all logits are -inf, reset to uniform
            if torch.isinf(logits).all():
                logits = torch.zeros_like(logits)
            
            # Sample from probability distribution
            probs = F.softmax(logits, dim=-1)
            if torch.isnan(probs).any() or torch.isinf(probs).any():
                probs = torch.softmax(torch.randn_like(logits) * 0.1, dim=-1)
            
            next_token = torch.multinomial(probs, num_samples=1)
            ys = torch.cat([ys, next_token], dim=1)
            if (next_token == eos_id).all():
                break
        
        return ys

class FusionModel(nn.Module):
    def __init__(self, progen_model, molt5_model, proj_in=256, proj_out=512):
        super().__init__()
        self.progen, self.molt5 = progen_model, molt5_model
        self.proj = nn.Sequential(
            nn.Linear(proj_in, proj_out), nn.LayerNorm(proj_out), nn.ReLU(), nn.Dropout(0.1)
        )

    def forward(self, seq_ids, tgt_ids, tokenizer):
        with torch.no_grad():
            seq_h = self.progen.embed(seq_ids)
        pooled = seq_h.mean(dim=1)
        enc_emb = self.proj(pooled).unsqueeze(1)
        tgt_in = tgt_ids[:, :-1]
        labels = tgt_ids[:, 1:].contiguous().view(-1)
        tgt_mask = torch.tril(torch.ones((tgt_in.size(1), tgt_in.size(1)), device=tgt_in.device)).unsqueeze(0).unsqueeze(0)
        logits = self.molt5.decoder_forward(tgt_in, enc_emb, tgt_mask=tgt_mask)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels, ignore_index=tokenizer.stoi[tokenizer.PAD])
        return logits, loss

# ==============
# Fusion Model Helper Functions
# ==============

def load_tokenizer_from_json(tokenizer_json_path: str) -> CharTokenizer:
    """Load tokenizer from JSON file"""
    try:
        with open(tokenizer_json_path, "r") as f:
            data = json.load(f)
        special = tuple(data.get("special", ["<pad>", "<s>", "</s>", "<unk>"]))
        tokens = data.get("tokens", [])
        return CharTokenizer(tokens=tokens, special=special)
    except Exception as e:
        logger.error(f"[Fusion] Failed to load tokenizer from {tokenizer_json_path}: {e}")
        return None

class MLP(nn.Module):
    def __init__(self, input_dim, hidden1=HIDDEN_DIM_1, hidden2=HIDDEN_DIM_2, num_classes=4, dropout=DROPOUT):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.BatchNorm1d(hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.BatchNorm1d(hidden2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden2, num_classes)
        )
    def forward(self, x):
        return self.net(x)

def load_models():
    global custom_protein_lm, mlp_model, label_encoder, pca_model, sequence_generator, protein_to_smile, smiles_tokenizer
    global progen_model, molt5_fusion, fusion_model, fusion_tokenizer
    try:
        logger.info("[Model] Loading models...")
        # Resolve absolute paths for clear diagnostics
        le_path = os.path.abspath("models/label_encoder.pkl")
        pca_path = os.path.abspath("models/pca_model.pkl")
        mlp_path = os.path.abspath("models/best_mlp_medium_adv.pth")
        seqgen_path = os.path.abspath("models/Sequence_Generator.pt")
        prot2smiles_path = os.path.abspath("models/Protein_to_Smile.pt")
        molt5_path = os.path.abspath("models/molt5_best.pt")
        # Try protein_classifier.pt first, fallback to custom_protein_lm.pt, then final_ckpt.pt
        protein_classifier_path = os.path.abspath("models/protein_classifier.pt")
        custom_lm_ckpt_path = os.path.abspath("models/custom_protein_lm.pt")
        final_ckpt_path = os.path.abspath("models/final_ckpt.pt")
        
        # ProteinLM - CRITICAL: Must load trained checkpoint for correct embeddings!
        custom_protein_lm = ProteinLM()
        checkpoint_loaded = False
        
        if os.path.exists(protein_classifier_path):
            logger.info(f"[Model] Loading ProteinLM from: {protein_classifier_path}")
            try:
                ckpt = torch.load(protein_classifier_path, map_location=device)
                
                # Handle different checkpoint formats (OrderedDict or dict)
                from collections import OrderedDict
                if isinstance(ckpt, (dict, OrderedDict)):
                    # Try to get model_state first
                    state_dict = ckpt.get("model_state", ckpt)
                    # Check if keys have 'base.' prefix (wrapped model)
                    if state_dict and len(state_dict) > 0:
                        first_key = next(iter(state_dict.keys()))
                        if first_key.startswith('base.'):
                            # Remove 'base.' prefix from all keys
                            logger.info("[Model] Detected 'base.' prefix in checkpoint, stripping...")
                            new_state_dict = {}
                            for key, value in state_dict.items():
                                new_key = key.replace('base.', '')
                                new_state_dict[new_key] = value
                            state_dict = new_state_dict
                    else:
                        state_dict = ckpt
                else:
                    state_dict = ckpt
                
                # Try loading the state dict
                result = custom_protein_lm.load_state_dict(state_dict, strict=False)
                if result.missing_keys:
                    logger.warning(f"[Model] Missing keys: {result.missing_keys[:5]}...")
                if result.unexpected_keys:
                    logger.warning(f"[Model] Unexpected keys: {result.unexpected_keys[:5]}...")
                logger.info("[Model] ✅ ProteinLM loaded successfully from protein_classifier.pt")
                checkpoint_loaded = True
            except Exception as e:
                logger.error(f"[Model] Failed to load protein_classifier.pt: {e}")
                import traceback
                logger.error(traceback.format_exc())
                logger.warning("[Model] Trying fallback checkpoints...")
        
        if not checkpoint_loaded and os.path.exists(custom_lm_ckpt_path):
            logger.info(f"[Model] Loading ProteinLM from: {custom_lm_ckpt_path}")
            try:
                ckpt = torch.load(custom_lm_ckpt_path, map_location=device)
                custom_protein_lm.load_state_dict(ckpt.get("model_state", ckpt), strict=False)
                logger.info("[Model] ✅ ProteinLM loaded successfully from custom_protein_lm.pt")
                checkpoint_loaded = True
            except Exception as e:
                logger.warning(f"[Model] Failed to load custom_protein_lm.pt: {e}, trying fallback...")
        
        if not checkpoint_loaded and os.path.exists(final_ckpt_path):
            logger.info(f"[Model] Loading ProteinLM from: {final_ckpt_path}")
            try:
                ckpt = torch.load(final_ckpt_path, map_location=device)
                custom_protein_lm.load_state_dict(ckpt.get("model_state", ckpt), strict=False)
                logger.info("[Model] ✅ ProteinLM loaded successfully from final_ckpt.pt")
                checkpoint_loaded = True
            except Exception as e:
                logger.warning(f"[Model] Failed to load final_ckpt.pt: {e}")
        
        if not checkpoint_loaded:
            logger.error("=" * 60)
            logger.error("[Model] ❌ CRITICAL ERROR: ProteinLM checkpoint NOT FOUND!")
            logger.error(f"[Model] Tried: {protein_classifier_path}")
            logger.error(f"[Model] Tried: {custom_lm_ckpt_path}")
            logger.error(f"[Model] Tried: {final_ckpt_path}")
            logger.error("[Model] ⚠️  Using RANDOM weights - predictions will be INCORRECT!")
            logger.error("=" * 60)
        
        custom_protein_lm.to(device).eval()

        # Label encoder, PCA
        logger.info(f"[Model] Loading label encoder: {le_path} (exists={os.path.exists(le_path)})")
        label_encoder = joblib.load(le_path)
        logger.info(f"[Model] Loading PCA model: {pca_path} (exists={os.path.exists(pca_path)})")
        pca_model = joblib.load(pca_path)
        pca_dim = getattr(pca_model, 'n_components_', None) or 512

        # MLP - Try both possible model filenames
        num_classes = len(label_encoder.classes_)
        mlp_model = MLP(input_dim=pca_dim, num_classes=num_classes).to(device)
        
        # Try custom_adv first (from notebook), fallback to medium_adv
        mlp_custom_path = os.path.abspath("models/best_mlp_custom_adv.pth")
        mlp_medium_path = os.path.abspath("models/best_mlp_medium_adv.pth")
        
        if os.path.exists(mlp_custom_path):
            logger.info(f"[Model] Loading MLP weights: {mlp_custom_path} (exists={os.path.exists(mlp_custom_path)})")
            state = torch.load(mlp_custom_path, map_location=device)
            mlp_model.load_state_dict(state, strict=False)
            logger.info("[Model] Loaded best_mlp_custom_adv.pth (matches training)")
        elif os.path.exists(mlp_medium_path):
            logger.info(f"[Model] Loading MLP weights: {mlp_medium_path} (exists={os.path.exists(mlp_medium_path)})")
            logger.warning("[Model] Using best_mlp_medium_adv.pth - consider using best_mlp_custom_adv.pth from training")
            state = torch.load(mlp_medium_path, map_location=device)
            mlp_model.load_state_dict(state, strict=False)
        else:
            raise FileNotFoundError(f"Neither {mlp_custom_path} nor {mlp_medium_path} found. Please ensure trained model exists.")
        
        mlp_model.eval()
        
        # Sequence Generator - needs ProteinLM with head for generation
        try:
            logger.info(f"[Model] Loading Sequence Generator: {seqgen_path} (exists={os.path.exists(seqgen_path)})")
            if os.path.exists(seqgen_path):
                seqgen_ckpt = torch.load(seqgen_path, map_location=device)
                
                # Extract model_state if it's a checkpoint dict
                if isinstance(seqgen_ckpt, dict) and "model_state" in seqgen_ckpt:
                    state_dict = seqgen_ckpt["model_state"]
                    # Get args from checkpoint if available
                    ckpt_args = seqgen_ckpt.get("args", {})
                else:
                    state_dict = seqgen_ckpt
                    ckpt_args = {}
                
                # Create ProteinLM with head for generation
                seqgen_model = ProteinLM(
                    vocab_size=len(VOCAB),
                    d_model=ckpt_args.get("d_model", 256),
                    nhead=ckpt_args.get("nhead", 8),
                    num_layers=ckpt_args.get("num_layers", 6),
                    d_ff=ckpt_args.get("d_ff", 1024),
                    max_len=ckpt_args.get("max_len", 1024),
                    dropout=ckpt_args.get("dropout", 0.1),
                    with_head=True  # Generation needs head
                )
                
                # Check for key prefixes and strip them
                if len(state_dict) > 0:
                    first_key = next(iter(state_dict.keys()))
                    if first_key.startswith('base.'):
                        logger.info("[Model] Detected 'base.' prefix in sequence generator, stripping...")
                        new_state_dict = {}
                        for key, value in state_dict.items():
                            new_key = key.replace('base.', '')
                            new_state_dict[new_key] = value
                        state_dict = new_state_dict
                
                result = seqgen_model.load_state_dict(state_dict, strict=False)
                if result.missing_keys:
                    logger.warning(f"[Model] Sequence generator missing keys: {len(result.missing_keys)} keys")
                if result.unexpected_keys:
                    logger.warning(f"[Model] Sequence generator unexpected keys: {len(result.unexpected_keys)} keys")
                
                seqgen_model.to(device).eval()
                sequence_generator = seqgen_model
                logger.info("[Model] ✅ Sequence generator loaded successfully")
            else:
                raise FileNotFoundError(f"Sequence generator checkpoint not found: {seqgen_path}")
        except Exception as e:
            logger.error(f"[Model] Failed to load sequence generator: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            sequence_generator = None
        
        # Protein to SMILES Generator - MiniT5 (encoder-decoder)
        try:
            # Try MiniT5 first (molt5_best.pt)
            if os.path.exists(molt5_path):
                logger.info(f"[Model] Loading MiniT5 SMILES Generator: {molt5_path}")
                molt5_ckpt = torch.load(molt5_path, map_location=device)
                
                # Extract state dict and args
                if isinstance(molt5_ckpt, dict):
                    state_dict = molt5_ckpt.get("model_state", molt5_ckpt)
                    ckpt_args = molt5_ckpt.get("args", {})
                    # Try to extract tokenizer info if saved
                    if "tokenizer_tokens" in molt5_ckpt:
                        tokenizer_tokens = molt5_ckpt["tokenizer_tokens"]
                        smiles_tokenizer = CharTokenizer(corpus=[])
                        smiles_tokenizer.tokens = tokenizer_tokens
                        smiles_tokenizer.stoi = {s: i for i, s in enumerate(tokenizer_tokens)}
                        smiles_tokenizer.itos = {i: s for s, i in smiles_tokenizer.stoi.items()}
                        vocab_size = len(smiles_tokenizer)
                        logger.info(f"[Model] Loaded tokenizer with vocab size: {vocab_size}")
                    else:
                        # Extract vocab size from checkpoint embedding layer
                        # Find the encoder embedding layer to get the actual vocab size
                        vocab_size = None
                        for key in state_dict.keys():
                            if "encoder.tok_emb.weight" in key or "tok_emb.weight" in key:
                                vocab_size = state_dict[key].shape[0]
                                logger.info(f"[Model] Extracted vocab_size={vocab_size} from checkpoint embedding layer: {key}")
                                break
                        
                        if vocab_size is None:
                            logger.error("[Model] Could not determine vocab_size from checkpoint")
                            raise ValueError("Cannot determine vocab_size from checkpoint")
                        
                        # Create a minimal tokenizer that matches the vocab size
                        # We'll use common characters, but the exact mapping doesn't matter as long as size matches
                        # The model will work, but decoding might not be perfect without exact tokenizer
                        smiles_tokenizer = CharTokenizer(corpus=[])
                        # Create tokens list with special tokens + enough chars to match vocab_size
                        special_tokens = ["<pad>", "<s>", "</s>", "<unk>"]
                        num_chars_needed = vocab_size - len(special_tokens)
                        # Use common SMILES and text characters
                        common_chars = "CNOSFBrIPc1234567890[]()=#+-nopsABCDEFGHIKLMNPQRSTVWYabcdefghijklmnopqrstuvwxyz.,;:!? "
                        chars_to_use = sorted(list(set(common_chars)))[:num_chars_needed]
                        smiles_tokenizer.tokens = special_tokens + chars_to_use
                        smiles_tokenizer.stoi = {s: i for i, s in enumerate(smiles_tokenizer.tokens)}
                        smiles_tokenizer.itos = {i: s for s, i in smiles_tokenizer.stoi.items()}
                        
                        if len(smiles_tokenizer) != vocab_size:
                            # Pad with dummy tokens if needed
                            while len(smiles_tokenizer.tokens) < vocab_size:
                                dummy_char = f"<extra{len(smiles_tokenizer.tokens)}>"
                                smiles_tokenizer.tokens.append(dummy_char)
                            smiles_tokenizer.stoi = {s: i for i, s in enumerate(smiles_tokenizer.tokens)}
                            smiles_tokenizer.itos = {i: s for s, i in smiles_tokenizer.stoi.items()}
                        
                        logger.info(f"[Model] Created tokenizer with vocab size: {len(smiles_tokenizer)} (matched to checkpoint)")
                else:
                    # Direct state dict
                    state_dict = molt5_ckpt
                    ckpt_args = {}
                    # Extract vocab size from embedding layer
                    vocab_size = None
                    for key in state_dict.keys():
                        if "encoder.tok_emb.weight" in key or "tok_emb.weight" in key:
                            vocab_size = state_dict[key].shape[0]
                            logger.info(f"[Model] Extracted vocab_size={vocab_size} from checkpoint: {key}")
                            break
                    
                    if vocab_size is None:
                        logger.error("[Model] Could not determine vocab_size from checkpoint")
                        raise ValueError("Cannot determine vocab_size from checkpoint")
                    
                    # Create minimal tokenizer matching vocab_size
                    special_tokens = ["<pad>", "<s>", "</s>", "<unk>"]
                    num_chars_needed = vocab_size - len(special_tokens)
                    common_chars = "CNOSFBrIPc1234567890[]()=#+-nopsABCDEFGHIKLMNPQRSTVWYabcdefghijklmnopqrstuvwxyz.,;:!? "
                    chars_to_use = sorted(list(set(common_chars)))[:num_chars_needed]
                    smiles_tokenizer = CharTokenizer(corpus=[])
                    smiles_tokenizer.tokens = special_tokens + chars_to_use
                    if len(smiles_tokenizer.tokens) < vocab_size:
                        while len(smiles_tokenizer.tokens) < vocab_size:
                            smiles_tokenizer.tokens.append(f"<extra{len(smiles_tokenizer.tokens)}>")
                    smiles_tokenizer.stoi = {s: i for i, s in enumerate(smiles_tokenizer.tokens)}
                    smiles_tokenizer.itos = {i: s for s, i in smiles_tokenizer.stoi.items()}
                
                # Create MiniT5 model with EXACT vocab_size from checkpoint
                d_model = ckpt_args.get("d_model", 512)
                d_ff = ckpt_args.get("d_ff", 1024)
                n_layers = ckpt_args.get("n_layers", 6)
                n_heads = ckpt_args.get("n_heads", 8)
                max_len = ckpt_args.get("max_len", 512)
                
                logger.info(f"[Model] Creating MiniT5 with vocab_size={vocab_size}, d_model={d_model}, n_layers={n_layers}, n_heads={n_heads}")
                molt5_model = MiniT5(vocab_size=vocab_size, d_model=d_model, d_ff=d_ff, 
                                    n_layers=n_layers, n_heads=n_heads, max_len=max_len)
                
                # Get initial weights (before loading) to verify they change
                initial_weights_before = {}
                for name, param in molt5_model.named_parameters():
                    if "encoder.tok_emb.weight" in name:
                        initial_weights_before[name] = param.data.clone()
                        logger.info(f"[Model] Initial embedding weight sample (before load): mean={param.data.mean().item():.6f}, std={param.data.std().item():.6f}")
                        logger.info(f"[Model] First 5 values: {param.data.flatten()[:5].tolist()}")
                        break
                
                # Load state dict with STRICT=False to handle any minor mismatches
                logger.info("[Model] Loading state dict from checkpoint...")
                result = molt5_model.load_state_dict(state_dict, strict=False)
                if result.missing_keys:
                    logger.warning(f"[Model] MiniT5 missing keys: {len(result.missing_keys)} keys")
                    logger.debug(f"[Model] First 5 missing: {result.missing_keys[:5]}")
                if result.unexpected_keys:
                    logger.warning(f"[Model] MiniT5 unexpected keys: {len(result.unexpected_keys)} keys")
                    logger.debug(f"[Model] First 5 unexpected: {result.unexpected_keys[:5]}")
                
                # Verify weights were loaded (not random)
                weights_loaded = False
                for name, param in molt5_model.named_parameters():
                    if "encoder.tok_emb.weight" in name and name in initial_weights_before:
                        if not torch.equal(param.data, initial_weights_before[name]):
                            weights_loaded = True
                            logger.info(f"[Model] ✅ VERIFIED: Weights changed - using SAVED weights from checkpoint")
                            logger.info(f"[Model] After load - mean={param.data.mean().item():.6f}, std={param.data.std().item():.6f}")
                            logger.info(f"[Model] First 5 values: {param.data.flatten()[:5].tolist()}")
                            break
                
                if not weights_loaded:
                    logger.error("[Model] ⚠️  WARNING: Weights may not have loaded correctly - checking...")
                    # Check if any weights match checkpoint
                    for name, param in molt5_model.named_parameters():
                        if name in state_dict:
                            if torch.equal(param.data, state_dict[name]):
                                logger.info(f"[Model] ✅ Weight {name} matches checkpoint")
                                weights_loaded = True
                                break
                
                molt5_model.to(device).eval()
                protein_to_smile = molt5_model
                if weights_loaded:
                    logger.info("[Model] ✅ MiniT5 SMILES generator loaded successfully with SAVED WEIGHTS")
                else:
                    logger.error("[Model] ❌ MiniT5 loaded but weights verification failed - may be using RANDOM weights!")
            # Fallback to old Protein_to_Smile.pt if molt5_best.pt doesn't exist
            elif os.path.exists(prot2smiles_path):
                logger.info(f"[Model] Loading Protein->SMILES Generator (legacy): {prot2smiles_path}")
                prot2smile_ckpt = torch.load(prot2smiles_path, map_location=device)
                
                # Handle OrderedDict or dict
                from collections import OrderedDict
                if isinstance(prot2smile_ckpt, (dict, OrderedDict)):
                    if "model_state" in prot2smile_ckpt:
                        state_dict = prot2smile_ckpt["model_state"]
                        ckpt_args = prot2smile_ckpt.get("args", {})
                    else:
                        state_dict = prot2smile_ckpt
                        ckpt_args = {}
                else:
                    state_dict = prot2smile_ckpt
                    ckpt_args = {}
                
                # Check for 'progen.' prefix (as seen in the checkpoint)
                if len(state_dict) > 0:
                    first_key = next(iter(state_dict.keys()))
                    if first_key.startswith('progen.'):
                        logger.info("[Model] Detected 'progen.' prefix in Protein->SMILES generator, stripping...")
                        new_state_dict = {}
                        for key, value in state_dict.items():
                            new_key = key.replace('progen.', '')
                            new_state_dict[new_key] = value
                        state_dict = new_state_dict
                
                # Try to create ProteinLM model (might need adjustment based on actual architecture)
                # For now, we'll use ProteinLM and see if it loads
                prot2smile_model = ProteinLM(
                    vocab_size=len(VOCAB),
                    d_model=ckpt_args.get("d_model", 256),
                    nhead=ckpt_args.get("nhead", 8),
                    num_layers=ckpt_args.get("num_layers", 6),
                    d_ff=ckpt_args.get("d_ff", 1024),
                    max_len=ckpt_args.get("max_len", 1024),
                    dropout=ckpt_args.get("dropout", 0.1),
                    with_head=True  # SMILES generation likely needs head
                )
                
                result = prot2smile_model.load_state_dict(state_dict, strict=False)
                if result.missing_keys:
                    logger.warning(f"[Model] Protein->SMILES missing keys: {len(result.missing_keys)} keys")
                if result.unexpected_keys:
                    logger.warning(f"[Model] Protein->SMILES unexpected keys: {len(result.unexpected_keys)} keys")
                
                prot2smile_model.to(device).eval()
                protein_to_smile = prot2smile_model
                logger.info("[Model] ✅ Protein to SMILES generator loaded successfully (legacy)")
            else:
                logger.warning(f"[Model] Neither molt5_best.pt nor Protein_to_Smile.pt found - SMILES generation will use placeholder")
        except Exception as e:
            logger.error(f"[Model] Failed to load Protein->SMILES generator: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            protein_to_smile = None
        
        # ==============
        # Load Fusion Model for SMILES Generation
        # ==============
        try:
            fusion_progen_path = os.path.abspath("models/progen.pt")
            fusion_molt5_path = os.path.abspath("models/molt5.pt")
            fusion_model_path = os.path.abspath("models/fusion_best.pt")
            tokenizer_json_path = os.path.abspath("models/tokenizer.json")
            
            # Check if fusion model files exist
            if os.path.exists(fusion_model_path) and os.path.exists(tokenizer_json_path):
                logger.info("[Fusion] Loading Fusion Model for SMILES generation...")
                
                # Load tokenizer
                fusion_tokenizer = load_tokenizer_from_json(tokenizer_json_path)
                if fusion_tokenizer is None:
                    logger.warning("[Fusion] Failed to load tokenizer, skipping fusion model")
                else:
                    logger.info(f"[Fusion] ✅ Loaded tokenizer with vocab size: {len(fusion_tokenizer)}")
                    
                    # Load models EXACTLY matching the working code's build_models function
                    PROGEN_DMODEL, MOLT5_DMODEL = 256, 512
                    fusion_vocab_size = len(fusion_tokenizer)
                    
                    # Build models matching the user's build_models function EXACTLY
                    if os.path.exists(fusion_progen_path) and os.path.exists(fusion_molt5_path):
                        logger.info("[Fusion] Building fusion models (matching working code)...")
                        
                        # Create models with tokenizer vocab size (matches working code exactly)
                        logger.info(f"[Fusion] Creating ProGen model with vocab_size={fusion_vocab_size}")
                        progen_model = FusionProteinLM(vocab_size=fusion_vocab_size, d_model=PROGEN_DMODEL).to(device)
                        
                        logger.info(f"[Fusion] Creating MolT5 model with vocab_size={fusion_vocab_size}")
                        molt5_fusion = FusionMiniT5(vocab_size=fusion_vocab_size, d_model=MOLT5_DMODEL).to(device)
                        
                        # Create fusion model wrapper
                        fusion_model = FusionModel(progen_model, molt5_fusion, proj_in=PROGEN_DMODEL, proj_out=MOLT5_DMODEL).to(device)
                        
                        # Load checkpoints EXACTLY like working code (non-strict, no prefix handling)
                        # The working code loads all three checkpoints separately, then fusion_best.pt may override
                        try:
                            logger.info(f"[Fusion] Loading ProGen checkpoint: {fusion_progen_path}")
                            progen_ckpt = torch.load(fusion_progen_path, map_location=device)
                            progen_model.load_state_dict(progen_ckpt, strict=False)
                            logger.info("[Fusion] ✅ ProGen loaded")
                        except Exception as e:
                            logger.warning(f"[Fusion] ProGen checkpoint load issue: {e}")
                        
                        # Skip separate MolT5 loading - fusion_best.pt contains the correct decoder weights
                        # The separate molt5.pt may have different vocab size, fusion_best.pt is the source of truth
                        logger.info("[Fusion] MolT5 decoder will be loaded from fusion_best.pt checkpoint")
                        
                        try:
                            logger.info(f"[Fusion] Loading Fusion checkpoint: {fusion_model_path}")
                            fusion_ckpt = torch.load(fusion_model_path, map_location=device)
                            # Fusion checkpoint contains progen., molt5., and proj. prefixes
                            # Distribute to respective models
                            progen_state = {}
                            molt5_state = {}
                            proj_state = {}
                            
                            for k, v in fusion_ckpt.items():
                                if k.startswith('progen.'):
                                    progen_state[k[7:]] = v  # Remove 'progen.' prefix -> direct key
                                elif k.startswith('molt5.'):
                                    # Remove 'molt5.' prefix, keep 'decoder.xxx' structure for FusionMiniT5
                                    new_key = k[6:]  # Remove 'molt5.' -> 'decoder.xxx'
                                    # FusionMiniT5 has self.decoder, so keys should be 'decoder.xxx'
                                    molt5_state[new_key] = v
                                elif k.startswith('proj.'):
                                    proj_state[k] = v
                            
                            # Load into respective models
                            if progen_state:
                                progen_model.load_state_dict(progen_state, strict=False)
                            if molt5_state:
                                molt5_fusion.load_state_dict(molt5_state, strict=False)
                            if proj_state:
                                fusion_model.proj.load_state_dict(proj_state, strict=False)
                            
                            logger.info("[Fusion] ✅ Fusion checkpoint loaded (distributed to models)")
                        except Exception as e:
                            logger.warning(f"[Fusion] Fusion checkpoint load issue: {e}")
                        
                        # Freeze progen parameters
                        for p in progen_model.parameters():
                            p.requires_grad = False
                        
                        # Set to eval mode
                        progen_model.eval()
                        molt5_fusion.eval()
                        fusion_model.eval()
                        logger.info("[Fusion] ✅ Fusion model fully loaded and ready!")
                    else:
                        logger.warning(f"[Fusion] Required model files not found:")
                        logger.warning(f"  - ProGen: {fusion_progen_path} (exists: {os.path.exists(fusion_progen_path)})")
                        logger.warning(f"  - MolT5: {fusion_molt5_path} (exists: {os.path.exists(fusion_molt5_path)})")
                        progen_model = None
                        molt5_fusion = None
                        fusion_model = None
            else:
                logger.info("[Fusion] Fusion model files not found, using legacy SMILES generation")
        except Exception as e:
            logger.error(f"[Fusion] Error loading fusion model: {e}")
            import traceback
            logger.error(traceback.format_exc())
        
        logger.info("[Model] All models loaded")
        
        # Final verification
        missing_critical = []
        if not checkpoint_loaded:
            missing_critical.append("ProteinLM checkpoint (protein_classifier.pt, custom_protein_lm.pt, or final_ckpt.pt - using random embeddings!)")
        if not os.path.exists(mlp_custom_path) and not os.path.exists(mlp_medium_path):
            missing_critical.append("MLP model file")
        if sequence_generator is None:
            missing_critical.append("Sequence Generator (Sequence_Generator.pt - using random mutations)")
        if protein_to_smile is None:
            missing_critical.append("SMILES Generator (molt5_best.pt or Protein_to_Smile.pt - using placeholder)")
        
        if missing_critical:
            logger.warning("[Model] ⚠️  WARNING: Missing critical files:")
            for f in missing_critical:
                logger.warning(f"[Model]   - {f}")
            if "ProteinLM checkpoint" in str(missing_critical):
                logger.warning("[Model] Predictions will be INCORRECT!")
            if "Sequence Generator" in str(missing_critical) or "Protein->SMILES" in str(missing_critical):
                logger.warning("[Model] Generation features may not work properly!")
        else:
            logger.info("[Model] ✅ All critical model files present")
    except Exception as e:
        logger.error(f"[Model] Load error: {e}")
        raise

@torch.no_grad()
def embed_sequence(seq: str):
    """
    Generate embeddings matching notebook's embed_batch function.
    Uses mean pooling over sequence length (matching notebook: outputs.mean(dim=1))
    """
    # Tokenize (includes SOS and EOS)
    ids = tokenize(seq)
    if len(ids) > MAX_TOKEN_LENGTH:
        ids = ids[:MAX_TOKEN_LENGTH-1] + [EOS_ID]
    
    # Pad to MAX_TOKEN_LENGTH
    padded = ids + [PAD_ID] * (MAX_TOKEN_LENGTH - len(ids))
    input_ids = torch.tensor([padded], dtype=torch.long).to(device)
    
    # Create attention mask (1 for real tokens, 0 for padding)
    mask = [1 if tok != PAD_ID else 0 for tok in padded]
    attention_mask = torch.tensor([mask], dtype=torch.long).to(device)
    
    # Forward through custom ProteinLM
    out = custom_protein_lm(input_ids, attention_mask=attention_mask)
    
    # Mean pooling over sequence dimension (B, T, d_model) -> (B, d_model)
    # This matches notebook: mean_emb = outputs.mean(dim=1)
    mean_emb = out.mean(dim=1)
    
    return mean_emb.detach().cpu().float().numpy()

def preprocess_sequence(seq: str) -> str:
    """
    Preprocess sequence to match notebook training pipeline.
    Uses sliding window approach with center alignment (shifts=[-2, 0, 2], default shift=0).
    """
    s = seq.strip().upper()
    valid = set("ACDEFGHIKLMNPQRSTVWY")
    if not s or not all(c in valid for c in s):
        raise ValueError("Sequence contains invalid amino acids")
    if len(s) < 10:
        raise ValueError("Sequence too short (minimum 10 amino acids)")
    
    # Match notebook preprocessing: sliding window with center alignment
    # During training, it uses shifts=[-2, 0, 2] but for inference we use shift=0 (center)
    L = len(s)
    
    if L > WINDOW_LEN:
        # Center crop (matching notebook's shift=0 window)
        mid = L // 2
        start = max(0, mid - WINDOW_LEN // 2)
        end = start + WINDOW_LEN
        if end > L:
            end = L
            start = max(0, end - WINDOW_LEN)
        s = s[start:end]
    else:
        # Pad with 'A' to match training augmentation
        s = s.ljust(WINDOW_LEN, 'A')
    
    # Ensure exactly WINDOW_LEN (safety check)
    if len(s) != WINDOW_LEN:
        if len(s) > WINDOW_LEN:
            s = s[:WINDOW_LEN]
        else:
            s = s.ljust(WINDOW_LEN, 'A')
    
    return s

# Similarity calculation functions
def levenshtein_similarity(seq1: str, seq2: str) -> float:
    """Calculate Levenshtein similarity between two sequences"""
    return SequenceMatcher(None, seq1, seq2).ratio()

def hamming_similarity(seq1: str, seq2: str) -> float:
    """Calculate Hamming similarity between two sequences"""
    if len(seq1) != len(seq2):
        # Pad shorter sequence
        max_len = max(len(seq1), len(seq2))
        seq1 = seq1.ljust(max_len, 'X')
        seq2 = seq2.ljust(max_len, 'X')
    
    # Convert to numeric arrays for hamming distance
    seq1_nums = [ord(c) for c in seq1]
    seq2_nums = [ord(c) for c in seq2]
    
    hamming_dist = hamming(seq1_nums, seq2_nums)
    return 1 - hamming_dist  # Convert distance to similarity

def cosine_similarity_sequences(seq1: str, seq2: str) -> float:
    """Calculate cosine similarity between two sequences"""
    # Create one-hot encoding
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    
    def encode_sequence(seq):
        encoding = []
        for char in seq:
            vector = [0] * len(amino_acids)
            if char in amino_acids:
                vector[amino_acids.index(char)] = 1
            encoding.append(vector)
        return np.array(encoding).flatten()
    
    vec1 = encode_sequence(seq1)
    vec2 = encode_sequence(seq2)
    
    # Ensure same length
    min_len = min(len(vec1), len(vec2))
    vec1 = vec1[:min_len]
    vec2 = vec2[:min_len]
    
    if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
        return 0.0
    
    return cosine_similarity([vec1], [vec2])[0][0]

def pearson_similarity(seq1: str, seq2: str) -> float:
    """Calculate Pearson correlation similarity between two sequences"""
    # Convert to numeric representation
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    
    def encode_sequence(seq):
        return [amino_acids.index(c) if c in amino_acids else 0 for c in seq]
    
    vec1 = encode_sequence(seq1)
    vec2 = encode_sequence(seq2)
    
    # Ensure same length
    min_len = min(len(vec1), len(vec2))
    vec1 = vec1[:min_len]
    vec2 = vec2[:min_len]
    
    if len(vec1) < 2:
        return 0.0
    
    try:
        correlation, _ = pearsonr(vec1, vec2)
        return max(0, correlation)  # Return only positive correlations
    except:
        return 0.0

@torch.no_grad()
def sample_sequence(model: nn.Module, context: str = "M", max_new_tokens: int = 200,
                    temperature: float = 1.0, top_k: int = 0, top_p: float = 0.0):
    """
    Autoregressive sequence generation using the trained model.
    Based on the training code's sample_sequence function.
    Optimized to use GPU for faster inference.
    """
    model.eval()
    # Use the global device (GPU preferred if available)
    gen_device = device
    
    start_ids = tokenize(context)
    if len(start_ids) == 0:
        start_ids = [SOS_ID]
    
    ids = torch.tensor(start_ids, dtype=torch.long, device=gen_device).unsqueeze(0)  # (1, L)
    
    # Optimize for GPU: batch tensor operations when possible
    for _ in range(max_new_tokens):
        if ids.size(1) > model.max_len:
            # Trim leftmost tokens (sliding window approach)
            ids = ids[:, -model.max_len:]
        
        attention_mask = (ids != PAD_ID).long()
        logits = model(ids, attention_mask=attention_mask)  # (1, L, V)
        logits = logits[:, -1, :] / max(1e-8, temperature)  # Take last token logits
        probs = F.softmax(logits, dim=-1)
        
        # Sampling strategy - keep operations on GPU
        if top_k > 0:
            topk_vals, topk_idx = torch.topk(probs, top_k)
            topk_vals = topk_vals / torch.sum(topk_vals, dim=-1, keepdim=True)
            # Keep sampling on GPU, only transfer final result
            next_id = topk_idx[0, torch.multinomial(topk_vals[0], 1)].item()
        elif 0.0 < top_p < 1.0:
            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
            cumulative = torch.cumsum(sorted_probs, dim=-1)
            cutoff = cumulative > top_p
            cutoff_index = torch.argmax(cutoff.float(), dim=-1).item()
            keep = cutoff_index + 1
            keep_idx = sorted_idx[:, :keep]
            keep_probs = sorted_probs[:, :keep]
            keep_probs = keep_probs / torch.sum(keep_probs, dim=-1, keepdim=True)
            next_id = keep_idx[0, torch.multinomial(keep_probs[0], 1)].item()
        else:
            next_id = torch.multinomial(probs[0], 1).item()
        
        # Efficient concatenation on GPU
        ids = torch.cat([ids, torch.tensor([[next_id]], dtype=torch.long, device=gen_device)], dim=1)
        
        if next_id == EOS_ID:
            break
    
    # Convert token IDs back to sequence (single transfer from GPU)
    seq_ids = ids.squeeze(0).cpu().tolist()
    tokens = []
    for i in seq_ids:
        if i in (PAD_ID, SOS_ID, EOS_ID):
            continue
        tokens.append(idx2token.get(i, "X"))
    return "".join(tokens)

def generate_sequences(input_seq: str, num_sequences: int = 10) -> list:
    """
    Generate new protein sequences using the sequence generator model.
    Uses GPU for optimized inference.
    """
    if sequence_generator is None:
        logger.warning("[Generate] Sequence generator not loaded, using fallback random mutations")
        # Fallback: generate random sequences based on input
        generated = []
        for _ in range(num_sequences):
            # Simple mutation-based generation
            seq = list(input_seq)
            for i in range(random.randint(1, 3)):
                if len(seq) > 0:
                    pos = random.randint(0, len(seq) - 1)
                    seq[pos] = random.choice(AMINO_ACIDS)
            generated.append(''.join(seq))
        return generated
    
    try:
        logger.info(f"[Generate] Generating {num_sequences} sequences from context: {input_seq[:30]}... (device: {device_type})")
        generated_sequences = []
        for i in range(num_sequences):
            # Use context as starting point (or just first few tokens if context is too long)
            context = input_seq[:min(20, len(input_seq))]
            max_len = max(50, len(input_seq) + 20)  # Generate at least as long as input + some extra
            
            generated_seq = sample_sequence(
                sequence_generator,
                context=context,
                max_new_tokens=max_len,
                temperature=1.0,
                top_k=0,
                top_p=0.0
            )
            
            # Ensure minimum length
            if len(generated_seq) < len(input_seq):
                generated_seq += input_seq[len(generated_seq):]
            
            generated_sequences.append(generated_seq)
            logger.debug(f"[Generate] Generated sequence {i+1}/{num_sequences}: {generated_seq[:50]}...")
        
        logger.info(f"[Generate] Successfully generated {len(generated_sequences)} sequences")
        return generated_sequences
    except Exception as e:
        logger.error(f"[Generate] Sequence generation error: {e}", exc_info=True)
        # Fallback to random generation
        generated = []
        for _ in range(num_sequences):
            seq = list(input_seq)
            for i in range(random.randint(1, 3)):
                if len(seq) > 0:
                    pos = random.randint(0, len(seq) - 1)
                    seq[pos] = random.choice(AMINO_ACIDS)
            generated.append(''.join(seq))
        return generated

def calculate_sequence_probabilities(input_seq: str, generated_seqs: list) -> list:
    """
    Calculate similarity probabilities for generated sequences.
    Uses only Levenshtein and Hamming similarities.
    Returns top 10 sequences sorted by average probability.
    """
    results = []
    
    for seq in generated_seqs:
        # Calculate only Levenshtein and Hamming similarities
        lev_sim = levenshtein_similarity(input_seq, seq)
        ham_sim = hamming_similarity(input_seq, seq)
        
        # Calculate average probability (only Levenshtein + Hamming)
        avg_prob = (lev_sim + ham_sim) / 2
        
        results.append({
            'sequence': seq,
            'average_probability': avg_prob,
            'levenshtein': lev_sim,
            'hamming': ham_sim
        })
    
    # Sort by average probability (descending)
    results.sort(key=lambda x: x['average_probability'], reverse=True)
    
    return results[:10]  # Return top 10

def generate_smiles(input_seq: str, gen_max_len: int = 256, temperature: float = 1.5, 
                     top_k: int = 50, top_p: float = 0.9, repetition_penalty: float = 1.15) -> str:
    """Generate SMILES structure from protein sequence using Fusion Model (matches working code exactly)"""
    # Try fusion model first - EXACT match to working inference code
    if fusion_model is not None and fusion_tokenizer is not None and progen_model is not None and molt5_fusion is not None:
        try:
            logger.info(f"[SMILES-Fusion] Generating SMILES for sequence (length {len(input_seq)})")
            
            # Encode sequence EXACTLY like working code (MAX_SEQ_LEN = 128)
            MAX_SEQ_LEN_FUSION = 128
            seq_ids = torch.tensor([fusion_tokenizer.encode(input_seq, True, MAX_SEQ_LEN_FUSION)], device=device)
            
            # Base sampling parameters
            base_temp = max(0.5, min(temperature, 2.5))
            base_top_k = max(0, top_k)
            base_top_p = max(0.0, min(top_p, 1.0))
            base_rep_pen = max(1.0, min(repetition_penalty, 2.0))
            
            # Generate with retry logic - EXACT match to working code
            max_retries = 5
            smiles = None
            valid = None
            
            with torch.no_grad():
                # EXACT inference flow from working code
                seq_h = progen_model.embed(seq_ids)
                pooled = seq_h.mean(dim=1)
                enc_emb = fusion_model.proj(pooled).unsqueeze(1)
                
                for attempt in range(max_retries):
                    # Adjust parameters for retries (exactly like working code)
                    if attempt == 0:
                        temp = base_temp
                        tk = base_top_k
                        tp = base_top_p
                        rep = base_rep_pen
                    else:
                        temp = max(0.8, base_temp - 0.2 * attempt)
                        tk = min(base_top_k + 10 * attempt, 100)
                        tp = min(base_top_p + 0.05 * attempt, 0.95)
                        rep = max(1.05, base_rep_pen - 0.05 * attempt)
                    
                    # Generate SMILES - EXACT call from working code
                    out_ids = molt5_fusion.generate_from_enc(
                        enc_emb, fusion_tokenizer, max_len=gen_max_len,
                        temperature=temp, top_k=tk, top_p=tp, repetition_penalty=rep
                    )
                    
                    # Decode - EXACT from working code
                    smiles = fusion_tokenizer.decode(out_ids[0].cpu().tolist())
                    
                    # Validate - check for invalid characters (matches working code)
                    invalid_chars = ['<unk>', '[', ']', '*', 'Ml', 'RW', 'VW', 'NCW']
                    has_invalid = any(char in smiles for char in invalid_chars)
                    
                    if not has_invalid and smiles and len(smiles) >= 3:
                        # Valid SMILES found
                        valid = True
                        logger.info(f"[SMILES-Fusion] ✅ Generated valid SMILES: {smiles[:50]}...")
                        return smiles
                    elif attempt < max_retries - 1:
                        # Invalid, retry
                        logger.warning(f"[SMILES-Fusion] Invalid SMILES on attempt {attempt + 1}, retrying...")
                        continue
                    else:
                        # Last attempt
                        valid = False
                        logger.warning(f"[SMILES-Fusion] Generated SMILES after {max_retries} attempts: {smiles}")
                        return smiles if smiles else "C[C@H](N)C(=O)O"
            
            return smiles if smiles else "C[C@H](N)C(=O)O"
            
        except Exception as e:
            logger.error(f"[SMILES-Fusion] Error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            # Fall through to legacy model
    
    # Fallback to legacy MiniT5 model
    if protein_to_smile is not None and smiles_tokenizer is not None:
        logger.info("[SMILES] Using legacy MiniT5 model for SMILES generation")
        try:
            if isinstance(protein_to_smile, MiniT5):
                description = f"A protein sequence: {input_seq}"
                input_ids = smiles_tokenizer.encode(description, add_special=True, max_len=256)
                max_src_len = 256
                if len(input_ids) > max_src_len:
                    input_ids = input_ids[:max_src_len-1] + [smiles_tokenizer.stoi[smiles_tokenizer.EOS]]
                else:
                    pad_id = smiles_tokenizer.stoi[smiles_tokenizer.PAD]
                    input_ids = input_ids + [pad_id] * (max_src_len - len(input_ids))
                
                src_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)
                
                with torch.no_grad():
                    output_ids = protein_to_smile.generate(src_tensor, smiles_tokenizer, max_len=128)
                    generated_ids = output_ids[0].cpu().tolist()
                    smiles = smiles_tokenizer.decode(generated_ids)
                    valid_smiles_chars = set("CNOSFBrIPc1234567890[]()=#+-nopsABCDEFGHIKLMNPQRSTVWYabcdefghijklmnopqrstuvwxyz")
                    cleaned_smiles = "".join(c for c in smiles if c in valid_smiles_chars)
                    
                    if cleaned_smiles and len(cleaned_smiles) >= 3:
                        return cleaned_smiles
                    return "C[C@H](N)C(=O)O"
        except Exception as e:
            logger.error(f"[SMILES-Legacy] Error: {e}")
    
    logger.warning("[SMILES] No SMILES model available, returning placeholder")
    return "C[C@H](N)C(=O)O"  # Placeholder

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'service': 'authentication+prediction',
        'models_loaded': all([custom_protein_lm is not None, mlp_model is not None, label_encoder is not None, pca_model is not None])
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json() or {}
        seq = data.get('sequence', '')
        
        # Log input for debugging
        logger.info(f"[Predict] Input sequence length: {len(seq)}")
        
        # Preprocess (center-crop to WINDOW_LEN if longer, pad if shorter)
        proc = preprocess_sequence(seq)
        logger.info(f"[Predict] Processed sequence length: {len(proc)}, sequence: {proc[:50]}...")
        
        # Generate embedding
        emb = embed_sequence(proc)
        logger.info(f"[Predict] Embedding shape before PCA: {emb.shape}")
        
        # Apply PCA transformation
        if pca_model is not None:
            emb = pca_model.transform(emb)
            logger.info(f"[Predict] Embedding shape after PCA: {emb.shape}")
        else:
            logger.warning("[Predict] PCA model is None, skipping PCA transformation")
        
        # Convert to tensor and predict
        X = torch.tensor(emb, dtype=torch.float32).to(device)
        logits = mlp_model(X)
        probs = F.softmax(logits, dim=1)
        idx = torch.argmax(probs, dim=1).item()
        conf = probs[0, idx].item()
        pred = label_encoder.classes_[idx]
        
        # Log prediction details
        logger.info(f"[Predict] Predicted class: {pred}, Confidence: {conf:.4f}")
        logger.info(f"[Predict] All probabilities: {[f'{label_encoder.classes_[i]}: {probs[0, i].item():.4f}' for i in range(len(label_encoder.classes_))]}")
        
        prob_map = {label_encoder.classes_[i]: probs[0, i].item() for i in range(len(label_encoder.classes_))}
        return jsonify({
            'success': True,
            'result': {
                'prediction': pred,
                'confidence': conf,
                'probabilities': prob_map
            },
            'processed_sequence': proc,
            'device_type': device_type,
            'device': str(device)
        })
    except ValueError as ve:
        logger.error(f"[Predict] ValueError: {ve}")
        return jsonify({'success': False, 'error': str(ve)}), 400
    except Exception as e:
        logger.error(f"[Predict] Prediction error: {e}", exc_info=True)
        return jsonify({'success': False, 'error': 'Prediction failed'}), 500

@app.route('/classes', methods=['GET'])
def classes():
    if label_encoder is None:
        return jsonify({'success': False, 'error': 'Model not loaded'}), 500
    return jsonify({'success': True, 'classes': label_encoder.classes_.tolist()})

@app.route('/generate-sequences', methods=['POST'])
def generate_sequences_endpoint():
    """
    Generate new protein sequences and calculate similarity probabilities.
    Returns top 10 sequences based on average of Levenshtein and Hamming similarities.
    """
    try:
        data = request.get_json() or {}
        input_sequence = data.get('sequence', '').strip().upper()
        
        # Validation
        if not input_sequence:
            return jsonify({'success': False, 'error': 'Sequence is required'}), 400
        
        valid_amino_acids = set("ACDEFGHIKLMNPQRSTVWY")
        if not all(c in valid_amino_acids for c in input_sequence):
            return jsonify({'success': False, 'error': 'Invalid amino acid sequence'}), 400
        
        if len(input_sequence) < 10:
            return jsonify({'success': False, 'error': 'Sequence too short (minimum 10 amino acids)'}), 400
        
        # Generate more sequences to ensure we have good diversity for top 10 selection
        # Generate 20-30 sequences to have enough candidates for top 10
        num_to_generate = 30
        logger.info(f"[Generate] Generating {num_to_generate} sequences for selection...")
        generated_seqs = generate_sequences(input_sequence, num_sequences=num_to_generate)
        
        # Calculate probabilities (only Levenshtein + Hamming)
        results = calculate_sequence_probabilities(input_sequence, generated_seqs)
        
        logger.info(f"[Generate] Selected top {len(results)} sequences from {len(generated_seqs)} generated")
        
        return jsonify({
            'success': True,
            'input_sequence': input_sequence,
            'total_generated': len(generated_seqs),
            'top_sequences': results,
            'metrics_used': ['levenshtein', 'hamming'],
            'ranking': 'sorted by average of levenshtein and hamming similarities',
            'device_type': device_type,
            'device': str(device)
        })
        
    except Exception as e:
        logger.error(f"[Generate] Sequence generation error: {e}", exc_info=True)
        return jsonify({'success': False, 'error': 'Sequence generation failed'}), 500

# ======================
# ALPHAFOLD2 & DOCKING ENDPOINTS (ngrok integration)
# ======================

# Ngrok URLs - should be set via environment variables or config
# The backend is pre-configured with the current ngrok URL
# Only update if the Colab notebook generates a NEW ngrok URL
ALPHAFOLD2_NGROK_URL = os.environ.get('ALPHAFOLD2_NGROK_URL', 'https://muzzleloading-pedro-originally.ngrok-free.dev')
DOCKING_NGROK_URL = os.environ.get('DOCKING_NGROK_URL', 'http://localhost:8001')

def check_ngrok_health(ngrok_url: str, retries: int = 2) -> dict:
    """
    Check if ngrok endpoint is online and accessible
    Returns dict with status and diagnostic info
    """
    import time
    health_endpoint = f"{ngrok_url.rstrip('/')}/health"
    headers = {
        'ngrok-skip-browser-warning': 'true',
        'User-Agent': 'Mozilla/5.0 (compatible; Flask-Backend/1.0)'
    }
    
    last_error = None
    for attempt in range(retries + 1):
        try:
            response = requests.get(
                health_endpoint, 
                headers=headers, 
                timeout=15,  # Increased timeout
                allow_redirects=True,
                verify=True  # Verify SSL certificates
            )
            
            if response.status_code == 200:
                try:
                    result = response.json()
                    return {'status': 'online', 'response': result}
                except ValueError:
                    return {'status': 'online', 'response': {'raw': response.text[:100]}}
            elif response.headers.get('Content-Type', '').startswith('text/html'):
                return {
                    'status': 'offline',
                    'error': 'Ngrok returned HTML error page (endpoint may be offline)',
                    'status_code': response.status_code
                }
            else:
                return {
                    'status': 'error',
                    'error': f'Unexpected status code: {response.status_code}',
                    'status_code': response.status_code
                }
        except requests.exceptions.SSLError as e:
            last_error = f'SSL error: {str(e)}'
            if attempt < retries:
                time.sleep(1)  # Wait before retry
            continue
        except requests.exceptions.ConnectionError as e:
            error_str = str(e)
            # Check if it's a DNS resolution error
            if 'getaddrinfo failed' in error_str or 'Name or service not known' in error_str:
                last_error = f'DNS resolution failed: Cannot resolve hostname. This may be a temporary network issue. Error: {error_str[:200]}'
            else:
                last_error = f'Connection error: {error_str[:200]}'
            if attempt < retries:
                time.sleep(2)  # Wait before retry (longer for DNS issues)
            continue
        except requests.exceptions.Timeout as e:
            last_error = f'Request timeout: {str(e)}'
            if attempt < retries:
                time.sleep(1)
            continue
        except requests.exceptions.RequestException as e:
            last_error = f'Request failed: {str(e)[:200]}'
            if attempt < retries:
                time.sleep(1)
            continue
    
    return {
        'status': 'offline',
        'error': last_error or 'Connection failed after retries'
    }

@app.route('/alphafold2/health', methods=['GET'])
def alphafold2_health_check():
    """
    Check if AlphaFold2 ngrok endpoint is online
    Useful for diagnostics
    """
    health_status = check_ngrok_health(ALPHAFOLD2_NGROK_URL)
    return jsonify({
        'success': health_status['status'] == 'online',
        'ngrok_url': ALPHAFOLD2_NGROK_URL,
        'status': health_status['status'],
        'details': health_status
    }), 200 if health_status['status'] == 'online' else 503

@app.route('/alphafold2/predict', methods=['POST'])
def alphafold2_predict():
    """
    Proxy endpoint for AlphaFold2 structure prediction
    Forwards request to ngrok URL running AlphaFold2 notebook
    """
    try:
        data = request.get_json() or {}
        sequence = data.get('sequence', '').strip().upper()
        
        # Validation
        if not sequence:
            return jsonify({'success': False, 'error': 'Protein sequence is required'}), 400
        
        valid_amino_acids = set("ACDEFGHIKLMNPQRSTVWY")
        if not all(c in valid_amino_acids for c in sequence):
            return jsonify({'success': False, 'error': 'Invalid amino acid sequence'}), 400
        
        if len(sequence) < 10:
            return jsonify({'success': False, 'error': 'Sequence too short (minimum 10 amino acids)'}), 400
        
        logger.info(f"[AlphaFold2] Requesting structure prediction for sequence (length: {len(sequence)})")
        
        # Quick health check before making the request (non-blocking - warn but don't fail)
        health_status = check_ngrok_health(ALPHAFOLD2_NGROK_URL)
        if health_status['status'] != 'online':
            logger.warning(f"[AlphaFold2] Health check failed: {health_status.get('error', 'Unknown error')}")
            logger.info(f"[AlphaFold2] Proceeding with prediction request anyway (health check may have false negatives)")
            # Don't block the request - DNS issues might be temporary but the actual request might work
        
        # Forward request to ngrok endpoint
        # The notebook expects: /predict endpoint
        ngrok_endpoint = f"{ALPHAFOLD2_NGROK_URL}/predict"
        
        # Remove trailing slash if present
        if ngrok_endpoint.endswith('/'):
            ngrok_endpoint = ngrok_endpoint[:-1]
        if not ngrok_endpoint.endswith('/predict'):
            ngrok_endpoint = ngrok_endpoint.rstrip('/') + '/predict'
        
        logger.info(f"[AlphaFold2] Forwarding request to: {ngrok_endpoint}")
        
        try:
            # Headers to bypass ngrok free tier warning page and CORS
            headers = {
                'Content-Type': 'application/json',
                'ngrok-skip-browser-warning': 'true',
                'User-Agent': 'Mozilla/5.0 (compatible; Flask-Backend/1.0)'
            }
            
            # Retry logic for connection issues
            import time
            max_retries = 3
            last_exception = None
            
            for attempt in range(max_retries):
                try:
                    response = requests.post(
                        ngrok_endpoint,
                        json={'sequence': sequence},
                        timeout=900,  # AlphaFold2 can take 5-15 minutes, increase timeout
                        headers=headers,
                        allow_redirects=True,
                        verify=True
                    )
                    break  # Success, exit retry loop
                except requests.exceptions.ConnectionError as e:
                    error_str = str(e)
                    last_exception = e
                    if 'getaddrinfo failed' in error_str:
                        logger.warning(f"[AlphaFold2] DNS resolution failed (attempt {attempt + 1}/{max_retries}): {error_str[:200]}")
                        if attempt < max_retries - 1:
                            wait_time = (attempt + 1) * 2  # Exponential backoff: 2s, 4s, 6s
                            logger.info(f"[AlphaFold2] Retrying in {wait_time} seconds...")
                            time.sleep(wait_time)
                        else:
                            # Last attempt failed
                            logger.error(f"[AlphaFold2] DNS resolution failed after {max_retries} attempts")
                            return jsonify({
                                'success': False,
                                'error': f'Cannot resolve ngrok domain. This may be a DNS or network issue on your local machine. Please check your internet connection. If the issue persists, try: 1) Restarting your router/network, 2) Using a different DNS server (e.g., 8.8.8.8), 3) Checking if the ngrok URL is accessible from your browser: {ALPHAFOLD2_NGROK_URL}/health'
                            }), 503
                    else:
                        # Other connection errors
                        logger.warning(f"[AlphaFold2] Connection error (attempt {attempt + 1}/{max_retries}): {error_str[:200]}")
                        if attempt < max_retries - 1:
                            time.sleep(2)
                        else:
                            raise  # Re-raise if not DNS issue
                except requests.exceptions.RequestException as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        logger.warning(f"[AlphaFold2] Request error (attempt {attempt + 1}/{max_retries}), retrying...")
                        time.sleep(2)
                    else:
                        raise  # Re-raise on last attempt
            
            # If we got here without breaking, the last attempt must have failed
            if last_exception:
                raise last_exception
            
            # Check if response is HTML error page (ngrok offline page)
            if response.status_code == 404 or (response.headers.get('Content-Type', '').startswith('text/html')):
                response_text = response.text[:500]  # Get first 500 chars for logging
                logger.error(f"[AlphaFold2] Ngrok endpoint returned HTML error page (status {response.status_code})")
                logger.error(f"[AlphaFold2] Response preview: {response_text}")
                
                # Check for specific ngrok error codes
                if 'ERR_NGROK_3200' in response_text or 'offline' in response_text.lower():
                    return jsonify({
                        'success': False,
                        'error': 'Ngrok endpoint is offline. Please ensure the Colab notebook is running and ngrok tunnel is active. The ngrok URL may have changed - check the Colab notebook output for the current URL.'
                    }), 503
                
                return jsonify({
                    'success': False,
                    'error': f'Ngrok endpoint returned error page (status {response.status_code}). The endpoint may be offline or the URL may be incorrect.'
                }), 503
            
            if response.status_code == 200:
                try:
                    result = response.json()
                except ValueError as e:
                    logger.error(f"[AlphaFold2] Response is not valid JSON: {e}")
                    logger.error(f"[AlphaFold2] Response content: {response.text[:500]}")
                    return jsonify({
                        'success': False,
                        'error': 'Invalid response format from AlphaFold2 service. The service may not be running correctly.'
                    }), 500
                
                # Check if result contains error from notebook
                if 'error' in result:
                    logger.error(f"[AlphaFold2] Notebook returned error: {result['error']}")
                    return jsonify({
                        'success': False,
                        'error': result['error']
                    }), 500
                
                logger.info(f"[AlphaFold2] Structure prediction completed successfully")
                logger.info(f"[AlphaFold2] pLDDT score: {result.get('plddt_score', 'N/A')}")
                
                # Return in format expected by frontend
                return jsonify({
                    'success': True,
                    'sequence': sequence,
                    'result': {
                        'pdb_content': result.get('pdb_content', ''),
                        'plddt_score': result.get('plddt_score', 85.0),
                        'confidence_scores': result.get('confidence_scores', {}),
                        'jobname': result.get('jobname', '')
                    },
                    'device_type': device_type,
                    'device': str(device),
                    'note': 'AlphaFold2 computation runs on external service (Colab), device info shows Flask backend device'
                })
            else:
                logger.error(f"[AlphaFold2] Ngrok endpoint returned status {response.status_code}")
                logger.error(f"[AlphaFold2] Response: {response.text[:500]}")
                return jsonify({
                    'success': False,
                    'error': f'AlphaFold2 service error: HTTP {response.status_code}. Please check if the Colab notebook is running.'
                }), 500
                
        except requests.exceptions.RequestException as e:
            logger.error(f"[AlphaFold2] Connection error to ngrok: {e}")
            return jsonify({
                'success': False,
                'error': 'Cannot connect to AlphaFold2 service. Please ensure the Colab notebook is running with ngrok.'
            }), 503
            
    except Exception as e:
        logger.error(f"[AlphaFold2] Error: {e}", exc_info=True)
        return jsonify({'success': False, 'error': 'AlphaFold2 prediction failed'}), 500

@app.route('/docking/run', methods=['POST'])
def docking_run():
    """
    Local molecular docking endpoint (runs on Flask backend)
    Uses RDKit, OpenBabel, and AutoDock Vina - all CPU-based, no GPU needed
    Requires: SMILES string and protein PDB file content
    """
    try:
        data = request.get_json() or {}
        smiles = data.get('smiles', '').strip()
        protein_sequence = data.get('protein_sequence', '').strip().upper()
        protein_pdb_content = data.get('protein_pdb', '').strip()  # Optional: PDB file content
        
        # Validation
        if not smiles:
            return jsonify({'success': False, 'error': 'SMILES string is required'}), 400
        
        if not protein_pdb_content and not protein_sequence:
            return jsonify({'success': False, 'error': 'Either protein sequence or PDB content is required'}), 400
        
        # If only sequence provided, need to get PDB from AlphaFold2 first
        if not protein_pdb_content and protein_sequence:
            return jsonify({
                'success': False,
                'error': 'Please provide protein PDB content (from AlphaFold2 prediction) or run AlphaFold2 first'
            }), 400
        
        logger.info(f"[Docking] Running local docking for SMILES (length: {len(smiles)}) and protein PDB (length: {len(protein_pdb_content)})")
        
        # Try local docking first
        try:
            from local_docking import run_local_docking
            result = run_local_docking(smiles, protein_pdb_content)
            logger.info(f"[Docking] ✅ Local docking completed successfully")
            return jsonify({
                'success': True,
                'smiles': smiles,
                'result': result,
                'device_type': device_type,
                'device': str(device),
                'note': 'Docking computation runs locally (CPU-based, no GPU needed)'
            })
        except ImportError as e:
            error_msg = str(e)
            if "rdkit" in error_msg.lower() or "vina" in error_msg.lower():
                logger.error(f"[Docking] Missing dependencies: {error_msg}")
                return jsonify({
                    'success': False,
                    'error': 'Docking dependencies not installed. Install with: pip install rdkit (Python), brew install open-babel vina (Mac). See INSTALL_DOCKING_DEPS.md for details.'
                }), 500
            else:
                raise
        except Exception as e:
            logger.error(f"[Docking] Local docking error: {e}", exc_info=True)
            # Fallback to Colab if configured and local fails
            if DOCKING_NGROK_URL and 'localhost' not in DOCKING_NGROK_URL:
                logger.info("[Docking] Falling back to Colab docking service...")
                try:
                    ngrok_endpoint = f"{DOCKING_NGROK_URL}/dock"
                    payload = {
                        'smiles': smiles,
                        'protein_pdb': protein_pdb_content
                    }
                    response = requests.post(
                        ngrok_endpoint,
                        json=payload,
                        timeout=180,
                        headers={'Content-Type': 'application/json'}
                    )
                    if response.status_code == 200:
                        result = response.json()
                        logger.info(f"[Docking] ✅ Colab docking completed")
                        return jsonify({
                            'success': True,
                            'smiles': smiles,
                            'result': result,
                            'device_type': device_type,
                            'device': str(device),
                            'note': 'Docking computation runs on external service (Colab fallback)'
                        })
                except Exception as fallback_error:
                    logger.error(f"[Docking] Colab fallback also failed: {fallback_error}")
            
            return jsonify({
                'success': False,
                'error': f'Docking failed: {str(e)}. Ensure dependencies are installed: pip install rdkit-pypi vina'
            }), 500
            
    except Exception as e:
        logger.error(f"[Docking] Error: {e}", exc_info=True)
        return jsonify({'success': False, 'error': f'Docking failed: {str(e)}'}), 500

@app.route('/pipeline/alphafold2-to-docking', methods=['POST'])
def combined_pipeline():
    """
    Complete pipeline: Protein Sequence → AlphaFold2 → Rank 1 PDB → Docking
    
    Request: {"protein_sequence": "...", "smiles": "..."}
    Response: Combined AlphaFold2 + Docking results
    """
    try:
        data = request.get_json() or {}
        protein_sequence = data.get('protein_sequence', '').strip().upper()
        smiles = data.get('smiles', '').strip()
        
        # Validation
        if not protein_sequence:
            return jsonify({'success': False, 'error': 'Protein sequence is required'}), 400
        
        if not smiles:
            return jsonify({'success': False, 'error': 'SMILES string is required'}), 400
        
        valid_amino_acids = set("ACDEFGHIKLMNPQRSTVWY")
        if not all(c in valid_amino_acids for c in protein_sequence):
            return jsonify({'success': False, 'error': 'Invalid amino acid sequence'}), 400
        
        if len(protein_sequence) < 10:
            return jsonify({'success': False, 'error': 'Sequence too short (minimum 10 amino acids)'}), 400
        
        logger.info(f"[Pipeline] Starting combined pipeline:")
        logger.info(f"  - Protein sequence length: {len(protein_sequence)}")
        logger.info(f"  - SMILES: {smiles[:50]}...")
        
        # Step 1: Run AlphaFold2 prediction
        logger.info("[Pipeline] Step 1/2: Running AlphaFold2 prediction...")
        alphafold2_result = None
        try:
            health_status = check_ngrok_health(ALPHAFOLD2_NGROK_URL)
            if health_status['status'] != 'online':
                return jsonify({
                    'success': False,
                    'error': 'AlphaFold2 service is offline. Please ensure the Colab notebook is running.',
                    'step': 'alphafold2_health_check'
                }), 503
            
            ngrok_endpoint = f"{ALPHAFOLD2_NGROK_URL.rstrip('/')}/predict"
            headers = {
                'Content-Type': 'application/json',
                'ngrok-skip-browser-warning': 'true',
                'User-Agent': 'Mozilla/5.0 (compatible; Flask-Backend/1.0)'
            }
            
            response = requests.post(
                ngrok_endpoint,
                json={'sequence': protein_sequence},
                timeout=900,  # 15 minutes for AlphaFold2
                headers=headers,
                allow_redirects=True,
                verify=True
            )
            
            if response.status_code == 200:
                result = response.json()
                if 'error' in result:
                    return jsonify({
                        'success': False,
                        'error': f'AlphaFold2 error: {result["error"]}',
                        'step': 'alphafold2_prediction'
                    }), 500
                
                alphafold2_result = result
                logger.info(f"[Pipeline] ✅ AlphaFold2 completed! pLDDT: {result.get('plddt_score', 'N/A')}")
            else:
                return jsonify({
                    'success': False,
                    'error': f'AlphaFold2 service error: HTTP {response.status_code}',
                    'step': 'alphafold2_prediction'
                }), 500
                
        except requests.exceptions.RequestException as e:
            logger.error(f"[Pipeline] AlphaFold2 connection error: {e}")
            return jsonify({
                'success': False,
                'error': f'Cannot connect to AlphaFold2 service: {str(e)}',
                'step': 'alphafold2_connection'
            }), 503
        
        # Step 2: Extract rank 1 PDB (should already be extracted by AlphaFold2 service)
        pdb_content = alphafold2_result.get('pdb_content', '')
        if not pdb_content:
            return jsonify({
                'success': False,
                'error': 'No PDB content received from AlphaFold2',
                'step': 'pdb_extraction',
                'alphafold2': alphafold2_result
            }), 500
        
        logger.info(f"[Pipeline] ✅ Rank 1 PDB extracted ({len(pdb_content)} characters)")
        
        # Step 3: Run Docking with rank 1 PDB (locally)
        logger.info("[Pipeline] Step 2/2: Running molecular docking locally...")
        docking_result = None
        try:
            from local_docking import run_local_docking
            docking_result = run_local_docking(smiles, pdb_content)
            logger.info(f"[Pipeline] ✅ Local docking completed! Best affinity: {docking_result.get('best_affinity', 'N/A')} kcal/mol")
        except ImportError as e:
            error_msg = str(e)
            logger.error(f"[Pipeline] Docking dependencies missing: {error_msg}")
            return jsonify({
                'success': False,
                'error': f'Docking dependencies not installed: {error_msg}. Install with: pip install rdkit-pypi vina',
                'step': 'docking_dependencies',
                'alphafold2': alphafold2_result
            }), 500
        except Exception as e:
            logger.error(f"[Pipeline] Docking error: {e}", exc_info=True)
            # Fallback to Colab if available
            if DOCKING_NGROK_URL and 'localhost' not in DOCKING_NGROK_URL:
                logger.info("[Pipeline] Falling back to Colab docking...")
                try:
                    health_status = check_ngrok_health(DOCKING_NGROK_URL)
                    if health_status['status'] == 'online':
                        ngrok_endpoint = f"{DOCKING_NGROK_URL.rstrip('/')}/dock"
                        response = requests.post(
                            ngrok_endpoint,
                            json={'smiles': smiles, 'protein_pdb': pdb_content},
                            timeout=180,
                            headers={'Content-Type': 'application/json'}
                        )
                        if response.status_code == 200:
                            docking_result = response.json()
                            logger.info(f"[Pipeline] ✅ Colab docking completed (fallback)")
                        else:
                            raise Exception(f"Colab docking failed: {response.status_code}")
                    else:
                        raise Exception("Colab docking service offline")
                except Exception as fallback_error:
                    logger.error(f"[Pipeline] Colab fallback also failed: {fallback_error}")
            
            return jsonify({
                'success': False,
                'error': f'Docking failed: {str(e)}',
                'step': 'docking_prediction',
                'alphafold2': alphafold2_result
            }), 500
        
        # Step 4: Return combined results
        logger.info("[Pipeline] ✅ Pipeline completed successfully!")
        return jsonify({
            'success': True,
            'protein_sequence': protein_sequence,
            'smiles': smiles,
            'alphafold2': {
                'pdb_content': alphafold2_result.get('pdb_content', ''),
                'plddt_score': alphafold2_result.get('plddt_score', 0),
                'jobname': alphafold2_result.get('jobname', ''),
                'rank1_file': alphafold2_result.get('rank1_file', 'unknown')
            },
            'docking': {
                **docking_result,
                'visualization_data': {
                    'protein_pdb': pdb_content,  # Include protein PDB for 3D visualization
                    'ligand_pdbqt': docking_result.get('pdbqt_content', '')
                }
            },
            'pipeline_status': 'complete',
            'device_type': device_type,
            'device': str(device)
        })
        
    except Exception as e:
        logger.error(f"[Pipeline] Error: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': f'Pipeline failed: {str(e)}'
        }), 500

@app.route('/generate-smiles', methods=['POST'])
def generate_smiles_endpoint():
    """Generate SMILES structure from protein sequence - proxies to FastAPI service"""
    try:
        data = request.get_json() or {}
        input_sequence = data.get('sequence', '').strip().upper()
        
        # Optional generation parameters
        max_length = int(data.get('max_length', data.get('gen_max_len', 256)))
        temperature = float(data.get('temperature', 1.5))
        top_k = int(data.get('top_k', 50))
        top_p = float(data.get('top_p', 0.9))
        repetition_penalty = float(data.get('repetition_penalty', 1.15))
        
        # Validation
        if not input_sequence:
            return jsonify({'success': False, 'error': 'Sequence is required'}), 400
        
        valid_amino_acids = set("ACDEFGHIKLMNPQRSTVWY")
        if not all(c in valid_amino_acids for c in input_sequence):
            return jsonify({'success': False, 'error': 'Invalid amino acid sequence'}), 400
        
        if len(input_sequence) < 10:
            return jsonify({'success': False, 'error': 'Sequence too short (minimum 10 amino acids)'}), 400
        
        # Call FastAPI service at http://localhost:8000/predict
        fastapi_url = os.environ.get('FASTAPI_URL', 'http://localhost:8000/predict')
        try:
            response = requests.post(
                fastapi_url,
                json={
                    'sequence': input_sequence,
                    'max_length': max_length,
                    'temperature': temperature,
                    'top_k': top_k,
                    'top_p': top_p,
                    'repetition_penalty': repetition_penalty
                },
                timeout=120  # 2 minute timeout for model inference
            )
            response.raise_for_status()
            result = response.json()
            
            # Return in format compatible with frontend
            return jsonify({
                'success': True,
                'input_sequence': result.get('sequence', input_sequence),
                'smiles': result.get('smiles', ''),
                'isValid': result.get('isValid'),
                'model_type': 'fusion',
                'deviceType': result.get('deviceType'),
                'sampling': result.get('sampling', {})
            })
            
        except requests.exceptions.ConnectionError:
            logger.error("FastAPI service not available at http://localhost:8000")
            return jsonify({
                'success': False, 
                'error': 'SMILES generation service unavailable. Please start the Python API service (python-api/api.py) on port 8000.'
            }), 503
        except requests.exceptions.Timeout:
            logger.error("FastAPI service timeout")
            return jsonify({
                'success': False, 
                'error': 'SMILES generation timed out. The model may be processing a very long sequence.'
            }), 504
        except requests.exceptions.RequestException as e:
            logger.error(f"FastAPI service error: {e}")
            return jsonify({
                'success': False, 
                'error': f'SMILES generation service error: {str(e)}'
            }), 502
        
    except Exception as e:
        logger.error(f"SMILES generation error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({'success': False, 'error': 'SMILES generation failed'}), 500

# ======================
# AUTH ENDPOINTS
# ======================

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'authentication',
        'timestamp': datetime.utcnow().isoformat()
    })

@app.route('/api/register', methods=['POST'])
def register():
    """Register a new user"""
    try:
        data = request.get_json()
        username = data.get('username', '').strip()
        password = data.get('password', '').strip()
        
        # Validation
        if not username or not password:
            return jsonify({
                'success': False,
                'error': 'Username and password are required'
            }), 400
        
        if len(username) < 3:
            return jsonify({
                'success': False,
                'error': 'Username must be at least 3 characters long'
            }), 400
        
        if len(password) < 6:
            return jsonify({
                'success': False,
                'error': 'Password must be at least 6 characters long'
            }), 400
        
        # Check if user already exists
        existing_user = users_collection.find_one({'username': username})
        if existing_user:
            return jsonify({
                'success': False,
                'error': 'Username already exists'
            }), 400
        
        # Create new user
        hashed_password = hash_password(password)
        user_data = {
            'username': username,
            'password': hashed_password,
            'created_at': datetime.utcnow(),
            'last_login': None
        }
        
        try:
            result = users_collection.insert_one(user_data)
        except DuplicateKeyError:
            return jsonify({'success': False, 'error': 'Username already exists'}), 400
        
        # Create session
        session['user_id'] = str(result.inserted_id)
        session['username'] = username
        session.permanent = True
        
        logger.info(f"New user registered: {username}")
        
        return jsonify({
            'success': True,
            'message': 'User registered successfully',
            'user': {
                'id': str(result.inserted_id),
                'username': username
            }
        })
        
    except Exception as e:
        logger.error(f"Registration error: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Registration failed'
        }), 500

@app.route('/api/login', methods=['POST'])
def login():
    """Login user"""
    try:
        data = request.get_json()
        username = data.get('username', '').strip()
        password = data.get('password', '').strip()
        
        # Validation
        if not username or not password:
            return jsonify({
                'success': False,
                'error': 'Username and password are required'
            }), 400
        
        # Find user by username or mobile (user can enter either)
        user = users_collection.find_one({'$or': [
            {'username': username},
            {'mobile': username}
        ]})
        if not user:
            return jsonify({
                'success': False,
                'error': 'Invalid username or password'
            }), 401
        
        # Verify password
        if not verify_password(password, user['password']):
            return jsonify({
                'success': False,
                'error': 'Invalid username or password'
            }), 401
        
        # Update last login
        users_collection.update_one(
            {'_id': user['_id']},
            {'$set': {'last_login': datetime.utcnow()}}
        )
        
        # Create session
        session['user_id'] = str(user['_id'])
        session['username'] = username
        session.permanent = True
        
        logger.info(f"User logged in: {username}")
        
        return jsonify({
            'success': True,
            'message': 'Login successful',
            'user': {
                'id': str(user['_id']),
                'username': username
            }
        })
        
    except Exception as e:
        logger.error(f"Login error: {str(e)}")
        return jsonify({
            'success': False,
            'message': 'Login failed'
        }), 500

@app.route('/api/request-otp-mobile', methods=['POST'])
def request_otp_mobile():
    try:
        data = request.get_json() or {}
        mobile = data.get('mobile', '').strip()
        username = data.get('username', '').strip()
        if not mobile or not username:
            return jsonify({'success': False, 'error': 'Username and mobile are required'}), 400
        phone = normalize_phone(mobile)
        code = generate_otp()
        expires = datetime.utcnow() + timedelta(minutes=10)
        otps_collection.delete_many({'phone': phone})
        otps_collection.insert_one({'phone': phone, 'code': code, 'expires_at': expires, 'verified': False, 'username': username})
        if not send_sms_otp(phone, code):
            return jsonify({'success': False, 'error': 'Failed to send OTP'}), 500
        return jsonify({'success': True})
    except Exception as e:
        logger.error(f"request_otp_mobile error: {e}")
        return jsonify({'success': False, 'error': 'request_otp_mobile_failed'}), 500

@app.route('/api/verify-otp-mobile', methods=['POST'])
def verify_otp_mobile():
    try:
        data = request.get_json() or {}
        mobile = data.get('mobile', '').strip()
        otp = data.get('otp', '').strip()
        phone = normalize_phone(mobile)
        rec = otps_collection.find_one({'phone': phone})
        if not rec:
            return jsonify({'success': False, 'error': 'otp_not_found'}), 400
        if rec['expires_at'] < datetime.utcnow():
            return jsonify({'success': False, 'error': 'otp_expired'}), 400
        if rec['code'] != otp:
            return jsonify({'success': False, 'error': 'otp_invalid'}), 400
        otps_collection.update_one({'_id': rec['_id']}, {'$set': {'verified': True}})
        # Keep info in session for password setup
        session['pending_mobile'] = {'phone': phone, 'username': rec.get('username', phone)}
        return jsonify({'success': True})
    except Exception as e:
        logger.error(f"verify_otp_mobile error: {e}")
        return jsonify({'success': False, 'error': 'verify_otp_mobile_failed'}), 500

@app.route('/api/set-password-mobile', methods=['POST'])
def set_password_mobile():
    try:
        data = request.get_json() or {}
        password = data.get('password', '').strip()
        pend = session.get('pending_mobile')
        if not pend:
            return jsonify({'success': False, 'error': 'no_pending_mobile'}), 400
        if not password or len(password) < 6:
            return jsonify({'success': False, 'error': 'weak_password'}), 400
        username = pend['username']
        phone = pend['phone']
        # Create or update user
        existing = users_collection.find_one({'username': username})
        if existing:
            users_collection.update_one({'_id': existing['_id']}, {'$set': {'password': hash_password(password), 'mobile': phone}})
            user_id = str(existing['_id'])
        else:
            result = users_collection.insert_one({
                'username': username,
                'password': hash_password(password),
                'mobile': phone,
                'created_at': datetime.utcnow(),
                'last_login': datetime.utcnow(),
            })
            user_id = str(result.inserted_id)
        session.pop('pending_mobile', None)
        session['user_id'] = user_id
        session['username'] = username
        session.permanent = True
        return jsonify({'success': True, 'user': {'id': user_id, 'username': username}})
    except Exception as e:
        logger.error(f"set_password_mobile error: {e}")
        return jsonify({'success': False, 'error': 'set_password_mobile_failed'}), 500

@app.route('/api/logout', methods=['POST'])
def logout():
    """Logout user"""
    try:
        username = session.get('username', 'Unknown')
        session.clear()
        
        logger.info(f"User logged out: {username}")
        
        return jsonify({
            'success': True,
            'message': 'Logout successful'
        })
        
    except Exception as e:
        logger.error(f"Logout error: {str(e)}")
        return jsonify({
            'success': False,
            'message': 'Logout failed'
        }), 500

@app.route('/api/check-auth', methods=['GET'])
def check_auth():
    """Check if user is authenticated"""
    try:
        if 'user_id' in session and 'username' in session:
            return jsonify({
                'success': True,
                'authenticated': True,
                'user': {
                    'id': session['user_id'],
                    'username': session['username']
                }
            })
        else:
            return jsonify({
                'success': True,
                'authenticated': False
            })
            
    except Exception as e:
        logger.error(f"Auth check error: {str(e)}")
        return jsonify({
            'success': False,
            'message': 'Auth check failed'
        }), 500

# ======================
# OAUTH SECTION: Google
# ======================
@app.route('/api/auth/google', methods=['GET'])
def google_auth():
    """Initiate Google OAuth flow - redirect to Google login page"""
    try:
        # Get Google OAuth credentials from environment variables
        GOOGLE_CLIENT_ID = os.environ.get('GOOGLE_CLIENT_ID')
        
        if not GOOGLE_CLIENT_ID:
            return jsonify({'error': 'Google OAuth not configured. Please set GOOGLE_CLIENT_ID environment variable.'}), 500
        
        # Google OAuth authorization URL
        auth_url = 'https://accounts.google.com/o/oauth2/v2/auth'
        params = {
            'client_id': GOOGLE_CLIENT_ID,
            'redirect_uri': 'http://localhost:5001/api/auth/google/callback',
            'scope': 'openid email profile',
            'response_type': 'code',
            'access_type': 'offline',
            'prompt': 'select_account'
        }
        
        # Build the authorization URL with proper URL encoding
        from urllib.parse import urlencode
        auth_url_with_params = f"{auth_url}?{urlencode(params)}"
        
        return redirect(auth_url_with_params)
        
    except Exception as e:
        logger.error(f"Google OAuth error: {str(e)}")
        return jsonify({'error': 'Google OAuth failed'}), 500

@app.route('/api/auth/google/callback', methods=['GET'])
def google_auth_callback():
    """Handle Google OAuth callback"""
    try:
        code = request.args.get('code')
        error = request.args.get('error')
        
        if error:
            logger.error(f"Google OAuth error: {error}")
            # Redirect to frontend with error
            return redirect(f"http://localhost:5173?auth=error&error={error}")
        
        if not code:
            return redirect("http://localhost:5173?auth=error&error=no_code")
        
        # Exchange code for tokens
        try:
            # Get Google OAuth credentials
            GOOGLE_CLIENT_ID = os.environ.get('GOOGLE_CLIENT_ID')
            GOOGLE_CLIENT_SECRET = os.environ.get('GOOGLE_CLIENT_SECRET')
            
            if not GOOGLE_CLIENT_ID or not GOOGLE_CLIENT_SECRET:
                return redirect("http://localhost:5173?auth=error&error=oauth_not_configured")
            
            token_url = 'https://oauth2.googleapis.com/token'
            token_data = {
                'client_id': GOOGLE_CLIENT_ID,
                'client_secret': GOOGLE_CLIENT_SECRET,
                'code': code,
                'grant_type': 'authorization_code',
                'redirect_uri': 'http://localhost:5001/api/auth/google/callback'
            }
            
            token_response = requests.post(token_url, data=token_data)
            token_response.raise_for_status()
            tokens = token_response.json()
            
            # Get user info
            user_info_url = 'https://www.googleapis.com/oauth2/v2/userinfo'
            headers = {'Authorization': f"Bearer {tokens['access_token']}"}
            user_response = requests.get(user_info_url, headers=headers)
            user_response.raise_for_status()
            user_info = user_response.json()
            
            # Check if user exists or create new user
            google_id = user_info.get('id')
            email = user_info.get('email')
            name = user_info.get('name')
            
            if not google_id or not email:
                return redirect("http://localhost:5173?auth=error&error=invalid_user_info")
            
            # Check if user exists by Google ID
            existing_user = users_collection.find_one({'google_id': google_id})
            
            if existing_user:
                # Update last login
                users_collection.update_one(
                    {'_id': existing_user['_id']},
                    {'$set': {'last_login': datetime.utcnow()}}
                )
                user_id = str(existing_user['_id'])
                username = existing_user.get('username', email.split('@')[0])
            else:
                # Create new user
                user_data = {
                    'google_id': google_id,
                    'email': email,
                    'username': email.split('@')[0],
                    'name': name,
                    'created_at': datetime.utcnow(),
                    'last_login': datetime.utcnow()
                }
                
                result = users_collection.insert_one(user_data)
                user_id = str(result.inserted_id)
                username = email.split('@')[0]
            
            # Create session
            session['user_id'] = user_id
            session['username'] = username
            session.permanent = True
            
            logger.info(f"Google OAuth login successful: {username}")
            # Redirect to frontend with success
            return redirect(f"http://localhost:5173?auth=success&user={username}")
            
        except requests.RequestException as e:
            logger.error(f"Google OAuth token exchange error: {str(e)}")
            return redirect("http://localhost:5173?auth=error&error=token_exchange_failed")
        
    except Exception as e:
        logger.error(f"Google OAuth callback error: {str(e)}")
        return redirect("http://localhost:5173?auth=error&error=callback_failed")

@app.route('/api/set-password-google', methods=['POST'])
def set_password_google():
    """Set a password for the currently authenticated Google user"""
    try:
        if 'user_id' not in session:
            return jsonify({'success': False, 'error': 'not_authenticated'}), 401
        data = request.get_json() or {}
        password = (data.get('password') or '').strip()
        if len(password) < 6:
            return jsonify({'success': False, 'error': 'weak_password'}), 400
        user_id = session['user_id']
        users_collection.update_one({'_id': ObjectId(user_id)}, {
            '$set': {
                'password': hash_password(password),
                'last_login': datetime.utcnow()
            }
        })
        return jsonify({'success': True})
    except Exception as e:
        logger.error(f"set_password_google error: {e}")
        return jsonify({'success': False, 'error': 'set_password_google_failed'}), 500

if __name__ == '__main__':
    logger.info("Starting Authentication Server...")
    logger.info(f"MongoDB URI: {MONGO_URI}")
    # Load models at startup
    try:
        load_models()
    except Exception as e:
        logger.error(f"Failed to load models at startup: {e}")
    app.run(host='0.0.0.0', port=5001, debug=True)
