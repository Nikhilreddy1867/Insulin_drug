# Fusion Model Integration - Setup Guide

## ✅ Files Required

Upload these **4 files** to `backend/models/` directory:

1. **`progen.pt`** - ProGen model checkpoint (for protein embeddings)
2. **`molt5.pt`** - MolT5 decoder model checkpoint
3. **`fusion_best.pt`** - Fusion model checkpoint (contains projection layer weights)
4. **`tokenizer.json`** - Tokenizer configuration file

## 📁 Upload Location

```
/Users/nikhilreddymallela/Downloads/Insulindrug/Insulin_Drug_Synthesis/backend/models/
```

## 🔧 How It Works

The integration matches your Gradio app's `build_models` function:

1. **Loads ProGen model** (`progen.pt`) for extracting protein embeddings
2. **Loads MolT5 decoder** (`molt5.pt`) for SMILES generation
3. **Creates FusionModel** that combines ProGen embeddings with MolT5 decoder
4. **Loads fusion checkpoint** (`fusion_best.pt`) which contains the trained projection layer weights
5. **Uses tokenizer** (`tokenizer.json`) for encoding/decoding sequences and SMILES

## 🚀 Usage

### API Endpoint

**POST** `/generate-smiles`

**Request Body:**
```json
{
  "sequence": "MKTAYIAKQR...",
  "gen_max_len": 128,
  "temperature": 1.5,
  "top_k": 50,
  "top_p": 0.9,
  "repetition_penalty": 1.15
}
```

**Response:**
```json
{
  "success": true,
  "input_sequence": "MKTAYIAKQR...",
  "smiles": "C[C@H](N)C(=O)O",
  "model_type": "fusion"
}
```

### Generation Parameters

- **`gen_max_len`** (default: 128): Maximum length of generated SMILES
- **`temperature`** (default: 1.5): Controls randomness (higher = more creative)
- **`top_k`** (default: 50): Sample from top K tokens (0 = disabled)
- **`top_p`** (default: 0.9): Nucleus sampling threshold (0.0 = disabled)
- **`repetition_penalty`** (default: 1.15): Penalty for repeated tokens (higher = less repetition)

## 🔄 Model Loading Order

1. Load tokenizer from `tokenizer.json`
2. Load ProGen model from `progen.pt` → freeze parameters
3. Load MolT5 decoder from `molt5.pt`
4. Create FusionModel wrapper
5. Load projection layer weights from `fusion_best.pt`

## ✅ Verification

After uploading files and restarting the backend, check logs for:
```
[Fusion] ✅ Loaded tokenizer with vocab size: X
[Fusion] ✅ ProGen model loaded successfully
[Fusion] ✅ MolT5 decoder loaded successfully
[Fusion] ✅ Fusion projection layer loaded from checkpoint
[Fusion] ✅ Fusion model fully loaded and ready!
```

## 🐛 Troubleshooting

- **Models not loading?** Check that all 4 files are in `backend/models/` directory
- **Prefix errors?** The code automatically strips common prefixes (`progen.`, `molt5.`, `decoder.`, `base.`, `proj.`)
- **Missing keys warnings?** Normal if checkpoint structure differs slightly - models load with `strict=False`
- **Fallback to legacy?** If fusion model fails, system falls back to legacy MiniT5 model

## 📝 Notes

- The fusion model uses **sampling** (not greedy decoding) for more diverse SMILES generation
- **Retry logic** automatically retries invalid SMILES up to 5 times with adjusted parameters
- ProGen model parameters are **frozen** (no gradients) during inference

