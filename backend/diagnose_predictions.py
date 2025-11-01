#!/usr/bin/env python3
"""
Diagnostic script to check model loading and test predictions
"""
import os
import sys
import joblib
import torch
import numpy as np

print("=" * 60)
print("MODEL DIAGNOSTIC SCRIPT")
print("=" * 60)

# Check label encoder
print("\n1. Checking Label Encoder...")
try:
    le = joblib.load('models/label_encoder.pkl')
    print(f"   [OK] Label encoder loaded")
    print(f"   Classes: {le.classes_.tolist()}")
    print(f"   Number of classes: {len(le.classes_)}")
    print(f"   Class indices: {dict(enumerate(le.classes_))}")
except Exception as e:
    print(f"   [ERROR] Error loading label encoder: {e}")
    sys.exit(1)

# Check PCA model
print("\n2. Checking PCA Model...")
try:
    pca = joblib.load('models/pca_model.pkl')
    print(f"   [OK] PCA model loaded")
    print(f"   PCA components: {pca.n_components_ if hasattr(pca, 'n_components_') else 'Unknown'}")
    print(f"   PCA input shape: {pca.n_features_in_ if hasattr(pca, 'n_features_in_') else 'Unknown'}")
except Exception as e:
    print(f"   [ERROR] Error loading PCA: {e}")
    sys.exit(1)

# Check MLP models
print("\n3. Checking MLP Models...")
mlp_custom = 'models/best_mlp_custom_adv.pth'
mlp_medium = 'models/best_mlp_medium_adv.pth'

if os.path.exists(mlp_custom):
    print(f"   [OK] Found: {mlp_custom}")
    try:
        state = torch.load(mlp_custom, map_location='cpu')
        print(f"      Keys: {list(state.keys())[:5]}...")
        print(f"      Total parameters: {sum(p.numel() for p in state.values() if isinstance(p, torch.Tensor))}")
    except Exception as e:
        print(f"      [WARNING]  Error loading: {e}")
else:
    print(f"   [ERROR] NOT FOUND: {mlp_custom}")

if os.path.exists(mlp_medium):
    print(f"   [OK] Found: {mlp_medium}")
    try:
        state = torch.load(mlp_medium, map_location='cpu')
        if isinstance(state, dict):
            print(f"      Keys: {list(state.keys())[:5]}...")
            total = sum(p.numel() for p in state.values() if isinstance(p, torch.Tensor))
        else:
            print(f"      Direct state_dict (not dict)")
            total = sum(p.numel() for p in state if isinstance(p, torch.Tensor))
        print(f"      Total parameters: {total}")
    except Exception as e:
        print(f"      [WARNING]  Error loading: {e}")
else:
    print(f"   [ERROR] NOT FOUND: {mlp_medium}")

# Check ProteinLM checkpoint (try multiple possible names)
print("\n4. Checking ProteinLM Checkpoint...")
checkpoint_files = [
    'models/protein_classifier.pt',
    'models/custom_protein_lm.pt',
    'models/final_ckpt.pt'
]
checkpoint_found = False

for custom_lm in checkpoint_files:
    if os.path.exists(custom_lm):
        print(f"   [OK] Found: {custom_lm}")
        checkpoint_found = True
        try:
            ckpt = torch.load(custom_lm, map_location='cpu')
            if isinstance(ckpt, dict):
                if 'model_state' in ckpt:
                    print(f"      [OK] Has 'model_state' key")
                    state = ckpt['model_state']
                    print(f"      Model parameters: {sum(p.numel() for p in state.values() if isinstance(p, torch.Tensor))}")
                else:
                    print(f"      [WARNING]  No 'model_state' key, treating as direct state_dict")
                    print(f"      Keys: {list(ckpt.keys())[:5]}...")
            else:
                print(f"      Direct state_dict")
        except Exception as e:
            print(f"      [WARNING]  Error loading: {e}")
        break

if not checkpoint_found:
    print(f"   [ERROR] NOT FOUND: None of the checkpoint files exist")
    for f in checkpoint_files:
        print(f"      - {f}")
    print(f"   [WARNING]  CRITICAL: Without a checkpoint, embeddings will use RANDOM weights!")
    print(f"      This will cause incorrect predictions!")

# Test prediction on sample sequence
print("\n5. Testing Prediction Pipeline...")
test_seq = "LSHAFGRRLVLSSTFRILADLLGFARPLCIFGIVDHILGKENDVFQPKTQF"
print(f"   Test sequence: {test_seq[:50]}... (length: {len(test_seq)})")

# Import model loading code
sys.path.insert(0, os.path.dirname(__file__))
try:
    from combined_server import preprocess_sequence, embed_sequence, load_models
    from combined_server import custom_protein_lm, mlp_model, label_encoder, pca_model, device
    
    print("\n6. Loading Models (this may take a moment)...")
    load_models()
    
    # Access models directly from combined_server module after loading
    import combined_server
    # Force reload to get updated globals
    import importlib
    importlib.reload(combined_server)
    
    # Get models from the module
    custom_protein_lm = combined_server.custom_protein_lm
    mlp_model = combined_server.mlp_model
    label_encoder = combined_server.label_encoder
    pca_model = combined_server.pca_model
    device = combined_server.device
    
    # Verify models loaded
    if custom_protein_lm is None:
        print("   [ERROR] custom_protein_lm is None!")
        raise RuntimeError("ProteinLM model not loaded")
    if mlp_model is None:
        print("   [ERROR] mlp_model is None!")
        raise RuntimeError("MLP model not loaded")
    if label_encoder is None:
        print("   [ERROR] label_encoder is None!")
        raise RuntimeError("Label encoder not loaded")
    if pca_model is None:
        print("   [WARNING] pca_model is None - PCA will be skipped")
    else:
        print("   [OK] All models loaded successfully")
    
    print("\n7. Running Test Prediction...")
    proc = preprocess_sequence(test_seq)
    print(f"   Processed: {proc[:50]}... (length: {len(proc)})")
    
    emb = embed_sequence(proc)
    print(f"   Embedding shape (before PCA): {emb.shape}")
    
    if pca_model is not None:
        emb_pca = pca_model.transform(emb)
        print(f"   Embedding shape (after PCA): {emb_pca.shape}")
    else:
        emb_pca = emb
        print(f"   [WARNING]  PCA model is None - using raw embeddings!")
    
    X = torch.tensor(emb_pca, dtype=torch.float32).to(device)
    with torch.no_grad():
        logits = mlp_model(X)
        probs = torch.nn.functional.softmax(logits, dim=1)
        idx = torch.argmax(probs, dim=1).item()
        conf = probs[0, idx].item()
        pred = label_encoder.classes_[idx]
    
    print(f"\n   Prediction Result:")
    print(f"   - Predicted class: {pred}")
    print(f"   - Confidence: {conf:.4f}")
    print(f"\n   All probabilities:")
    for i, cls in enumerate(label_encoder.classes_):
        prob = probs[0, i].item()
        marker = " <--" if i == idx else ""
        print(f"   - {cls}: {prob:.4f}{marker}")
    
    # Check if model is biased
    print(f"\n8. Checking for Model Bias...")
    all_probs = probs[0].cpu().numpy()
    max_prob = np.max(all_probs)
    if max_prob > 0.9:
        print(f"   [WARNING]  WARNING: Very high confidence ({max_prob:.4f})")
        print(f"      Model might be overconfident or biased")
    if np.std(all_probs) < 0.1:
        print(f"   [WARNING]  WARNING: Low probability variance (std={np.std(all_probs):.4f})")
        print(f"      Probabilities are too uniform - model may not be learning")
    
except Exception as e:
    print(f"   [ERROR] Error during prediction test: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("DIAGNOSTIC COMPLETE")
print("=" * 60)

