# AlphaFold2 Colab Notebook Integration Guide

This guide explains how to modify the ColabFold AlphaFold2 notebook to integrate with your Flask backend using ngrok.

## 📋 Step-by-Step Instructions

### Step 1: Open the ColabFold Notebook
1. Go to: https://colab.research.google.com/github/sokrypton/ColabFold/blob/main/AlphaFold2.ipynb
2. **Make a copy** (File → Save a copy in Drive)

### Step 2: Run All Initial Setup Cells
1. Run the "Install dependencies" cell (if needed)
2. Set your input sequence and other parameters
3. Run the "Run Prediction" cell and wait for it to complete

### Step 3: Add Flask & ngrok Setup Cell
**Add this as a NEW cell AFTER the "Run Prediction" cell completes:**

```python
#@title Setup Flask Server & ngrok (Run this after prediction completes)
# Install required packages
!pip install -q flask flask-cors pyngrok

from flask import Flask, request, jsonify
from flask_cors import CORS
import threading
import os
import zipfile
import glob
import re
from pathlib import Path

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests

# Store the results from the prediction
global_prediction_results = results  # This uses the 'results' variable from prediction cell
global_jobname = jobname  # This uses the 'jobname' variable from prediction cell
```

### Step 4: Add Flask Endpoints Cell
**Add this as a NEW cell:**

```python
#@title Flask Endpoints for API Access

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'service': 'AlphaFold2'})

@app.route('/predict', methods=['POST'])
def predict_structure():
    """
    Endpoint for AlphaFold2 structure prediction
    Expects: {'sequence': 'MKTAYIAKQR...'}
    Returns: {
        'pdb_content': '...',  # Rank 1 PDB content
        'plddt_score': 85.5,
        'jobname': '...',
        'rank1_file': '...'
    }
    """
    try:
        data = request.get_json()
        sequence = data.get('sequence', '').strip().upper()
        
        if not sequence:
            return jsonify({'error': 'Sequence is required'}), 400
        
        print(f"\n[AlphaFold2] ==========================================")
        print(f"[AlphaFold2] Received prediction request")
        print(f"[AlphaFold2] Sequence length: {len(sequence)}")
        print(f"[AlphaFold2] Starting prediction...")
        
        # Re-run prediction for the new sequence
        # (You can modify this to reuse existing results if sequence matches)
        from colabfold.download import download_alphafold_params
        from colabfold.utils import setup_logging
        from colabfold.batch import get_queries, run, set_model_type
        
        # Setup job for new sequence
        import hashlib
        def add_hash(x, y):
            return x + "_" + hashlib.sha1(y.encode()).hexdigest()[:5]
        
        basejobname = "".join(re.sub(r'\W+', '', "prediction"))
        new_jobname = add_hash(basejobname, sequence)
        
        # Check if directory exists
        n = 0
        while os.path.exists(new_jobname):
            new_jobname = f"{new_jobname}_{n}"
            n += 1
        
        os.makedirs(new_jobname, exist_ok=True)
        
        # Save query
        queries_path = os.path.join(new_jobname, f"{new_jobname}.csv")
        with open(queries_path, "w") as f:
            f.write(f"id,sequence\n{new_jobname},{sequence}")
        
        print(f"[AlphaFold2] Job name: {new_jobname}")
        
        # Setup logging
        log_filename = os.path.join(new_jobname, "log.txt")
        setup_logging(Path(log_filename))
        
        # Get queries and model type
        queries, is_complex = get_queries(queries_path)
        model_type = set_model_type(is_complex, model_type)
        
        # Download parameters if needed
        download_alphafold_params(model_type, Path("."))
        
        print(f"[AlphaFold2] Running prediction (this may take 5-15 minutes)...")
        
        # Run prediction
        results = run(
            queries=queries,
            result_dir=new_jobname,
            use_templates=use_templates,
            custom_template_path=custom_template_path,
            num_relax=num_relax,
            msa_mode=msa_mode,
            model_type=model_type,
            num_models=5,
            num_recycles=num_recycles,
            relax_max_iterations=relax_max_iterations,
            recycle_early_stop_tolerance=recycle_early_stop_tolerance,
            num_seeds=num_seeds,
            use_dropout=use_dropout,
            model_order=[1,2,3,4,5],
            is_complex=is_complex,
            data_dir=Path("."),
            keep_existing_results=False,
            rank_by="auto",
            pair_mode=pair_mode,
            pairing_strategy=pairing_strategy,
            stop_at_score=float(100),
            dpi=dpi,
            zip_results=False,  # Don't create ZIP, we'll extract PDB directly
            save_all=save_all,
            max_msa=max_msa,
            use_cluster_profile=use_cluster_profile if "multimer" not in model_type else False,
            calc_extra_ptm=calc_extra_ptm,
        )
        
        print(f"[AlphaFold2] Prediction completed! Extracting rank 1 PDB...")
        
        # Extract rank 1 PDB from results
        jobname_prefix = ".custom" if msa_mode == "custom" else ""
        tag = results["rank"][0][0]  # Get rank 1 tag (first element)
        
        # Construct PDB filename
        pdb_filename = f"{new_jobname}/{new_jobname}{jobname_prefix}_unrelaxed_{tag}.pdb"
        
        if not os.path.exists(pdb_filename):
            # Try alternative naming
            pdb_files = glob.glob(f"{new_jobname}/*unrelaxed*.pdb")
            if pdb_files:
                pdb_files.sort()
                pdb_filename = pdb_files[0]
            else:
                return jsonify({'error': 'PDB file not found after prediction'}), 500
        
        # Read PDB content
        with open(pdb_filename, 'r') as f:
            pdb_content = f.read()
        
        print(f"[AlphaFold2] ✅ Rank 1 PDB extracted: {os.path.basename(pdb_filename)}")
        
        # Extract pLDDT score from PDB (average B-factor)
        plddt_values = []
        for line in pdb_content.split('\n'):
            if line.startswith('ATOM'):
                try:
                    b_factor = float(line[60:66].strip())
                    if 0 <= b_factor <= 100:
                        plddt_values.append(b_factor)
                except (ValueError, IndexError):
                    continue
        
        plddt_score = sum(plddt_values) / len(plddt_values) if plddt_values else 85.0
        
        print(f"[AlphaFold2] Average pLDDT: {plddt_score:.2f}")
        print(f"[AlphaFold2] ✅ SUCCESS! Returning results...")
        
        return jsonify({
            'pdb_content': pdb_content,
            'plddt_score': round(plddt_score, 2),
            'jobname': new_jobname,
            'rank1_file': os.path.basename(pdb_filename)
        })
        
    except Exception as e:
        print(f"[AlphaFold2] ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

print("✅ Flask endpoints defined")
```

### Step 5: Start Flask Server & ngrok
**Add this as a NEW cell:**

```python
#@title Start Flask Server and ngrok

def run_flask():
    """Run Flask server in background"""
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)

# Start Flask in background thread
print("Starting Flask server...")
flask_thread = threading.Thread(target=run_flask, daemon=True)
flask_thread.start()
print("✅ Flask server started on port 5000")

# Setup ngrok
from pyngrok import ngrok

# IMPORTANT: Replace with your ngrok auth token
# Get your token from: https://dashboard.ngrok.com/get-started/your-authtoken
NGROK_AUTH_TOKEN = "YOUR_NGROK_TOKEN_HERE"  # ⚠️ REPLACE THIS!

if NGROK_AUTH_TOKEN and NGROK_AUTH_TOKEN != "YOUR_NGROK_TOKEN_HERE":
    ngrok.set_auth_token(NGROK_AUTH_TOKEN)
    public_url = ngrok.connect(5000)
    print(f"\n🌐 ==========================================")
    print(f"🌐 Public ngrok URL: {public_url}")
    print(f"🌐 ==========================================")
    print(f"\n📋 Next Steps:")
    print(f"1. Copy the ngrok URL above")
    print(f"2. Add to your Flask backend .env file:")
    print(f"   ALPHAFOLD2_NGROK_URL={public_url}")
    print(f"3. Restart your Flask backend server")
    print(f"\n✅ Your Colab notebook is now ready to receive API requests!")
    print(f"   Test health: {public_url}/health")
else:
    print("⚠️  WARNING: Set your NGROK_AUTH_TOKEN in the cell above!")
    print("   Get your token from: https://dashboard.ngrok.com/get-started/your-authtoken")
```

### Step 6: Test the Setup (Optional)
**Add this as a test cell:**

```python
#@title Test the API (Optional)

# Test health endpoint
import requests
public_url = ngrok.get_tunnels()[0].public_url if ngrok.get_tunnels() else None

if public_url:
    print(f"Testing health endpoint: {public_url}/health")
    try:
        response = requests.get(f"{public_url}/health", timeout=5)
        print(f"✅ Health check: {response.json()}")
    except Exception as e:
        print(f"❌ Health check failed: {e}")
else:
    print("⚠️  No ngrok tunnel found. Make sure you ran the previous cell.")
```

## 🔧 Configuration

### Get ngrok Auth Token
1. Sign up at https://ngrok.com (free tier works)
2. Go to: https://dashboard.ngrok.com/get-started/your-authtoken
3. Copy your auth token
4. Replace `YOUR_NGROK_TOKEN_HERE` in Step 5

### Update Backend Configuration
1. Copy the ngrok URL from Colab output
2. Add to `backend/.env`:
   ```
   ALPHAFOLD2_NGROK_URL=https://your-ngrok-url.ngrok-free.app
   ```
3. Restart Flask backend

## 📝 How It Works

1. **Your Flask Backend** receives a request from the frontend at `/alphafold2/predict`
2. **Backend forwards** the request to the Colab notebook's ngrok URL: `{ALPHAFOLD2_NGROK_URL}/predict`
3. **Colab notebook** runs AlphaFold2 prediction (takes 5-15 minutes)
4. **Colab extracts** the rank 1 PDB file from results
5. **Colab returns** the PDB content directly (no ZIP download)
6. **Backend receives** the PDB content and returns it to the frontend

## 🎯 Key Points

- **No manual copying**: PDB content is sent directly via API
- **Automatic extraction**: Rank 1 PDB is automatically extracted
- **Real-time integration**: Works seamlessly with your existing Flask backend
- **No ZIP files**: Returns only the rank 1 PDB content as text

## 🔄 After Prediction Completes

Once you've added these cells:
1. The notebook will run AlphaFold2 prediction as normal
2. After prediction, Flask server starts automatically
3. You'll get an ngrok URL to use in your backend
4. Future requests from your website will automatically:
   - Trigger new predictions in Colab
   - Extract rank 1 PDB
   - Return it directly to your backend

## ⚠️ Important Notes

- **Keep Colab tab open**: The notebook must stay running for the API to work
- **Ngrok free tier**: URLs change on restart (upgrade for static URLs)
- **Colab timeouts**: Free Colab sessions timeout after ~12 hours of inactivity
- **GPU required**: Make sure Runtime → Change runtime type → GPU is selected

## 🐛 Troubleshooting

**"Connection refused" errors:**
- Make sure Flask server cell ran successfully
- Check that ngrok tunnel is active (should show URL)

**"No PDB file found" errors:**
- Check that prediction completed successfully
- Verify the jobname folder contains PDB files

**Ngrok URL not working:**
- Regenerate ngrok tunnel (re-run Step 5 cell)
- Update backend .env file with new URL

