# Colab Notebook + Ngrok Setup Guide

This guide explains how to set up AlphaFold2 and Docking notebooks in Google Colab with ngrok integration.

## Prerequisites

1. Google Colab account
2. Ngrok account (free tier works): https://ngrok.com/
3. Your AlphaFold2.ipynb and Docking.ipynb notebooks

## Step 1: Install Dependencies in Colab

Add this cell at the beginning of each notebook:

### For AlphaFold2.ipynb:
```python
# Install Flask and ngrok
!pip install flask flask-cors pyngrok

# Start Flask server in background
from flask import Flask, request, jsonify
from flask_cors import CORS
import threading

app = Flask(__name__)
CORS(app)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    sequence = data.get('sequence', '')
    
    # Your AlphaFold2 prediction code here
    # ... (use the sequence variable)
    
    # Return results
    return jsonify({
        'pdb_content': pdb_content,  # Your generated PDB
        'plddt_score': plddt_score,
        'confidence_scores': confidence_dict
    })

def run_flask():
    app.run(host='0.0.0.0', port=5000)

threading.Thread(target=run_flask, daemon=True).start()
print("✅ Flask server started on port 5000")
```

### For Docking.ipynb:
```python
# Install Flask and ngrok
!pip install flask flask-cors pyngrok

# Start Flask server in background
from flask import Flask, request, jsonify
from flask_cors import CORS
import threading

app = Flask(__name__)
CORS(app)

@app.route('/dock', methods=['POST'])
def dock():
    data = request.get_json()
    smiles = data.get('smiles', '')
    protein_sequence = data.get('protein_sequence', '')
    protein_pdb = data.get('protein_pdb', '')
    
    # Your Docking code here
    # ... (use smiles, protein_sequence, or protein_pdb)
    
    # Return results
    return jsonify({
        'affinities': affinities_list,
        'best_affinity': min_affinity,
        'docked_poses': num_poses,
        'pdbqt_content': pdbqt_content
    })

def run_flask():
    app.run(host='0.0.0.0', port=5000)

threading.Thread(target=run_flask, daemon=True).start()
print("✅ Flask server started on port 5000")
```

## Step 2: Setup Ngrok

Add this cell after Flask server starts:

```python
# Setup ngrok
from pyngrok import ngrok

# Get your ngrok auth token from https://dashboard.ngrok.com/get-started/your-authtoken
ngrok.set_auth_token("YOUR_NGROK_AUTH_TOKEN")  # Replace with your token

# Create public URL
public_url = ngrok.connect(5000)
print(f"🌐 Public ngrok URL: {public_url}")
print(f"✅ Use this URL in Flask backend: {public_url}")
```

## Step 3: Update Flask Backend

Set the ngrok URLs in your Flask backend environment or `.env`:

```bash
ALPHAFOLD2_NGROK_URL=https://xxxx-xxxx-xxxx.ngrok-free.app
DOCKING_NGROK_URL=https://yyyy-yyyy-yyyy.ngrok-free.app
```

Or set them directly in `combined_server.py`:
```python
ALPHAFOLD2_NGROK_URL = "https://xxxx-xxxx-xxxx.ngrok-free.app"
DOCKING_NGROK_URL = "https://yyyy-yyyy-yyyy.ngrok-free.app"
```

## Step 4: Integration Flow

1. **User enters protein sequence** → Frontend sends to Flask `/alphafold2/predict`
2. **Flask forwards to Colab ngrok** → `/predict` endpoint
3. **Colab runs AlphaFold2** → Returns PDB file and scores
4. **User gets SMILES from SMILES generation** → Can use in Docking
5. **User enters SMILES + PDB** → Frontend sends to Flask `/docking/run`
6. **Flask forwards to Colab ngrok** → `/dock` endpoint
7. **Colab runs Docking** → Returns binding affinities and poses

## Notes

- Ngrok free tier: URLs change on restart (need to update backend)
- Ngrok paid tier: Static URLs available
- Colab sessions: Notebook needs to stay running
- Timeouts: AlphaFold2 can take 5-10+ minutes, adjust Flask timeout accordingly


