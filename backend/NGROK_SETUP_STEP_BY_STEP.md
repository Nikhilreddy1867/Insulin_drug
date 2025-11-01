# Step-by-Step Guide: Linking AlphaFold2.ipynb and Docking.ipynb to Frontend via Ngrok

This guide provides detailed step-by-step instructions for setting up AlphaFold2 and Docking notebooks in Google Colab and connecting them to your frontend application using ngrok.

---

## Prerequisites

1. ✅ Google Colab account (free)
2. ✅ Ngrok account (sign up at https://ngrok.com/ - free tier works)
3. ✅ Your Flask backend running on `localhost:5001`
4. ✅ Frontend application running

---

## PART 1: AlphaFold2 Setup

### Step 1: Get Ngrok Auth Token

1. Go to https://ngrok.com/ and sign up/login
2. Navigate to: **Dashboard → Your Authtoken** (or https://dashboard.ngrok.com/get-started/your-authtoken)
3. Copy your auth token (looks like: `2abc123def456ghi789jkl012mno345pqr678stu901vwx234yz`)

### Step 2: Open AlphaFold2 Notebook in Colab

1. Upload `backend/AlphaFold2 .ipynb` to Google Colab
2. Or open it directly if you have it in Google Drive

### Step 3: Add Flask Server Code to Notebook

1. **Create a new cell** at the end of your notebook (or before the prediction code)
2. **Copy and paste** the following code:

```python
# ============================================
# FLASK SERVER FOR ALPHAFOLD2
# ============================================

# Install dependencies
!pip install flask flask-cors pyngrok

from flask import Flask, request, jsonify
from flask_cors import CORS
import threading
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend access

# Global variable to store latest results
latest_result = {}

@app.route('/predict', methods=['POST'])
def predict_structure():
    """
    API endpoint for AlphaFold2 prediction
    Receives: {'sequence': 'MKTAYIAKQR...'}
    Returns: {'pdb_content': '...', 'plddt_score': 85.5, ...}
    """
    try:
        data = request.get_json()
        sequence = data.get('sequence', '').strip().upper()
        
        if not sequence:
            return jsonify({'error': 'Sequence is required'}), 400
        
        print(f"[Flask] Received AlphaFold2 request for sequence length: {len(sequence)}")
        
        # ============================================
        # YOUR ALPHAFOLD2 PREDICTION CODE HERE
        # Replace this section with your actual ColabFold code
        # ============================================
        
        # Example structure (adapt to your notebook):
        # 1. Set up query file
        jobname = f"prediction_{hash(sequence) % 10000}"
        queries_path = f"/content/{jobname}.fa"
        with open(queries_path, 'w') as f:
            f.write(f">query\n{sequence}\n")
        
        # 2. Run your AlphaFold2 prediction
        # ... (paste your existing ColabFold code here)
        # Make sure it saves results to a directory
        
        # 3. Read the generated PDB file
        result_dir = jobname  # or wherever your notebook saves results
        pdb_file = None
        
        # Find the PDB file (adjust pattern based on your notebook)
        import glob
        pdb_files = glob.glob(f"{result_dir}/*unrelaxed*.pdb")
        if pdb_files:
            pdb_file = pdb_files[0]
        
        if pdb_file and os.path.exists(pdb_file):
            with open(pdb_file, 'r') as f:
                pdb_content = f.read()
            
            # Extract pLDDT score if available (adjust based on your notebook)
            plddt_score = 85.0  # Replace with actual calculation
            
            return jsonify({
                'pdb_content': pdb_content,
                'plddt_score': plddt_score,
                'confidence_scores': {},  # Add if available
                'jobname': jobname
            })
        else:
            return jsonify({'error': 'PDB file not found after prediction'}), 500
            
    except Exception as e:
        print(f"[Flask] Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'service': 'AlphaFold2'})

# Start Flask in background
def run_flask():
    app.run(host='0.0.0.0', port=5000, debug=False)

print("🚀 Starting Flask server...")
flask_thread = threading.Thread(target=run_flask, daemon=True)
flask_thread.start()
print("✅ Flask server started on port 5000")

# ============================================
# SETUP NGROK
# ============================================

from pyngrok import ngrok

# Replace with your ngrok auth token
NGROK_AUTH_TOKEN = "YOUR_NGROK_AUTH_TOKEN_HERE"  # ⚠️ PASTE YOUR TOKEN HERE

ngrok.set_auth_token(NGROK_AUTH_TOKEN)

# Create public tunnel
public_url = ngrok.connect(5000)
print(f"🌐 AlphaFold2 ngrok URL: {public_url}")
print(f"✅ Use this in Flask backend: ALPHAFOLD2_NGROK_URL='{public_url}'")
```

### Step 4: Configure Your Code

1. **Replace `YOUR_NGROK_AUTH_TOKEN_HERE`** with your actual ngrok token from Step 1
2. **Replace the prediction code section** (marked with `# YOUR ALPHAFOLD2 PREDICTION CODE HERE`) with your actual ColabFold code from the notebook
3. **Adjust the PDB file path** to match where your notebook saves results

### Step 5: Run the Flask Cell

1. **Run the cell** containing the Flask server code
2. **Wait for output** - you should see:
   ```
   ✅ Flask server started on port 5000
   🌐 AlphaFold2 ngrok URL: https://xxxx-xxxx-xxxx.ngrok-free.app
   ```
3. **Copy the ngrok URL** (e.g., `https://xxxx-xxxx-xxxx.ngrok-free.app`)

### Step 6: Keep Notebook Running

⚠️ **IMPORTANT**: Keep the Colab notebook tab open and running. If you close it, the Flask server and ngrok tunnel will stop.

---

## PART 2: Docking Setup

### Step 1: Open Docking Notebook in Colab

1. Upload `backend/Docking.ipynb` to Google Colab (or open from Drive)

### Step 2: Add Flask Server Code to Notebook

1. **Create a new cell** at the end of your notebook
2. **Copy and paste** the following code:

```python
# ============================================
# FLASK SERVER FOR DOCKING
# ============================================

# Install dependencies
!pip install flask flask-cors pyngrok

from flask import Flask, request, jsonify
from flask_cors import CORS
import threading
import os
import tempfile
from rdkit import Chem
from rdkit.Chem import AllChem, rdmolfiles
from vina import Vina

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend access

@app.route('/dock', methods=['POST'])
def dock_molecules():
    """
    API endpoint for molecular docking
    Receives: {
        'smiles': 'C[C@H](N)C(=O)O',
        'protein_pdb': 'ATOM 1 ...' OR
        'protein_sequence': 'MKTAYIAKQR...'
    }
    Returns: {
        'affinities': [-7.5, -7.2, ...],
        'best_affinity': -7.5,
        'pdbqt_content': '...'
    }
    """
    try:
        data = request.get_json()
        smiles = data.get('smiles', '').strip()
        protein_pdb = data.get('protein_pdb', '').strip()
        protein_sequence = data.get('protein_sequence', '').strip()
        
        if not smiles:
            return jsonify({'error': 'SMILES string is required'}), 400
        
        if not protein_pdb and not protein_sequence:
            return jsonify({'error': 'Either protein_pdb or protein_sequence is required'}), 400
        
        print(f"[Flask] Received docking request for SMILES: {smiles[:50]}...")
        
        # Create temporary directory
        with tempfile.TemporaryDirectory() as tmpdir:
            # 1. Generate ligand 3D structure
            print("Generating 3D structure for ligand...")
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return jsonify({'error': 'Failed to parse SMILES'}), 400
            
            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol, AllChem.ETKDG())
            AllChem.UFFOptimizeMolecule(mol)
            
            ligand_pdb = os.path.join(tmpdir, 'ligand.pdb')
            rdmolfiles.MolToPDBFile(mol, ligand_pdb)
            
            # Convert to PDBQT
            ligand_pdbqt = os.path.join(tmpdir, 'ligand.pdbqt')
            os.system(f"obabel {ligand_pdb} -O {ligand_pdbqt} -xh")
            
            # 2. Handle protein
            if protein_pdb:
                # Use provided PDB
                protein_pdb_file = os.path.join(tmpdir, 'protein.pdb')
                with open(protein_pdb_file, 'w') as f:
                    f.write(protein_pdb)
            elif protein_sequence:
                # If only sequence provided, return error (needs AlphaFold2 first)
                return jsonify({
                    'error': 'Please provide protein PDB file content. Use AlphaFold2 prediction first.'
                }), 400
            
            # Convert protein to PDBQT
            protein_pdbqt = os.path.join(tmpdir, 'protein.pdbqt')
            protein_pdb_file = os.path.join(tmpdir, 'protein.pdb')
            os.system(f"obabel {protein_pdb_file} -O {protein_pdbqt} -xr")
            
            # 3. Run Vina docking
            print("Running AutoDock Vina...")
            v = Vina(sf_name='vina')
            v.set_receptor(protein_pdbqt)
            v.set_ligand_from_file(ligand_pdbqt)
            
            # Compute binding site (adjust coordinates as needed)
            v.compute_vina_maps(center=[0, 0, 0], box_size=[25, 25, 25])
            
            # Perform docking
            v.dock(exhaustiveness=8, n_poses=5)
            
            # Save results
            docked_pdbqt = os.path.join(tmpdir, 'docked_out.pdbqt')
            v.write_poses(docked_pdbqt, n_poses=5, overwrite=True)
            
            # Get affinities
            scores = v.poses()
            affinities = [pose[1] for pose in scores] if scores else []
            
            # If scores empty, parse from file
            if not affinities:
                with open(docked_pdbqt, 'r') as f:
                    for line in f:
                        if line.startswith("REMARK VINA RESULT"):
                            parts = line.split()
                            if len(parts) > 3:
                                affinities.append(float(parts[3]))
            
            # Read PDBQT content
            with open(docked_pdbqt, 'r') as f:
                pdbqt_content = f.read()
            
            return jsonify({
                'affinities': affinities,
                'best_affinity': min(affinities) if affinities else None,
                'docked_poses': len(affinities),
                'pdbqt_content': pdbqt_content
            })
            
    except Exception as e:
        print(f"[Flask] Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'service': 'Docking'})

# Start Flask in background
def run_flask():
    app.run(host='0.0.0.0', port=5000, debug=False)

print("🚀 Starting Flask server...")
flask_thread = threading.Thread(target=run_flask, daemon=True)
flask_thread.start()
print("✅ Flask server started on port 5000")

# ============================================
# SETUP NGROK
# ============================================

from pyngrok import ngrok

# Replace with your ngrok auth token
NGROK_AUTH_TOKEN = "YOUR_NGROK_AUTH_TOKEN_HERE"  # ⚠️ PASTE YOUR TOKEN HERE

ngrok.set_auth_token(NGROK_AUTH_TOKEN)

# Create public tunnel
public_url = ngrok.connect(5000)
print(f"🌐 Docking ngrok URL: {public_url}")
print(f"✅ Use this in Flask backend: DOCKING_NGROK_URL='{public_url}'")
```

### Step 3: Configure Your Code

1. **Replace `YOUR_NGROK_AUTH_TOKEN_HERE`** with your ngrok token
2. **Adjust docking parameters** if needed (binding site coordinates, exhaustiveness, etc.)

### Step 4: Run the Flask Cell

1. **Run the cell** containing the Flask server code
2. **Copy the ngrok URL** (e.g., `https://yyyy-yyyy-yyyy.ngrok-free.app`)

---

## PART 3: Configure Flask Backend

### Step 1: Update combined_server.py

Open `backend/combined_server.py` and find these lines (around line 1308):

```python
ALPHAFOLD2_NGROK_URL = os.environ.get('ALPHAFOLD2_NGROK_URL', 'http://localhost:8000')
DOCKING_NGROK_URL = os.environ.get('DOCKING_NGROK_URL', 'http://localhost:8001')
```

### Step 2: Set the Ngrok URLs

Replace with your actual ngrok URLs from Part 1 Step 5 and Part 2 Step 4:

```python
ALPHAFOLD2_NGROK_URL = "https://xxxx-xxxx-xxxx.ngrok-free.app"  # From AlphaFold2 notebook
DOCKING_NGROK_URL = "https://yyyy-yyyy-yyyy.ngrok-free.app"  # From Docking notebook
```

**OR** set environment variables before starting Flask:

```bash
# Windows PowerShell
$env:ALPHAFOLD2_NGROK_URL="https://xxxx-xxxx-xxxx.ngrok-free.app"
$env:DOCKING_NGROK_URL="https://yyyy-yyyy-yyyy.ngrok-free.app"
python combined_server.py

# Linux/Mac
export ALPHAFOLD2_NGROK_URL="https://xxxx-xxxx-xxxx.ngrok-free.app"
export DOCKING_NGROK_URL="https://yyyy-yyyy-yyyy.ngrok-free.app"
python combined_server.py
```

### Step 3: Restart Flask Backend

1. Stop your Flask backend if running
2. Start it again: `python backend/combined_server.py`

---

## PART 4: Test the Integration

### Step 1: Test AlphaFold2

1. Open your frontend application
2. Go to Dashboard → **AlphaFold2** tab
3. Enter a protein sequence (e.g., `MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQFSTLKSTVEAIWAGIKATEAAVSEEFGLAPFLPDQIHFVHSQELLSRYPDLDAKGRERAIAKDLGAVFLVGIGGKLSDGHRHDVRAPDYDDWQTSTSTSLPRADLQLFVDGVRQLEWLSQRLQQPQQKSAFAVQEDFNRSWFRPGHRRNKVFDLPIGVLKSSAQNLMNQEDVHSKQAPGTILKSQGMQVFVLEELDKTLFTLGFHKPAIVQHASSAKDLGPLLDGIWKTTTTKQAAKCLQKNLPSFLGVTSSEFRYLMNSQTRLPDNYLPLLPAIIDRFDNTLPLTGQAQIIFRRFLPLQGKEFQ`)
4. Click **Predict Structure**
5. Wait for results (can take 5-10+ minutes)

### Step 2: Test Docking

1. In Dashboard → **SMILES Generation** tab, generate a SMILES string
2. In Dashboard → **AlphaFold2** tab, get a PDB file
3. Go to Dashboard → **Docking** tab
4. Paste the SMILES string
5. Paste the PDB file content (or use protein sequence)
6. Click **Run Docking**
7. View binding affinities

---

## Troubleshooting

### Issue: "Cannot connect to AlphaFold2/Docking service"

**Solution:**
- Check that Colab notebooks are still running
- Verify ngrok URLs are correct in `combined_server.py`
- Restart ngrok in Colab: Run the Flask cell again

### Issue: Ngrok URL changed

**Solution:**
- Free tier ngrok URLs change when you restart
- Update `combined_server.py` with the new URL
- Or use ngrok paid tier for static URLs

### Issue: Flask server not starting in Colab

**Solution:**
- Check for errors in Colab cell output
- Ensure all dependencies are installed
- Verify ngrok token is correct

### Issue: Prediction takes too long

**Solution:**
- This is normal - AlphaFold2 can take 5-10+ minutes
- Check Colab notebook output for progress
- Ensure Colab has GPU enabled (Runtime → Change runtime type → GPU)

---

## Important Notes

1. ⚠️ **Keep Colab notebooks running** - Closing tabs stops the servers
2. ⚠️ **Ngrok free tier URLs expire** - Update backend config if URL changes
3. ⚠️ **GPU required for AlphaFold2** - Enable GPU in Colab runtime settings
4. ✅ **Test with short sequences first** - Faster predictions for testing
5. ✅ **Check Colab logs** - Flask server logs appear in Colab cell output

---

## Complete Workflow Summary

```
1. User enters sequence in Dashboard → AlphaFold2 tab
   ↓
2. Frontend sends POST to Flask /alphafold2/predict
   ↓
3. Flask forwards to Colab ngrok URL /predict
   ↓
4. Colab runs AlphaFold2 → Returns PDB file
   ↓
5. User copies PDB + SMILES → Docking tab
   ↓
6. Frontend sends POST to Flask /docking/run
   ↓
7. Flask forwards to Colab ngrok URL /dock
   ↓
8. Colab runs Docking → Returns affinities
```

---

**You're all set!** The AlphaFold2 and Docking features are now integrated into your Dashboard. 🎉


