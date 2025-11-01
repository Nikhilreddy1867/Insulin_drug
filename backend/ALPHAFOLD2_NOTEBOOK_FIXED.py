# =============================================
# 🔹 STEP 1 — Install dependencies
# =============================================
!pip install flask flask-cors pyngrok biopython matplotlib colabfold[alphafold-minus-jax] --quiet

import os, re, hashlib, random, glob, warnings, threading, socket, time
from flask import Flask, request, jsonify
from flask_cors import CORS
from pyngrok import ngrok

warnings.simplefilter('ignore')

# =============================================
# 🔹 STEP 2 — Setup Flask
# =============================================
app = Flask(__name__)
CORS(app)

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'AlphaFold2 service healthy'})

@app.route('/predict', methods=['POST'])
def predict_structure():
    """
    Predict 3D protein structure using AlphaFold2
    Expected request: {'sequence': 'MKTAYIAKQR...'}
    Returns: {
        'pdb_content': '...',
        'plddt_score': 85.5,
        'jobname': '...',
        'confidence_scores': {}
    }
    """
    try:
        data = request.get_json()
        sequence = data.get('sequence', '').strip().upper()
        
        if not sequence:
            return jsonify({'error': 'Protein sequence required'}), 400

        # Validate sequence
        valid_amino_acids = set("ACDEFGHIKLMNPQRSTVWY")
        if not all(c in valid_amino_acids for c in sequence):
            return jsonify({'error': 'Invalid amino acid sequence. Only standard 20 amino acids allowed.'}), 400
        
        if len(sequence) < 10:
            return jsonify({'error': 'Sequence too short. Minimum 10 amino acids required.'}), 400

        print(f"\n[AlphaFold] ==========================================")
        print(f"[AlphaFold] Received sequence of length {len(sequence)}")
        print(f"[AlphaFold] Starting prediction...")

        # --- Prepare job folder ---
        jobname = "job_" + hashlib.sha1(sequence.encode()).hexdigest()[:8]
        os.makedirs(jobname, exist_ok=True)

        queries_path = os.path.join(jobname, f"{jobname}.csv")
        with open(queries_path, "w") as f:
            f.write(f"id,sequence\n{jobname},{sequence}")

        print(f"[AlphaFold] Job name: {jobname}")

        # --- AlphaFold prediction (ColabFold call) ---
        from colabfold.download import download_alphafold_params
        from colabfold.utils import setup_logging
        from colabfold.batch import get_queries, run, set_model_type
        from pathlib import Path

        setup_logging(Path(f"{jobname}/log.txt"))
        queries, is_complex = get_queries(queries_path)
        model_type = set_model_type(is_complex, "auto")

        print(f"[AlphaFold] Downloading AlphaFold parameters...")
        download_alphafold_params(model_type, Path("."))

        print(f"[AlphaFold] Running AlphaFold2 prediction...")
        print(f"[AlphaFold] This may take 5-15 minutes depending on sequence length...")
        
        results = run(
            queries=queries,
            result_dir=Path(jobname),
            use_templates=False,
            custom_template_path=None,
            num_relax=0,
            msa_mode="mmseqs2_uniref_env",
            model_type=model_type,
            num_models=1,
            num_recycles=3,
            relax_max_iterations=200,
            num_seeds=1,
            use_dropout=False,
            model_order=[1],
            is_complex=is_complex,
            data_dir=Path("."),
            keep_existing_results=False,
            rank_by="auto",
            pair_mode="unpaired_paired",
            pairing_strategy="greedy",
            stop_at_score=float(100),
            dpi=150,
            zip_results=False,
            save_all=False,
            max_msa=None,
            use_cluster_profile=True,
            calc_extra_ptm=False,
        )

        print(f"[AlphaFold] Prediction completed! Searching for PDB files...")

        # --- Find output PDB ---
        pdb_files = glob.glob(f"{jobname}/*unrelaxed*.pdb")
        if not pdb_files:
            print(f"[AlphaFold] ERROR: No PDB file found in {jobname}")
            print(f"[AlphaFold] Files in directory: {os.listdir(jobname)}")
            return jsonify({'error': 'No PDB file generated. Check logs for details.'}), 500

        pdb_file = pdb_files[0]
        print(f"[AlphaFold] Found PDB file: {pdb_file}")
        
        with open(pdb_file, 'r') as f:
            pdb_content = f.read()

        # --- Extract pLDDT score from PDB file (average of B-factor column) ---
        plddt_score = 85.0  # Default
        
        try:
            plddt_values = []
            for line in pdb_content.split('\n'):
                if line.startswith('ATOM'):
                    # pLDDT is typically in the B-factor column (column 60-66)
                    try:
                        b_factor = float(line[60:66].strip())
                        if 0 <= b_factor <= 100:  # Valid pLDDT range
                            plddt_values.append(b_factor)
                    except (ValueError, IndexError):
                        continue
            
            if plddt_values:
                plddt_score = sum(plddt_values) / len(plddt_values)
                print(f"[AlphaFold] Calculated pLDDT: {plddt_score:.2f} (from {len(plddt_values)} atoms)")
            else:
                print(f"[AlphaFold] Warning: Could not extract pLDDT from PDB, using default: 85.0")
        except Exception as e:
            print(f"[AlphaFold] Warning: Could not extract pLDDT from PDB: {e}")

        print(f"[AlphaFold] ==========================================")
        print(f"[AlphaFold] ✅ SUCCESS! Structure prediction complete")
        print(f"[AlphaFold] pLDDT score: {plddt_score:.2f}")
        print(f"[AlphaFold] PDB file size: {len(pdb_content)} characters")
        print(f"[AlphaFold] ==========================================\n")

        # --- Return result in format expected by frontend ---
        return jsonify({
            'pdb_content': pdb_content,
            'plddt_score': round(plddt_score, 2),
            'jobname': jobname,
            'confidence_scores': {}  # Can be populated with per-residue scores if needed
        })

    except Exception as e:
        import traceback
        error_msg = str(e)
        traceback.print_exc()
        print(f"\n[AlphaFold] ==========================================")
        print(f"[AlphaFold] ❌ ERROR: {error_msg}")
        print(f"[AlphaFold] ==========================================\n")
        return jsonify({'error': error_msg}), 500

# =============================================
# 🔹 STEP 3 — Helper function to check port
# =============================================
def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

# =============================================
# 🔹 STEP 4 — Kill existing Flask processes
# =============================================
def kill_port(port):
    """Kill process using the specified port"""
    try:
        import subprocess
        # Try lsof first (Linux/Colab)
        result = subprocess.run(['lsof', '-ti:{}'.format(port)], capture_output=True, text=True)
        if result.returncode == 0:
            pids = result.stdout.strip().split('\n')
            for pid in pids:
                if pid:
                    subprocess.run(['kill', '-9', pid], capture_output=True)
                    print(f"✅ Killed process {pid} on port {port}")
            return True
        else:
            # Try fuser as alternative
            subprocess.run(['fuser', '-k', '{}/tcp'.format(port)], capture_output=True, stderr=subprocess.DEVNULL)
            return True
    except Exception as e:
        print(f"⚠️  Could not kill port {port}: {e}")
        return False

# =============================================
# 🔹 STEP 5 — Free port 5000 if in use
# =============================================
FLASK_PORT = 5000

if is_port_in_use(FLASK_PORT):
    print(f"⚠️  Port {FLASK_PORT} is in use. Attempting to free it...")
    if kill_port(FLASK_PORT):
        time.sleep(1)  # Wait for port to be freed
        if is_port_in_use(FLASK_PORT):
            print(f"⚠️  Port {FLASK_PORT} still in use. Trying port 5001...")
            FLASK_PORT = 5001
    else:
        print(f"⚠️  Could not free port {FLASK_PORT}. Trying port 5001...")
        FLASK_PORT = 5001
else:
    print(f"✅ Port {FLASK_PORT} is available")

# =============================================
# 🔹 STEP 6 — Run Flask in background
# =============================================
flask_thread = None

def run_flask():
    try:
        app.run(host='0.0.0.0', port=FLASK_PORT, debug=False, use_reloader=False, threaded=True)
    except Exception as e:
        print(f"❌ Flask server error: {e}")
        print("   Try restarting the runtime (Runtime → Restart runtime)")

print("\n" + "="*60)
print("🚀 Starting Flask server in background...")
print("="*60)

# Start Flask server
flask_thread = threading.Thread(target=run_flask, daemon=True)
flask_thread.start()

# Wait for Flask to start
for i in range(5):
    time.sleep(1)
    if is_port_in_use(FLASK_PORT):
        print(f"✅ Flask server started successfully on port {FLASK_PORT}")
        break
else:
    print(f"⚠️  Flask server may not have started. Waiting a bit more...")
    time.sleep(2)

print("ℹ️  Keep this cell running to maintain the service!")
print("="*60 + "\n")

# =============================================
# 🔹 STEP 7 — Setup ngrok
# =============================================
NGROK_AUTH_TOKEN = "34rVSQJ2Qtvmo53ShGFusPikqyF_5BVRzLan4Tqhuw3tfLtdH"
ngrok.set_auth_token(NGROK_AUTH_TOKEN)

# Kill any existing ngrok tunnels to avoid conflicts
try:
    tunnels = ngrok.get_tunnels()
    for tunnel in tunnels:
        ngrok.disconnect(tunnel.public_url)
        print(f"🔌 Disconnected existing tunnel: {tunnel.public_url}")
except Exception:
    pass

# Create new tunnel
print(f"🌐 Creating ngrok tunnel to port {FLASK_PORT}...")
try:
    public_url = ngrok.connect(FLASK_PORT)
    
    # Extract clean URL from ngrok response
    ngrok_url_str = str(public_url)
    if 'NgrokTunnel:' in ngrok_url_str:
        # Extract URL from "NgrokTunnel: "https://..." -> "http://localhost:5000""
        match = re.search(r'"(https://[^"]+)"', ngrok_url_str)
        if match:
            ngrok_url = match.group(1)
        else:
            ngrok_url = ngrok_url_str.replace('NgrokTunnel: "', '').split('"')[0]
    else:
        ngrok_url = ngrok_url_str
    
    print("\n" + "="*60)
    print("🌍 PUBLIC NGROK URL:")
    print(f"   {ngrok_url}")
    print("\n📋 Endpoints:")
    print(f"   Health: {ngrok_url}/health")
    print(f"   Predict: {ngrok_url}/predict")
    print("\n✅ Flask Backend Configuration:")
    print(f"   The backend is already configured to use this URL.")
    print(f"   If the URL changes, update line 1309 in combined_server.py:")
    print(f"   ALPHAFOLD2_NGROK_URL = \"{ngrok_url}\"")
    print("\n⚠️  IMPORTANT:")
    print("   • Keep this notebook cell running!")
    print("   • The Flask server will stop if you interrupt this cell")
    print("   • If you restart the runtime, run this cell again")
    print("="*60 + "\n")
    
except Exception as e:
    print(f"❌ Error setting up ngrok: {e}")
    print("   Make sure your ngrok auth token is correct")



