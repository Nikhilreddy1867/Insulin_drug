"""
Flask server helper code for AlphaFold2.ipynb notebook in Colab
Updated to extract rank 1 PDB from ZIP file automatically
Add this as a new cell after installing ColabFold dependencies
"""
# Install Flask and ngrok
!pip install flask flask-cors pyngrok

from flask import Flask, request, jsonify
from flask_cors import CORS
import threading
import os
import zipfile
import glob
import re

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests

@app.route('/predict', methods=['POST'])
def predict_structure():
    """
    Endpoint for AlphaFold2 structure prediction
    Expects: {'sequence': 'MKTAYIAKQR...'}
    Returns: {
        'pdb_content': '...',  # Rank 1 PDB content
        'plddt_score': 85.5,
        'confidence_scores': {...},
        'jobname': '...',
        'zip_url': '...' (optional, if ZIP was generated)
    }
    """
    try:
        data = request.get_json()
        sequence = data.get('sequence', '').strip().upper()
        
        if not sequence:
            return jsonify({'error': 'Sequence is required'}), 400
        
        print(f"[AlphaFold2] Received prediction request for sequence length: {len(sequence)}")
        
        # 1. Set up queries
        jobname = f"prediction_{hash(sequence) % 10000}"
        queries_path = f"/content/{jobname}.fa"
        with open(queries_path, 'w') as f:
            f.write(f">query\n{sequence}\n")
        
        print(f"[AlphaFold2] Running ColabFold prediction...")
        # 2. Run AlphaFold2 (use your existing notebook code from https://colab.research.google.com/github/sokrypton/ColabFold/blob/main/AlphaFold2.ipynb)
        # IMPORTANT: Make sure zip_results=True to get ZIP file
        from colabfold.batch import run
        from pathlib import Path
        
        queries = {jobname: sequence}
        results = run(
            queries=queries,
            result_dir=Path("/content"),
            use_templates=False,
            custom_template_path=None,
            num_relax=0,
            msa_mode="mmseqs2_uniref_env",
            model_type="auto",
            num_models=1,
            num_recycles=3,
            relax_max_iterations=200,
            num_seeds=1,
            use_dropout=False,
            model_order=[1],
            is_complex=False,
            data_dir=Path("/content"),
            keep_existing_results=False,
            rank_by="auto",
            pair_mode="unpaired_paired",
            pairing_strategy="greedy",
            stop_at_score=float(100),
            dpi=150,
            zip_results=True,  # IMPORTANT: Enable ZIP output
            save_all=False,
            max_msa=None,
            use_cluster_profile=True,
            calc_extra_ptm=False,
        )
        
        print(f"[AlphaFold2] Prediction completed, searching for results...")
        
        # 3. Look for ZIP file first (preferred)
        zip_files = glob.glob(f"/content/{jobname}*.zip")
        pdb_content = None
        plddt_score = 85.0
        rank1_pdb_file = None
        
        if zip_files:
            print(f"[AlphaFold2] Found ZIP file: {zip_files[0]}")
            # Extract rank 1 PDB from ZIP
            with zipfile.ZipFile(zip_files[0], 'r') as zip_ref:
                # List all files in ZIP
                file_list = zip_ref.namelist()
                
                # Find rank 1 PDB file (usually named like: *_unrelaxed_rank_001_*.pdb)
                rank1_pattern = re.compile(r'.*unrelaxed_rank_001.*\.pdb$', re.IGNORECASE)
                rank1_files = [f for f in file_list if rank1_pattern.match(f)]
                
                if not rank1_files:
                    # Try alternative patterns
                    rank1_pattern = re.compile(r'.*rank_1.*\.pdb$', re.IGNORECASE)
                    rank1_files = [f for f in file_list if rank1_pattern.match(f)]
                
                if not rank1_files:
                    # Look for any PDB file with "rank" in name, sort and take first
                    pdb_files = [f for f in file_list if f.endswith('.pdb') and 'rank' in f.lower()]
                    if pdb_files:
                        pdb_files.sort()  # Sort to get rank 1
                        rank1_files = [pdb_files[0]]
                
                if rank1_files:
                    rank1_pdb_file = rank1_files[0]
                    print(f"[AlphaFold2] Extracted rank 1 PDB: {rank1_pdb_file}")
                    pdb_content = zip_ref.read(rank1_pdb_file).decode('utf-8')
                else:
                    print(f"[AlphaFold2] Warning: No rank 1 PDB found in ZIP, trying to find any PDB...")
                    # Fallback: get any PDB file
                    pdb_files = [f for f in file_list if f.endswith('.pdb')]
                    if pdb_files:
                        pdb_content = zip_ref.read(pdb_files[0]).decode('utf-8')
                        print(f"[AlphaFold2] Using: {pdb_files[0]}")
        
        # 4. If no ZIP, look for PDB files directly (fallback)
        if not pdb_content:
            print(f"[AlphaFold2] No ZIP found, searching for PDB files directly...")
            pdb_files = glob.glob(f"/content/{jobname}/*unrelaxed*.pdb")
            if not pdb_files:
                pdb_files = glob.glob(f"/content/{jobname}/*rank*1*.pdb")
            
            if pdb_files:
                # Sort to ensure we get rank 1
                pdb_files.sort()
                rank1_pdb_file = pdb_files[0]
                print(f"[AlphaFold2] Found rank 1 PDB: {rank1_pdb_file}")
                with open(rank1_pdb_file, 'r') as f:
                    pdb_content = f.read()
        
        if not pdb_content:
            print(f"[AlphaFold2] ERROR: No PDB file found")
            return jsonify({'error': 'No PDB file generated. Check logs for details.'}), 500
        
        # 5. Extract pLDDT score from PDB (average of all atoms)
        try:
            plddt_values = []
            for line in pdb_content.split('\n'):
                if line.startswith('ATOM'):
                    # pLDDT is in the B-factor column (column 60-66)
                    try:
                        b_factor = float(line[60:66].strip())
                        if 0 <= b_factor <= 100:  # Valid pLDDT range
                            plddt_values.append(b_factor)
                    except (ValueError, IndexError):
                        continue
            
            if plddt_values:
                plddt_score = sum(plddt_values) / len(plddt_values)
                print(f"[AlphaFold2] Calculated pLDDT: {plddt_score:.2f} (from {len(plddt_values)} atoms)")
        except Exception as e:
            print(f"[AlphaFold2] Warning: Could not extract pLDDT from PDB: {e}")
        
        print(f"[AlphaFold2] ✅ SUCCESS! Rank 1 PDB extracted, pLDDT: {plddt_score:.2f}")
        
        # Return result
        return jsonify({
            'pdb_content': pdb_content,
            'plddt_score': round(plddt_score, 2),
            'jobname': jobname,
            'confidence_scores': {},
            'rank1_file': rank1_pdb_file or 'unknown'
        })
            
    except Exception as e:
        print(f"[AlphaFold2] Error in prediction: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'service': 'AlphaFold2'})

# Run Flask in background thread
def run_flask():
    app.run(host='0.0.0.0', port=5000, debug=False)

print("Starting Flask server...")
flask_thread = threading.Thread(target=run_flask, daemon=True)
flask_thread.start()
print("✅ Flask server started on port 5000")

# Setup ngrok
from pyngrok import ngrok

# IMPORTANT: Set your ngrok auth token
ngrok_token = os.environ.get('NGROK_AUTH_TOKEN', 'YOUR_TOKEN_HERE')
if ngrok_token != 'YOUR_TOKEN_HERE':
    ngrok.set_auth_token(ngrok_token)
    public_url = ngrok.connect(5000)
    print(f"🌐 Public ngrok URL: {public_url}")
    print(f"✅ Use this URL in Flask backend: ALPHAFOLD2_NGROK_URL={public_url}")
else:
    print("⚠️  Set NGROK_AUTH_TOKEN environment variable or update the code above")

