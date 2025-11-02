# ============================================================================
# ALPHAFOLD2 COLAB NOTEBOOK INTEGRATION CODE
# Add this as a NEW CELL at the END of your ColabFold notebook
# ============================================================================

#@title Setup Flask Server & Extract Rank 1 PDB
# Install dependencies
!pip install -q flask flask-cors pyngrok

from flask import Flask, request, jsonify
from flask_cors import CORS
import threading
import os
import glob
import re
from pathlib import Path

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests

# Store global variables from prediction
# These will be set after prediction completes
prediction_results = None
prediction_jobname = None

def extract_rank1_pdb(jobname_path, jobname_val):
    """
    Extract rank 1 PDB file from prediction results
    Returns: (pdb_content, pdb_filename, plddt_score)
    """
    jobname_prefix = ".custom" if msa_mode == "custom" else ""
    
    # Method 1: Use results["rank"] to get rank 1 tag
    if prediction_results and "rank" in prediction_results and len(prediction_results["rank"]) > 0:
        try:
            tag = prediction_results["rank"][0][0]  # First rank (rank 1)
            pdb_filename = f"{jobname_path}/{jobname_val}{jobname_prefix}_unrelaxed_{tag}.pdb"
            if os.path.exists(pdb_filename):
                with open(pdb_filename, 'r') as f:
                    pdb_content = f.read()
                return pdb_content, os.path.basename(pdb_filename), None
        except Exception as e:
            print(f"[Extract] Method 1 failed: {e}")
    
    # Method 2: Search for unrelaxed PDB files and sort
    pdb_files = glob.glob(f"{jobname_path}/*unrelaxed*.pdb")
    if not pdb_files:
        pdb_files = glob.glob(f"{jobname_path}/*rank*1*.pdb")
    if not pdb_files:
        pdb_files = glob.glob(f"{jobname_path}/*.pdb")
    
    if pdb_files:
        # Sort to get rank 1 (usually first)
        pdb_files.sort()
        pdb_filename = pdb_files[0]
        with open(pdb_filename, 'r') as f:
            pdb_content = f.read()
        return pdb_content, os.path.basename(pdb_filename), None
    
    return None, None, None

def calculate_plddt(pdb_content):
    """Calculate average pLDDT score from PDB content"""
    if not pdb_content:
        return 85.0
    
    plddt_values = []
    for line in pdb_content.split('\n'):
        if line.startswith('ATOM'):
            try:
                # pLDDT is stored in B-factor column (60-66)
                b_factor = float(line[60:66].strip())
                if 0 <= b_factor <= 100:  # Valid pLDDT range
                    plddt_values.append(b_factor)
            except (ValueError, IndexError):
                continue
    
    return sum(plddt_values) / len(plddt_values) if plddt_values else 85.0

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
        'pdb_content': '...',  # Rank 1 PDB content as string
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
        
        # Validate sequence
        valid_amino_acids = set("ACDEFGHIKLMNPQRSTVWY")
        if not all(c in valid_amino_acids for c in sequence):
            return jsonify({'error': 'Invalid amino acid sequence'}), 400
        
        if len(sequence) < 10:
            return jsonify({'error': 'Sequence too short (minimum 10 amino acids)'}), 400
        
        print(f"[AlphaFold2] Starting prediction...")
        
        # Import ColabFold modules
        from colabfold.download import download_alphafold_params
        from colabfold.utils import setup_logging
        from colabfold.batch import get_queries, run, set_model_type
        import hashlib
        
        # Create jobname (same logic as notebook)
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
        model_type_val = set_model_type(is_complex, model_type)
        
        # Download parameters if needed
        download_alphafold_params(model_type_val, Path("."))
        
        print(f"[AlphaFold2] Running prediction (this may take 5-15 minutes)...")
        
        # Determine if cluster profile should be used
        use_cluster_profile = "multimer" not in model_type_val
        
        # Run prediction (using variables from notebook if available, otherwise defaults)
        try:
            num_recycles_val = num_recycles if 'num_recycles' in globals() and num_recycles is not None else 3
        except:
            num_recycles_val = 3
        
        try:
            recycle_tol_val = recycle_early_stop_tolerance if 'recycle_early_stop_tolerance' in globals() else None
        except:
            recycle_tol_val = None
        
        try:
            max_msa_val = max_msa if 'max_msa' in globals() else None
        except:
            max_msa_val = None
        
        results = run(
            queries=queries,
            result_dir=new_jobname,
            use_templates=use_templates if 'use_templates' in globals() else False,
            custom_template_path=custom_template_path if 'custom_template_path' in globals() else None,
            num_relax=num_relax if 'num_relax' in globals() else 0,
            msa_mode=msa_mode if 'msa_mode' in globals() else "mmseqs2_uniref_env",
            model_type=model_type_val,
            num_models=5,
            num_recycles=num_recycles_val,
            relax_max_iterations=relax_max_iterations if 'relax_max_iterations' in globals() else 200,
            recycle_early_stop_tolerance=recycle_tol_val,
            num_seeds=num_seeds if 'num_seeds' in globals() else 1,
            use_dropout=use_dropout if 'use_dropout' in globals() else False,
            model_order=[1,2,3,4,5],
            is_complex=is_complex,
            data_dir=Path("."),
            keep_existing_results=False,
            rank_by="auto",
            pair_mode=pair_mode if 'pair_mode' in globals() else "unpaired_paired",
            pairing_strategy=pairing_strategy if 'pairing_strategy' in globals() else "greedy",
            stop_at_score=float(100),
            dpi=dpi if 'dpi' in globals() else 200,
            zip_results=False,  # Don't create ZIP, we'll extract PDB directly
            save_all=save_all if 'save_all' in globals() else False,
            max_msa=max_msa_val,
            use_cluster_profile=use_cluster_profile,
            calc_extra_ptm=calc_extra_ptm if 'calc_extra_ptm' in globals() else False,
        )
        
        print(f"[AlphaFold2] Prediction completed! Extracting rank 1 PDB...")
        
        # Extract rank 1 PDB
        pdb_content, rank1_filename, _ = extract_rank1_pdb(new_jobname, new_jobname)
        
        if not pdb_content:
            return jsonify({'error': 'PDB file not found after prediction. Check logs for details.'}), 500
        
        # Calculate pLDDT score
        plddt_score = calculate_plddt(pdb_content)
        
        print(f"[AlphaFold2] ✅ Rank 1 PDB extracted: {rank1_filename}")
        print(f"[AlphaFold2] Average pLDDT: {plddt_score:.2f}")
        print(f"[AlphaFold2] ✅ SUCCESS! Returning results...")
        
        return jsonify({
            'pdb_content': pdb_content,
            'plddt_score': round(plddt_score, 2),
            'jobname': new_jobname,
            'rank1_file': rank1_filename or 'unknown'
        })
        
    except Exception as e:
        print(f"[AlphaFold2] ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

# Start Flask server
def run_flask():
    """Run Flask server in background"""
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)

print("Starting Flask server...")
flask_thread = threading.Thread(target=run_flask, daemon=True)
flask_thread.start()
print("✅ Flask server started on port 5000")

# Setup ngrok
from pyngrok import ngrok

# ⚠️ IMPORTANT: Replace with your ngrok auth token
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
    print("⚠️  WARNING: Set your NGROK_AUTH_TOKEN in the code above!")
    print("   Get your token from: https://dashboard.ngrok.com/get-started/your-authtoken")
    print("   Once set, the ngrok tunnel will start automatically")

