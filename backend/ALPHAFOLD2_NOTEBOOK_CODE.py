# =============================================
# 🔹 STEP 1 — Install dependencies
# =============================================
!pip install flask flask-cors pyngrok biopython matplotlib colabfold[alphafold-minus-jax] --quiet

import os, re, hashlib, random, glob, warnings, threading
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
    Returns: {'pdb_content': '...', 'plddt_score': 85.5, 'jobname': '...'}
    """
    try:
        data = request.get_json()
        sequence = data.get('sequence', '').strip().upper()
        if not sequence:
            return jsonify({'error': 'Protein sequence required'}), 400

        # Validate sequence
        valid_amino_acids = set("ACDEFGHIKLMNPQRSTVWY")
        if not all(c in valid_amino_acids for c in sequence):
            return jsonify({'error': 'Invalid amino acid sequence'}), 400

        print(f"[AlphaFold] Received sequence of length {len(sequence)}")

        # --- Prepare job folder ---
        jobname = "job_" + hashlib.sha1(sequence.encode()).hexdigest()[:8]
        os.makedirs(jobname, exist_ok=True)

        queries_path = os.path.join(jobname, f"{jobname}.csv")
        with open(queries_path, "w") as f:
            f.write(f"id,sequence\n{jobname},{sequence}")

        print(f"[AlphaFold] Starting prediction for job: {jobname}")

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

        print(f"[AlphaFold] Running AlphaFold2 prediction (this may take 5-15 minutes)...")
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

        print(f"[AlphaFold] Prediction completed, searching for PDB files...")

        # --- Find output PDB ---
        pdb_files = glob.glob(f"{jobname}/*unrelaxed*.pdb")
        if not pdb_files:
            print(f"[AlphaFold] ERROR: No PDB file found in {jobname}")
            return jsonify({'error': 'No PDB file generated. Check logs for details.'}), 500

        pdb_file = pdb_files[0]
        print(f"[AlphaFold] Found PDB file: {pdb_file}")
        
        with open(pdb_file, 'r') as f:
            pdb_content = f.read()

        # --- Extract pLDDT score from results or PDB file ---
        plddt_score = 85.0  # Default
        
        # Try to extract from results
        if 'rank' in results and len(results['rank']) > 0:
            try:
                # pLDDT score is usually in the ranking results
                plddt_score = float(results['rank'][0].get('plddt', 85.0))
            except:
                pass

        # Also try to calculate from PDB file (average of pLDDT values)
        try:
            plddt_values = []
            for line in pdb_content.split('\n'):
                if line.startswith('ATOM'):
                    # pLDDT is typically in the B-factor column (column 60-66)
                    try:
                        b_factor = float(line[60:66].strip())
                        if 0 <= b_factor <= 100:  # Valid pLDDT range
                            plddt_values.append(b_factor)
                    except:
                        continue
            if plddt_values:
                plddt_score = sum(plddt_values) / len(plddt_values)
        except Exception as e:
            print(f"[AlphaFold] Warning: Could not extract pLDDT from PDB: {e}")

        print(f"[AlphaFold] Success! pLDDT score: {plddt_score:.2f}")

        # --- Return result in format expected by frontend ---
        return jsonify({
            'pdb_content': pdb_content,
            'plddt_score': round(plddt_score, 2),
            'jobname': jobname,
            'confidence_scores': {}  # Can be populated if needed
        })

    except Exception as e:
        import traceback
        error_msg = str(e)
        traceback.print_exc()
        print(f"[AlphaFold] ERROR: {error_msg}")
        return jsonify({'error': error_msg}), 500

# =============================================
# 🔹 STEP 3 — Run Flask in background
# =============================================
def run_flask():
    app.run(host='0.0.0.0', port=5000, debug=False)

print("\n🚀 Starting Flask server in background...")
flask_thread = threading.Thread(target=run_flask, daemon=True)
flask_thread.start()
print("✅ Flask server started on port 5000")
print("ℹ️  Keep this cell running to maintain the service!\n")

# =============================================
# 🔹 STEP 4 — Setup ngrok
# =============================================
NGROK_AUTH_TOKEN = "34rVSQJ2Qtvmo53ShGFusPikqyF_5BVRzLan4Tqhuw3tfLtdH"  # Your token
ngrok.set_auth_token(NGROK_AUTH_TOKEN)

public_url = ngrok.connect(5000)
ngrok_url = str(public_url).replace('NgrokTunnel: "', '').replace('" -> "http://localhost:5000"', '')
print(f"\n{'='*60}")
print(f"🌍 PUBLIC NGROK URL: {ngrok_url}")
print(f"✅ Endpoint: {ngrok_url}/predict")
print(f"✅ Health check: {ngrok_url}/health")
print(f"{'='*60}")
print(f"\n📋 Copy this URL and update your Flask backend:")
print(f"   ALPHAFOLD2_NGROK_URL = \"{ngrok_url}\"")
print(f"\n⚠️  IMPORTANT: Keep this notebook cell running!")
print(f"   The Flask server will stop if you interrupt or restart this cell.\n")


