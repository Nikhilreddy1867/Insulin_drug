"""
Flask server helper code to add to your AlphaFold2.ipynb notebook in Colab
Add this as a new cell after installing ColabFold dependencies
"""

# Install Flask and ngrok
!pip install flask flask-cors pyngrok

from flask import Flask, request, jsonify
from flask_cors import CORS
import threading
import os

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests

# Global variable to store results
prediction_results = {}

@app.route('/predict', methods=['POST'])
def predict_structure():
    """
    Endpoint for AlphaFold2 structure prediction
    Expects: {'sequence': 'MKTAYIAKQR...'}
    Returns: {'pdb_content': '...', 'plddt_score': 85.5, 'confidence_scores': {...}}
    """
    try:
        data = request.get_json()
        sequence = data.get('sequence', '').strip().upper()
        
        if not sequence:
            return jsonify({'error': 'Sequence is required'}), 400
        
        print(f"[Flask] Received prediction request for sequence length: {len(sequence)}")
        
        # Your AlphaFold2 prediction code here
        # This should match your notebook's prediction logic
        # Example structure:
        
        # 1. Set up queries
        jobname = f"prediction_{hash(sequence) % 10000}"
        queries_path = f"/content/{jobname}.fa"
        with open(queries_path, 'w') as f:
            f.write(f">query\n{sequence}\n")
        
        # 2. Run AlphaFold2 (use your existing notebook code)
        # ... (copy your prediction code from the notebook)
        
        # 3. Read results
        result_dir = jobname
        pdb_file = None
        for f in os.listdir(result_dir):
            if f.endswith('.pdb') and 'unrelaxed' in f:
                pdb_file = os.path.join(result_dir, f)
                break
        
        if pdb_file and os.path.exists(pdb_file):
            with open(pdb_file, 'r') as f:
                pdb_content = f.read()
            
            # Calculate pLDDT (you may need to extract this from results)
            plddt_score = 85.0  # Replace with actual calculation
            
            return jsonify({
                'pdb_content': pdb_content,
                'plddt_score': plddt_score,
                'confidence_scores': {},  # Add per-residue scores if available
                'jobname': jobname
            })
        else:
            return jsonify({'error': 'PDB file not found after prediction'}), 500
            
    except Exception as e:
        print(f"[Flask] Error in prediction: {e}")
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
# Get it from: https://dashboard.ngrok.com/get-started/your-authtoken
ngrok_token = os.environ.get('NGROK_AUTH_TOKEN', 'YOUR_TOKEN_HERE')
if ngrok_token != 'YOUR_TOKEN_HERE':
    ngrok.set_auth_token(ngrok_token)
    public_url = ngrok.connect(5000)
    print(f"🌐 Public ngrok URL: {public_url}")
    print(f"✅ Use this URL in Flask backend ALPHAFOLD2_NGROK_URL={public_url}")
else:
    print("⚠️  Set NGROK_AUTH_TOKEN environment variable or update the code above")


