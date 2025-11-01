"""
Flask server helper code to add to your Docking.ipynb notebook in Colab
Add this as a new cell after installing docking dependencies
"""

# Install Flask and ngrok
!pip install flask flask-cors pyngrok

from flask import Flask, request, jsonify
from flask_cors import CORS
import threading
import os
from rdkit import Chem
from rdkit.Chem import AllChem, rdmolfiles
from vina import Vina
import tempfile

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests

@app.route('/dock', methods=['POST'])
def dock_molecules():
    """
    Endpoint for molecular docking
    Expects: {
        'smiles': 'C[C@H](N)C(=O)O',
        'protein_sequence': 'MKTAYIAKQR...' OR
        'protein_pdb': 'ATOM   1  N   ...'
    }
    Returns: {
        'affinities': [-7.5, -7.2, -6.9, ...],
        'best_affinity': -7.5,
        'docked_poses': 5,
        'pdbqt_content': '...'
    }
    """
    try:
        data = request.get_json()
        smiles = data.get('smiles', '').strip()
        protein_sequence = data.get('protein_sequence', '').strip()
        protein_pdb = data.get('protein_pdb', '').strip()
        
        if not smiles:
            return jsonify({'error': 'SMILES string is required'}), 400
        
        if not protein_sequence and not protein_pdb:
            return jsonify({'error': 'Either protein_sequence or protein_pdb is required'}), 400
        
        print(f"[Flask] Received docking request for SMILES: {smiles[:50]}...")
        
        # Create temporary directory for files
        with tempfile.TemporaryDirectory() as tmpdir:
            # 1. Generate ligand 3D structure from SMILES
            print("Generating 3D structure for ligand...")
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return jsonify({'error': 'Failed to parse SMILES string'}), 400
            
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
                # If only sequence provided, you might need to get PDB from AlphaFold2 first
                # For now, return error suggesting to use PDB
                return jsonify({
                    'error': 'Please provide protein PDB file content. Use AlphaFold2 prediction result.'
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
            
            # Compute binding site automatically or use provided coordinates
            v.compute_vina_maps(center=[0, 0, 0], box_size=[25, 25, 25])
            
            # Perform docking
            v.dock(exhaustiveness=8, n_poses=5)
            
            # Save results
            docked_pdbqt = os.path.join(tmpdir, 'docked_out.pdbqt')
            v.write_poses(docked_pdbqt, n_poses=5, overwrite=True)
            
            # Get affinities
            scores = v.poses()
            if scores:
                affinities = [pose[1] for pose in scores]
            else:
                # Parse from file
                affinities = []
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
        print(f"[Flask] Error in docking: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'service': 'Docking'})

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
    print(f"✅ Use this URL in Flask backend DOCKING_NGROK_URL={public_url}")
else:
    print("⚠️  Set NGROK_AUTH_TOKEN environment variable or update the code above")


