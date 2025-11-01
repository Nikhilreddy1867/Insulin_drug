"""
Flask server helper code for Docking.ipynb notebook in Colab
Updated to work with rank 1 PDB from AlphaFold2
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

def parse_pdbqt_affinities(pdbqt_file):
    """Parse affinity scores from PDBQT file"""
    affinities = []
    try:
        with open(pdbqt_file, 'r') as f:
            for line in f:
                if line.startswith("REMARK VINA RESULT"):
                    parts = line.split()
                    if len(parts) > 3:
                        try:
                            affinity = float(parts[3])
                            affinities.append(affinity)
                        except ValueError:
                            continue
        return affinities
    except Exception as e:
        print(f"Error parsing {pdbqt_file}: {e}")
        return []

@app.route('/dock', methods=['POST'])
def dock_molecules():
    """
    Endpoint for molecular docking
    Expects: {
        'smiles': 'C[C@H](N)C(=O)O',
        'protein_pdb': 'ATOM   1  N   ...'  # Rank 1 PDB from AlphaFold2
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
        protein_pdb = data.get('protein_pdb', '').strip()  # Rank 1 PDB from AlphaFold2
        
        if not smiles:
            return jsonify({'error': 'SMILES string is required'}), 400
        
        if not protein_pdb:
            return jsonify({'error': 'Protein PDB content is required (use rank 1 PDB from AlphaFold2)'}), 400
        
        print(f"[Docking] Received docking request")
        print(f"[Docking] SMILES length: {len(smiles)}")
        print(f"[Docking] PDB content length: {len(protein_pdb)}")
        
        # Create temporary directory for files
        with tempfile.TemporaryDirectory() as tmpdir:
            # 1. Generate ligand 3D structure from SMILES
            print("[Docking] Generating 3D structure for ligand...")
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return jsonify({'error': 'Failed to parse SMILES string'}), 400
            
            mol = Chem.AddHs(mol)
            try:
                AllChem.EmbedMolecule(mol, AllChem.ETKDG())
                AllChem.UFFOptimizeMolecule(mol)
            except Exception as e:
                print(f"[Docking] Warning in embedding: {e}, continuing...")
            
            ligand_pdb = os.path.join(tmpdir, 'ligand.pdb')
            rdmolfiles.MolToPDBFile(mol, ligand_pdb)
            
            # Convert ligand to PDBQT
            print("[Docking] Converting ligand to PDBQT...")
            ligand_pdbqt = os.path.join(tmpdir, 'ligand.pdbqt')
            os.system(f"obabel {ligand_pdb} -O {ligand_pdbqt} -xh 2>/dev/null")
            
            if not os.path.exists(ligand_pdbqt):
                return jsonify({'error': 'Failed to convert ligand to PDBQT'}), 500
            
            # 2. Save protein PDB (rank 1 from AlphaFold2)
            print("[Docking] Processing protein PDB (rank 1 from AlphaFold2)...")
            protein_pdb_file = os.path.join(tmpdir, 'protein.pdb')
            with open(protein_pdb_file, 'w') as f:
                f.write(protein_pdb)
            
            # Convert protein to PDBQT
            print("[Docking] Converting protein to PDBQT...")
            protein_pdbqt = os.path.join(tmpdir, 'protein.pdbqt')
            os.system(f"obabel {protein_pdb_file} -O {protein_pdbqt} -xr 2>/dev/null")
            
            if not os.path.exists(protein_pdbqt):
                return jsonify({'error': 'Failed to convert protein to PDBQT'}), 500
            
            # 3. Run Vina docking
            print("[Docking] Running AutoDock Vina...")
            v = Vina(sf_name='vina')
            v.set_receptor(protein_pdbqt)
            v.set_ligand_from_file(ligand_pdbqt)
            
            # Compute binding site (center on protein, size 25x25x25)
            # For better results, you could calculate protein center from PDB
            v.compute_vina_maps(center=[0, 0, 0], box_size=[25, 25, 25])
            
            # Perform docking
            v.dock(exhaustiveness=8, n_poses=5)
            
            # Save results
            docked_pdbqt = os.path.join(tmpdir, 'docked_out.pdbqt')
            v.write_poses(docked_pdbqt, n_poses=5, overwrite=True)
            
            # Get affinities
            print("[Docking] Extracting affinity scores...")
            scores = v.poses()
            affinities = []
            if scores:
                try:
                    affinities = [pose[1] for pose in scores]
                    print(f"[Docking] Affinities from v.poses(): {affinities}")
                except (IndexError, TypeError) as e:
                    print(f"[Docking] Error with v.poses(): {e}, parsing from file...")
                    affinities = parse_pdbqt_affinities(docked_pdbqt)
            else:
                print("[Docking] No poses from v.poses(), parsing from file...")
                affinities = parse_pdbqt_affinities(docked_pdbqt)
            
            if not affinities:
                return jsonify({'error': 'No affinity scores found in docking results'}), 500
            
            # Read PDBQT content
            with open(docked_pdbqt, 'r') as f:
                pdbqt_content = f.read()
            
            print(f"[Docking] ✅ Docking completed! Best affinity: {min(affinities):.2f} kcal/mol")
            
            return jsonify({
                'affinities': affinities,
                'best_affinity': min(affinities),
                'docked_poses': len(affinities),
                'pdbqt_content': pdbqt_content
            })
            
    except Exception as e:
        print(f"[Docking] Error in docking: {e}")
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
    print(f"✅ Use this URL in Flask backend: DOCKING_NGROK_URL={public_url}")
else:
    print("⚠️  Set NGROK_AUTH_TOKEN environment variable or update the code above")

