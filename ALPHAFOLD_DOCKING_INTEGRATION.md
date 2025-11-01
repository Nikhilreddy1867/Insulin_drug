# AlphaFold2 & Docking Integration Guide

## Overview

The application now supports two additional features:
1. **AlphaFold2 Structure Prediction** - Predict 3D protein structures
2. **Molecular Docking** - Dock drug molecules (SMILES) to protein structures

Both features run in Google Colab notebooks exposed via ngrok.

## Architecture

```
Frontend (React)
    ↓
Flask Backend (localhost:5001)
    ↓
Ngrok Tunnel
    ↓
Google Colab Notebook (Flask server on port 5000)
    ↓
AlphaFold2 / Docking Processing
```

## Setup Steps

### 1. Setup AlphaFold2 Notebook

1. Open `AlphaFold2 .ipynb` in Google Colab
2. Add the helper code from `colab_flask_helper_alphafold2.py` as a new cell
3. Update with your ngrok auth token
4. Run the notebook - it will start Flask server and create ngrok URL
5. Copy the ngrok URL (e.g., `https://xxxx-xxxx-xxxx.ngrok-free.app`)

### 2. Setup Docking Notebook

1. Open `Docking.ipynb` in Google Colab
2. Add the helper code from `colab_flask_helper_docking.py` as a new cell
3. Update with your ngrok auth token
4. Run the notebook - it will start Flask server and create ngrok URL
5. Copy the ngrok URL (e.g., `https://yyyy-yyyy-yyyy.ngrok-free.app`)

### 3. Configure Flask Backend

Update `backend/combined_server.py` or set environment variables:

```python
# In combined_server.py or .env file
ALPHAFOLD2_NGROK_URL = "https://xxxx-xxxx-xxxx.ngrok-free.app"
DOCKING_NGROK_URL = "https://yyyy-yyyy-yyyy.ngrok-free.app"
```

### 4. Restart Flask Backend

```bash
cd backend
python combined_server.py
```

## API Endpoints

### AlphaFold2
- **POST** `/alphafold2/predict`
  - Body: `{"sequence": "MKTAYIAKQR..."}`
  - Returns: `{"success": true, "result": {"pdb_content": "...", "plddt_score": 85.5}}`

### Docking
- **POST** `/docking/run`
  - Body: `{"smiles": "C[C@H](N)C(=O)O", "protein_pdb": "ATOM 1 ..."}`
  - Returns: `{"success": true, "result": {"affinities": [-7.5, -7.2], "best_affinity": -7.5}}`

## Frontend Pages

### AlphaFold2Page (`/alphafold2`)
- Input: Protein sequence
- Output: PDB file, confidence scores, visualization
- Download: PDB file

### DockingPage (`/docking`)
- Input: SMILES string + Protein PDB (from AlphaFold2) or sequence
- Output: Binding affinities, docked poses, PDBQT file
- Download: PDBQT file with docked conformations

## Workflow

1. **Generate SMILES** → Dashboard → SMILES Generation tab → Get SMILES string
2. **Predict Structure** → AlphaFold2 page → Enter sequence → Get PDB file
3. **Run Docking** → Docking page → Enter SMILES + PDB → Get binding affinities

## Navigation

New links added to navbar:
- **AlphaFold2** - Structure prediction
- **Docking** - Molecular docking

## Notes

- Keep Colab notebooks running while using the features
- Ngrok free tier URLs change on restart (update backend config)
- AlphaFold2 predictions can take 5-10+ minutes
- Docking typically takes 1-3 minutes
- Both pages have the same gradient background as homepage


