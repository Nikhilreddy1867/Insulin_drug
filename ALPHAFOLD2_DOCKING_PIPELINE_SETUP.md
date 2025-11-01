# AlphaFold2 → Docking Complete Pipeline Setup

This document explains how to set up and use the automated pipeline that runs AlphaFold2 prediction, extracts rank 1 PDB from ZIP, and performs molecular docking.

## Pipeline Overview

The pipeline automates the following steps:

1. **AlphaFold2 Prediction**: Takes protein sequence → runs ColabFold → generates ZIP file with PDB structures
2. **Rank 1 Extraction**: Automatically extracts `*unrelaxed_rank_001*.pdb` from the ZIP file
3. **Docking**: Uses the rank 1 PDB with SMILES string to run AutoDock Vina docking

## Setup Instructions

### Step 1: Update AlphaFold2 Colab Notebook

1. Open your AlphaFold2 Colab notebook: https://colab.research.google.com/github/sokrypton/ColabFold/blob/main/AlphaFold2.ipynb

2. After installing ColabFold dependencies, add a new cell and paste the contents of:
   ```
   backend/colab_flask_helper_alphafold2_fixed.py
   ```

3. **Important Changes**:
   - The code now enables `zip_results=True` in the ColabFold `run()` function
   - It automatically extracts rank 1 PDB (`*unrelaxed_rank_001*.pdb`) from the generated ZIP
   - The `/predict` endpoint returns the rank 1 PDB content directly (no manual ZIP download needed)

4. Set your ngrok auth token:
   ```python
   import os
   os.environ['NGROK_AUTH_TOKEN'] = 'your_ngrok_token_here'
   ```

5. Run the cell and copy the ngrok URL (e.g., `https://xxxx-xx-xx-xxx-xxx.ngrok-free.app`)

6. Update your `.env` file or Flask backend:
   ```
   ALPHAFOLD2_NGROK_URL=https://xxxx-xx-xx-xxx-xxx.ngrok-free.app
   ```

### Step 2: Update Docking Colab Notebook

1. Create a new Colab notebook for Docking

2. Install dependencies:
   ```python
   !apt-get install -y openbabel
   !pip install --no-build-isolation pandas==2.2.2 rdkit biopandas py3Dmol vina flask flask-cors pyngrok
   ```

3. Add a new cell and paste the contents of:
   ```
   backend/colab_flask_helper_docking_fixed.py
   ```

4. **Important**: The docking endpoint now accepts:
   - `smiles`: SMILES string of the drug molecule
   - `protein_pdb`: Rank 1 PDB content from AlphaFold2 (directly, not a file path)

5. Set your ngrok auth token and run to get the ngrok URL

6. Update `.env`:
   ```
   DOCKING_NGROK_URL=https://yyyy-yy-yy-yyy-yyy.ngrok-free.app
   ```

### Step 3: Verify Backend Endpoint

The Flask backend now includes a combined pipeline endpoint:

**Endpoint**: `POST /pipeline/alphafold2-to-docking`

**Request**:
```json
{
  "protein_sequence": "MKTAYIAKQR...",
  "smiles": "C[C@H](N)C(=O)O"
}
```

**Response**:
```json
{
  "success": true,
  "protein_sequence": "...",
  "smiles": "...",
  "alphafold2": {
    "pdb_content": "...",
    "plddt_score": 85.5,
    "jobname": "...",
    "rank1_file": "..."
  },
  "docking": {
    "affinities": [-7.5, -7.2, ...],
    "best_affinity": -7.5,
    "docked_poses": 5,
    "pdbqt_content": "..."
  },
  "pipeline_status": "complete"
}
```

## Using the Pipeline

### Via Frontend (Recommended)

1. Navigate to the **"Complete Pipeline"** tab in the dashboard
2. Enter:
   - **Protein Sequence**: The sequence for AlphaFold2 prediction
   - **SMILES String**: The drug molecule SMILES (can be from SMILES Generation tab)
3. Click **"Run Complete Pipeline"**
4. Wait 15-20 minutes (process can be long)
5. Results include:
   - AlphaFold2 rank 1 PDB (downloadable)
   - Docking affinities and poses (downloadable)

### Via API

```bash
curl -X POST http://localhost:5001/pipeline/alphafold2-to-docking \
  -H "Content-Type: application/json" \
  -d '{
    "protein_sequence": "MKTAYIAKQR",
    "smiles": "C[C@H](N)C(=O)O"
  }'
```

## How It Works

1. **AlphaFold2 Step**:
   - Receives protein sequence
   - Calls ColabFold AlphaFold2 notebook
   - ColabFold generates ZIP file with multiple PDB structures
   - Flask helper extracts rank 1 PDB from ZIP automatically
   - Returns rank 1 PDB content

2. **Docking Step**:
   - Receives rank 1 PDB content (from AlphaFold2) and SMILES string
   - Converts SMILES to 3D structure using RDKit
   - Converts both to PDBQT format using OpenBabel
   - Runs AutoDock Vina docking
   - Returns binding affinities and docked poses

## Troubleshooting

### AlphaFold2 returns error about ZIP
- Ensure `zip_results=True` is set in the ColabFold `run()` call
- Check that the ZIP file is generated in the output directory

### Rank 1 PDB not found
- The code tries multiple patterns: `*unrelaxed_rank_001*.pdb`, `*rank_1*.pdb`
- If no rank 1 found, it falls back to the first PDB file in the ZIP
- Check Colab notebook logs for which file was extracted

### Docking fails with "protein PDB required"
- Ensure AlphaFold2 step completed successfully
- Check that `pdb_content` is in the AlphaFold2 response

### Pipeline timeout
- AlphaFold2 can take 5-15 minutes per sequence
- Docking typically takes 1-3 minutes
- Total pipeline time: 15-20 minutes for typical sequences
- Increase timeout in frontend if needed (currently 20 minutes)

## Notes

- The pipeline automatically handles rank 1 PDB extraction from ZIP - no manual download needed
- Both AlphaFold2 and Docking run on external Colab services (not local)
- Device info in responses shows the Flask backend device (not Colab GPU)
- Partial success: If docking fails, AlphaFold2 results are still returned

