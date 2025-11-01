# Local Docking Setup Guide

## Overview

Molecular docking has been updated to run **locally** on your machine instead of requiring Google Colab. Docking uses CPU-based tools (RDKit, OpenBabel, AutoDock Vina) and does **not require GPU**.

## Why Local Docking?

- ✅ **Faster**: No network latency to Colab
- ✅ **More reliable**: No dependency on ngrok/Colab uptime
- ✅ **No GPU needed**: All tools are CPU-based
- ✅ **Better for pipeline**: Can run AlphaFold2 (Colab) → Docking (Local) seamlessly

## Dependencies Installation

### 1. Python Packages

Install Python dependencies:
```bash
pip install rdkit-pypi vina
```

Or update your `requirements.txt`:
```bash
pip install -r requirements.txt
```

### 2. OpenBabel (System Dependency)

OpenBabel is required for PDB to PDBQT conversion.

#### macOS (using Homebrew):
```bash
brew install openbabel
```

#### Linux (Ubuntu/Debian):
```bash
sudo apt-get update
sudo apt-get install -y openbabel
```

#### Windows:
1. Download installer from: https://openbabel.org/wiki/Category:Installation
2. Or use conda: `conda install -c conda-forge openbabel`

### 3. Verify Installation

Test that everything is installed:

```python
# Test RDKit
from rdkit import Chem
print("✅ RDKit installed")

# Test Vina
from vina import Vina
print("✅ Vina installed")

# Test OpenBabel
import subprocess
result = subprocess.run(["obabel", "--version"], capture_output=True, text=True)
if result.returncode == 0:
    print("✅ OpenBabel installed")
else:
    print("❌ OpenBabel not found")
```

## How It Works

1. **SMILES → 3D Structure**: RDKit converts SMILES string to 3D molecular structure
2. **PDB → PDBQT Conversion**: OpenBabel converts both ligand and protein PDB files to PDBQT format
3. **Docking**: AutoDock Vina performs the docking calculation (CPU-based)
4. **Results**: Returns binding affinities and docked poses

## Architecture

### Before (Colab):
```
Frontend → Flask Backend → ngrok → Colab Docking → Results
```

### Now (Local):
```
Frontend → Flask Backend → Local Docking (CPU) → Results
```

### Pipeline (Hybrid):
```
Frontend → Flask Backend → AlphaFold2 (Colab) → Rank 1 PDB → Local Docking (CPU) → Results
```

## Testing Local Docking

Test the endpoint:

```bash
curl -X POST http://localhost:5001/docking/run \
  -H "Content-Type: application/json" \
  -d '{
    "smiles": "C[C@H](N)C(=O)O",
    "protein_pdb": "ATOM      1  N   ALA A   1      20.154  16.967  10.000  1.00 85.00           N\n..."
  }'
```

## Fallback to Colab

If local dependencies are not installed, the system will automatically fall back to Colab (if `DOCKING_NGROK_URL` is configured). However, **local docking is recommended** for better performance and reliability.

## Performance Notes

- **Docking time**: 1-3 minutes per job (depending on protein size and exhaustiveness)
- **CPU usage**: Moderate (uses available CPU cores)
- **Memory**: ~500MB-2GB depending on protein size
- **No GPU needed**: All calculations are CPU-based

## Troubleshooting

### "OpenBabel (obabel) not found"
- Install OpenBabel system-wide (see above)
- Verify with: `obabel --version`
- Make sure it's in your PATH

### "Failed to parse SMILES string"
- Check your SMILES string is valid
- RDKit may have issues with unusual characters

### "Failed to convert to PDBQT"
- Ensure OpenBabel is properly installed
- Check file permissions in temp directory
- Verify PDB format is correct

### "No affinity scores found"
- Docking may have failed silently
- Check Vina logs
- Try increasing exhaustiveness (slower but more thorough)

## Comparison: Local vs Colab

| Feature | Local (Recommended) | Colab |
|---------|---------------------|-------|
| Speed | Fast (no network) | Slower (network latency) |
| Reliability | High | Depends on Colab/ngrok |
| GPU Required | ❌ No | ❌ No |
| Setup Complexity | Medium | High (ngrok setup) |
| Cost | Free | Free (Colab free tier) |
| Offline | ✅ Works offline | ❌ Needs internet |

## Next Steps

1. Install dependencies (see above)
2. Restart Flask backend
3. Test docking endpoint
4. Use "Complete Pipeline" tab - it will automatically use local docking

