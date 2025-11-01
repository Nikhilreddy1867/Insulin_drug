# Quick Install Guide: Docking Dependencies

## ✅ Installation Complete Status

### 1. RDKit ✅
```bash
# Already installed!
pip install rdkit
```

### 2. OpenBabel ✅
```bash
# Already installed!
brew install open-babel

# Location: /opt/homebrew/bin/obabel
```

### 3. AutoDock Vina ⚠️
```bash
# Install the binary (not Python package)
brew install vina

# OR download from: http://vina.scripps.edu/download.html
```

## Quick Verification

Run this to check all dependencies:

```bash
cd /Users/nikhilreddymallela/Downloads/Insulindrug/Insulin_Drug_Synthesis
source venv/bin/activate

python -c "
from rdkit import Chem
print('✅ RDKit: OK')

import subprocess
r = subprocess.run(['/opt/homebrew/bin/obabel', '--version'], capture_output=True)
if r.returncode == 0:
    print('✅ OpenBabel: OK')
else:
    print('❌ OpenBabel: Not found')

r2 = subprocess.run(['vina', '--version'], capture_output=True)
if r2.returncode == 0:
    print('✅ Vina: OK')
else:
    print('⚠️  Vina: Install with: brew install vina')
"
```

## What Changed?

- **RDKit**: Using Python package (`pip install rdkit`) ✅
- **OpenBabel**: Using system binary (`brew install open-babel`) ✅
- **Vina**: Using system binary (`brew install vina`) instead of Python package (easier to install)

The code now uses:
- RDKit Python library for SMILES → 3D conversion
- OpenBabel binary (`/opt/homebrew/bin/obabel`) for PDB → PDBQT conversion
- Vina binary for docking calculations

This is more reliable and easier to install!

