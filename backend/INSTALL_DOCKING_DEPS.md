# Installing Docking Dependencies

## Important: RDKit Installation

RDKit doesn't have a simple `pip install` package for all Python versions. Here are the recommended installation methods:

### Option 1: Using Conda (Recommended for RDKit)

If you have conda/miniconda:

```bash
# Create conda environment (if needed)
conda create -n insulindrug python=3.10
conda activate insulindrug

# Install RDKit from conda-forge
conda install -c conda-forge rdkit

# Install other dependencies
pip install vina flask flask-cors torch numpy joblib transformers python-dotenv pymongo twilio bcrypt
```

### Option 2: Using pip (Python 3.8-3.11)

For Python 3.8-3.11:

```bash
pip install rdkit vina
```

**Note**: RDKit wheels may not be available for Python 3.12+ via pip. Consider using Python 3.10 or 3.11, or use conda.

### Option 3: Install from Source (Advanced)

If pip doesn't work:

```bash
# Install dependencies first
pip install numpy boost cmake

# Clone and build RDKit
git clone https://github.com/rdkit/rdkit.git
cd rdkit
mkdir build && cd build
cmake -DRDK_BUILD_PYTHON_WRAPPERS=ON ..
make
make install
```

### Option 4: Use Pre-built Wheels

For macOS (Apple Silicon):

```bash
pip install https://github.com/rdkit/rdkit/releases/download/Release_2023_09_1/rdkit-2023.9.1-cp310-cp310-macosx_11_0_arm64.whl
```

For macOS (Intel):

```bash
pip install https://github.com/rdkit/rdkit/releases/download/Release_2023_09_1/rdkit-2023.9.1-cp310-cp310-macosx_10_9_x86_64.whl
```

## OpenBabel Installation

### macOS (using Homebrew):

```bash
brew install open-babel
```

**Note**: The package name is `open-babel` (with hyphen), not `openbabel`.

After installation, verify:
```bash
obabel --version
```

If `obabel` command is not found, add it to PATH:
```bash
# For Homebrew on Apple Silicon
export PATH="/opt/homebrew/bin:$PATH"

# For Homebrew on Intel
export PATH="/usr/local/bin:$PATH"
```

### Linux (Ubuntu/Debian):

```bash
sudo apt-get update
sudo apt-get install -y openbabel
```

### Verify Installation

Test that everything works:

```python
# Test RDKit
try:
    from rdkit import Chem
    print("✅ RDKit installed")
except ImportError:
    print("❌ RDKit not found")

# Test Vina
try:
    from vina import Vina
    print("✅ Vina installed")
except ImportError:
    print("❌ Vina not found")

# Test OpenBabel
import subprocess
result = subprocess.run(["obabel", "--version"], capture_output=True, text=True)
if result.returncode == 0:
    print("✅ OpenBabel installed")
    print(result.stdout)
else:
    print("❌ OpenBabel not found")
    print(result.stderr)
```

## Troubleshooting

### "ModuleNotFoundError: No module named 'rdkit'"

- Try using conda: `conda install -c conda-forge rdkit`
- Or downgrade to Python 3.10 or 3.11
- Or install from source (see Option 3 above)

### "obabel: command not found"

- Make sure `open-babel` is installed: `brew install open-babel`
- Add Homebrew to PATH if needed
- Restart your terminal after installation

### "Failed to convert ligand to PDBQT"

- Verify OpenBabel is in PATH: `which obabel`
- Check OpenBabel version: `obabel --version`

## Quick Install Script (macOS)

```bash
#!/bin/bash
# Install docking dependencies on macOS

# Install OpenBabel
echo "Installing OpenBabel..."
brew install open-babel

# Install Vina
echo "Installing Vina..."
pip3 install vina

# Install RDKit (prefer conda)
echo "Installing RDKit..."
if command -v conda &> /dev/null; then
    conda install -c conda-forge rdkit -y
else
    echo "⚠️  RDKit installation:"
    echo "   Option 1: Install conda and run: conda install -c conda-forge rdkit"
    echo "   Option 2: Use Python 3.10/3.11 and run: pip install rdkit"
    echo "   Option 3: Install from source"
fi

# Verify
echo "Verifying installations..."
python3 -c "from vina import Vina; print('✅ Vina OK')" 2>/dev/null || echo "❌ Vina failed"
obabel --version 2>/dev/null && echo "✅ OpenBabel OK" || echo "❌ OpenBabel not in PATH"
```

