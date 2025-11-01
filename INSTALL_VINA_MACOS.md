# Installing AutoDock Vina on macOS

## Status

- ✅ RDKit: Installed (`pip install rdkit`)
- ✅ OpenBabel: Installed (`brew install open-babel`)
- ⚠️  Vina: **Not available via Homebrew** (needs manual download)

## Option 1: Download Pre-built Binary (Easiest)

1. Visit: http://vina.scripps.edu/download.html

2. Download the macOS version:
   - For Apple Silicon (M1/M2): Look for `arm64` or `Apple Silicon` version
   - For Intel Mac: Look for `x86_64` or `Intel` version

3. Extract and install:
   ```bash
   # Extract the downloaded archive
   unzip vina_*.zip
   
   # Move to a location in PATH (e.g., /usr/local/bin or create ~/bin)
   sudo mv vina /usr/local/bin/
   # OR for Apple Silicon:
   sudo mv vina /opt/homebrew/bin/
   
   # Make executable
   chmod +x /usr/local/bin/vina
   
   # Verify
   vina --version
   ```

## Option 2: Use Colab Fallback (Current Setup)

The code is already configured to:
1. **Try local docking first** (if Vina is installed)
2. **Fall back to Colab** if Vina binary is not found

So you can:
- Use the system as-is (it will use Colab for docking)
- Install Vina later for faster local docking

## Option 3: Compile from Source (Advanced)

If you want to compile Vina yourself:

```bash
# Install dependencies
brew install boost eigen

# Clone Vina repository
git clone https://github.com/ccsb-scripps/AutoDock-Vina.git
cd AutoDock-Vina

# Build (adjust for your architecture)
mkdir build && cd build
cmake ..
make

# Install
sudo cp build/src/vina /usr/local/bin/
```

## Verification

After installation, verify:

```bash
vina --version
```

You should see something like:
```
AutoDock Vina 1.2.5
```

## Current Status

**You can use docking NOW** even without Vina installed:
- The system will automatically fall back to Colab docking
- Install Vina later for faster local docking

**Benefits of installing Vina locally:**
- ✅ Faster (no network latency)
- ✅ Works offline
- ✅ No dependency on Colab/ngrok

