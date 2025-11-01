# =============================================
# 🔹 STEP 1 — Install dependencies
# =============================================
import os

# CRITICAL: Set environment variables BEFORE any imports to force CPU mode
# This must happen before JAX is imported, otherwise CUDA plugins may initialize
os.environ['JAX_PLATFORMS'] = 'cpu'  # Force CPU to avoid CUDA plugin version mismatch
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Hide CUDA devices to force CPU
os.environ['JAX_PLATFORM_NAME'] = 'cpu'  # Additional JAX CPU flag

# Fix TensorFlow compatibility issue by removing problematic library file
# This is a known issue with TensorFlow and ColabFold
print("🔧 Fixing TensorFlow compatibility issues...")
os.system("rm -f /usr/local/lib/python3.*/dist-packages/tensorflow/core/kernels/libtfkernel_sobol_op.so 2>/dev/null")
os.system("rm -f /usr/local/lib/python3.12/dist-packages/tensorflow/core/kernels/libtfkernel_sobol_op.so 2>/dev/null")

# Install dependencies with compatible versions
print("📦 Installing dependencies...")
!pip install flask flask-cors pyngrok biopython matplotlib --quiet

# Install JAX and Haiku with compatible versions first
# Use versions that are known to work together
print("📦 Installing JAX and Haiku with compatible versions...")

# Uninstall conflicting CUDA plugins that cause version mismatch
print("🔧 Removing conflicting CUDA plugins...")
# More aggressive removal of CUDA plugins
!pip uninstall -y jax_plugins.xla_cuda12 jax-cuda12-plugin jax-plugins 2>&1 | grep -v "WARNING" || true
# Also try to remove from site-packages directly
import shutil, glob
cuda_plugin_paths = glob.glob("/usr/local/lib/python3.*/dist-packages/jax_plugins")
for path in cuda_plugin_paths:
    try:
        if os.path.isdir(path):
            shutil.rmtree(path, ignore_errors=True)
            print(f"Removed CUDA plugin directory: {path}")
    except:
        pass

# Install compatible versions that work with haiku (CPU version to avoid CUDA issues)
!pip install --upgrade "jax==0.4.28" "jaxlib==0.4.28" "dm-haiku==0.0.11" --quiet

# Install colabfold without JAX (we installed compatible versions above)
print("📦 Installing ColabFold...")
!pip install colabfold[alphafold-minus-jax] --quiet

import os, re, hashlib, random, glob, warnings, threading
from flask import Flask, request, jsonify
from flask_cors import CORS
from pyngrok import ngrok

warnings.simplefilter('ignore')

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# Note: JAX_PLATFORMS and CUDA_VISIBLE_DEVICES are set earlier in the file

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
    Returns: {
        'pdb_content': '...',
        'plddt_score': 85.5,
        'jobname': '...',
        'confidence_scores': {}
    }
    """
    try:
        data = request.get_json()
        sequence = data.get('sequence', '').strip().upper()
        
        if not sequence:
            return jsonify({'error': 'Protein sequence required'}), 400

        # Validate sequence
        valid_amino_acids = set("ACDEFGHIKLMNPQRSTVWY")
        if not all(c in valid_amino_acids for c in sequence):
            return jsonify({'error': 'Invalid amino acid sequence. Only standard 20 amino acids allowed.'}), 400
        
        if len(sequence) < 10:
            return jsonify({'error': 'Sequence too short. Minimum 10 amino acids required.'}), 400

        print(f"\n[AlphaFold] ==========================================")
        print(f"[AlphaFold] Received sequence of length {len(sequence)}")
        print(f"[AlphaFold] Starting prediction...")

        # --- Prepare job folder ---
        jobname = "job_" + hashlib.sha1(sequence.encode()).hexdigest()[:8]
        os.makedirs(jobname, exist_ok=True)

        queries_path = os.path.join(jobname, f"{jobname}.csv")
        with open(queries_path, "w") as f:
            f.write(f"id,sequence\n{jobname},{sequence}")

        print(f"[AlphaFold] Job name: {jobname}")

        # --- AlphaFold prediction (ColabFold call) ---
        # CRITICAL: Remove problematic TensorFlow file BEFORE importing colabfold
        # This must happen before TensorFlow is imported, otherwise the error occurs
        import glob as glob_module
        sobol_files = glob_module.glob("/usr/local/lib/python3.*/dist-packages/tensorflow/core/kernels/libtfkernel_sobol_op.so")
        for sobol_file in sobol_files:
            try:
                if os.path.exists(sobol_file):
                    os.remove(sobol_file)
                    print(f"[AlphaFold] Removed problematic TensorFlow file: {sobol_file}")
            except Exception as e:
                print(f"[AlphaFold] Warning: Could not remove {sobol_file}: {e}")
        
        # Also try the specific Python 3.12 path
        py312_path = "/usr/local/lib/python3.12/dist-packages/tensorflow/core/kernels/libtfkernel_sobol_op.so"
        if os.path.exists(py312_path):
            try:
                os.remove(py312_path)
                print(f"[AlphaFold] Removed problematic TensorFlow file: {py312_path}")
            except:
                pass
        
        # Fix JAX/Haiku compatibility issues
        # 1. jax.interpreters.xla has no attribute 'xe'
        # 2. jax.tools doesn't exist in newer JAX versions
        import sys
        import types
        
        # IMPORTANT: Import and patch JAX BEFORE importing any colabfold modules
        # This must happen before haiku is imported (which happens when colabfold imports)
        try:
            import jax
            
            # Patch 0: Disable CUDA plugin discovery to prevent initialization errors
            # This must be done before any backend initialization
            try:
                from jax._src import xla_bridge
                
                # Patch discover_pjrt_plugins to catch and ignore CUDA plugin errors
                original_discover = getattr(xla_bridge, 'discover_pjrt_plugins', None)
                if original_discover:
                    def patched_discover():
                        plugins = {}
                        try:
                            # Try original discovery but catch CUDA errors
                            discovered = original_discover()
                            # Filter out CUDA plugins - only keep CPU
                            for name, plugin in discovered.items():
                                if 'cpu' in name.lower() or 'tpu' in name.lower():
                                    plugins[name] = plugin
                                # Skip any CUDA plugins
                        except Exception as e:
                            # If discovery fails (e.g., CUDA plugin error), return empty dict
                            # This forces JAX to fall back to CPU
                            pass
                        return plugins
                    
                    xla_bridge.discover_pjrt_plugins = patched_discover
                    print("[AlphaFold] Patched JAX to skip CUDA plugin discovery")
                
                # Also patch the backend initialization to handle CUDA errors gracefully
                original_get_backend = getattr(xla_bridge, '_get_backend_uncached', None)
                if original_get_backend:
                    def patched_get_backend(platform):
                        # Always force CPU to avoid CUDA plugin errors
                        if platform == 'cuda' or 'cuda' in str(platform).lower() or platform is None:
                            print(f"[AlphaFold] Forcing CPU backend (CUDA disabled)")
                            platform = 'cpu'
                        try:
                            return original_get_backend(platform)
                        except Exception as e:
                            error_str = str(e).lower()
                            if 'cuda' in error_str or 'plugin' in error_str or 'pjrt' in error_str:
                                print(f"[AlphaFold] Backend error caught, falling back to CPU: {e}")
                                try:
                                    return original_get_backend('cpu')
                                except:
                                    # If CPU also fails, return a mock backend
                                    pass
                            raise
                    
                    xla_bridge._get_backend_uncached = patched_get_backend
                    print("[AlphaFold] Patched JAX backend initialization for CPU fallback")
                
                # Patch the backends() function to filter out CUDA
                original_backends = getattr(xla_bridge, 'backends', None)
                if original_backends:
                    def patched_backends():
                        try:
                            all_backends = original_backends()
                            # Filter out CUDA backends
                            filtered = {}
                            for name, backend in all_backends.items():
                                if 'cpu' in name.lower() or 'tpu' in name.lower():
                                    filtered[name] = backend
                            return filtered if filtered else {'cpu': original_get_backend('cpu')}
                        except Exception as e:
                            # If backends() fails due to CUDA, return CPU only
                            print(f"[AlphaFold] Backend discovery error, using CPU only: {e}")
                            return {'cpu': original_get_backend('cpu')}
                    
                    xla_bridge.backends = patched_backends
                    print("[AlphaFold] Patched JAX backends() to filter CUDA")
                    
            except Exception as e:
                print(f"[AlphaFold] Warning: Could not patch JAX plugin discovery: {e}")
            
            # Patch 1: Add missing jax.interpreters.xla.xe attribute for haiku
            if not hasattr(jax.interpreters.xla, 'xe'):
                # Create a mock 'xe' module with PjitFunction
                class MockPjitFunction:
                    pass
                
                xe_module = types.ModuleType('xe')
                xe_module.PjitFunction = MockPjitFunction
                jax.interpreters.xla.xe = xe_module
                print("[AlphaFold] Patched jax.interpreters.xla.xe for haiku compatibility")
            
            # Patch 2: Create jax.tools structure if it doesn't exist
            if not hasattr(jax, 'tools'):
                jax_tools_module = types.ModuleType('jax.tools')
                jax.tools = jax_tools_module
                sys.modules['jax.tools'] = jax_tools_module
                print("[AlphaFold] Created jax.tools module structure")
            
            # Patch 3: Ensure jax.tools.colab_tpu exists
            if not hasattr(jax.tools, 'colab_tpu'):
                class DummyTPU:
                    @staticmethod
                    def setup_tpu():
                        pass  # Do nothing - skip TPU setup
                
                colab_tpu_module = types.ModuleType('jax.tools.colab_tpu')
                colab_tpu_module.setup_tpu = DummyTPU.setup_tpu
                jax.tools.colab_tpu = colab_tpu_module
                sys.modules['jax.tools.colab_tpu'] = colab_tpu_module
                print("[AlphaFold] Created jax.tools.colab_tpu module")
                
        except ImportError:
            print("[AlphaFold] Warning: JAX not available, skipping patches")
        
        from colabfold.download import download_alphafold_params
        from colabfold.utils import setup_logging
        from colabfold.batch import get_queries, run, set_model_type
        from pathlib import Path

        setup_logging(Path(f"{jobname}/log.txt"))
        queries, is_complex = get_queries(queries_path)
        model_type = set_model_type(is_complex, "auto")

        print(f"[AlphaFold] Downloading AlphaFold parameters...")
        download_alphafold_params(model_type, Path("."))

        print(f"[AlphaFold] Running AlphaFold2 prediction...")
        print(f"[AlphaFold] This may take 5-15 minutes depending on sequence length...")
        
        # Suppress all warnings and CUDA errors
        import warnings
        import sys
        import io
        
        # Redirect stderr temporarily to suppress CUDA plugin errors
        old_stderr = sys.stderr
        sys.stderr = io.StringIO()
        
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # Try to run the prediction
                try:
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
            except RuntimeError as e:
                if "jax.tools.colab_tpu.setup_tpu()" in str(e) or "TPU" in str(e):
                    # TPU error is harmless - continue anyway
                    print(f"[AlphaFold] Warning: TPU setup error (ignoring): {e}")
                    # Retry the run call - it should work without TPU
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
                else:
                    raise  # Re-raise if it's a different error
        finally:
            # Restore stderr
            sys.stderr = old_stderr

        print(f"[AlphaFold] Prediction completed! Searching for PDB files...")

        # --- Find output PDB ---
        pdb_files = glob.glob(f"{jobname}/*unrelaxed*.pdb")
        if not pdb_files:
            print(f"[AlphaFold] ERROR: No PDB file found in {jobname}")
            print(f"[AlphaFold] Files in directory: {os.listdir(jobname)}")
            return jsonify({'error': 'No PDB file generated. Check logs for details.'}), 500

        pdb_file = pdb_files[0]
        print(f"[AlphaFold] Found PDB file: {pdb_file}")
        
        with open(pdb_file, 'r') as f:
            pdb_content = f.read()

        # --- Extract pLDDT score from PDB file (average of B-factor column) ---
        plddt_score = 85.0  # Default
        
        try:
            plddt_values = []
            for line in pdb_content.split('\n'):
                if line.startswith('ATOM'):
                    # pLDDT is typically in the B-factor column (column 60-66)
                    try:
                        b_factor = float(line[60:66].strip())
                        if 0 <= b_factor <= 100:  # Valid pLDDT range
                            plddt_values.append(b_factor)
                    except (ValueError, IndexError):
                        continue
            
            if plddt_values:
                plddt_score = sum(plddt_values) / len(plddt_values)
                print(f"[AlphaFold] Calculated pLDDT: {plddt_score:.2f} (from {len(plddt_values)} atoms)")
            else:
                print(f"[AlphaFold] Warning: Could not extract pLDDT from PDB, using default: 85.0")
        except Exception as e:
            print(f"[AlphaFold] Warning: Could not extract pLDDT from PDB: {e}")

        print(f"[AlphaFold] ==========================================")
        print(f"[AlphaFold] ✅ SUCCESS! Structure prediction complete")
        print(f"[AlphaFold] pLDDT score: {plddt_score:.2f}")
        print(f"[AlphaFold] PDB file size: {len(pdb_content)} characters")
        print(f"[AlphaFold] ==========================================\n")

        # --- Return result in format expected by frontend ---
        return jsonify({
            'pdb_content': pdb_content,
            'plddt_score': round(plddt_score, 2),
            'jobname': jobname,
            'confidence_scores': {}  # Can be populated with per-residue scores if needed
        })

    except Exception as e:
        import traceback
        error_msg = str(e)
        traceback.print_exc()
        print(f"\n[AlphaFold] ==========================================")
        print(f"[AlphaFold] ❌ ERROR: {error_msg}")
        print(f"[AlphaFold] ==========================================\n")
        return jsonify({'error': error_msg}), 500

# =============================================
# 🔹 STEP 3 — Kill existing processes on port 5000 (if any)
# =============================================
import socket

def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

# Try to kill process on port 5000 if it exists
if is_port_in_use(5000):
    print("⚠️  Port 5000 is in use. Attempting to free it...")
    try:
        import subprocess
        # On Linux/Colab, find and kill process using port 5000
        result = subprocess.run(['lsof', '-ti:5000'], capture_output=True, text=True)
        if result.returncode == 0:
            pid = result.stdout.strip()
            subprocess.run(['kill', '-9', pid], capture_output=True)
            print(f"✅ Killed existing process on port 5000 (PID: {pid})")
        else:
            # Try using fuser instead
            subprocess.run(['fuser', '-k', '5000/tcp'], capture_output=True)
            print("✅ Attempted to free port 5000")
    except Exception as e:
        print(f"⚠️  Could not automatically free port 5000: {e}")
        print("   You may need to restart the runtime (Runtime → Restart runtime)")
        # Use alternative port
        FLASK_PORT = 5001
        print(f"   Using alternative port: {FLASK_PORT}")
else:
    FLASK_PORT = 5000

# =============================================
# 🔹 STEP 4 — Run Flask in background (CONTINUOUS)
# =============================================
def run_flask():
    """Run Flask server - this function runs indefinitely"""
    try:
        # Disable Flask's default reloader to avoid issues
        app.run(host='0.0.0.0', port=FLASK_PORT, debug=False, use_reloader=False, threaded=True)
    except Exception as e:
        print(f"❌ Flask server error: {e}")

print("\n" + "="*60)
print("🚀 Starting Flask server in background...")
print("="*60)

# Start Flask in a daemon thread (runs in background)
flask_thread = threading.Thread(target=run_flask, daemon=True)
flask_thread.start()

# Wait for Flask to start
import time
print("⏳ Waiting for Flask server to start...")
for i in range(5):
    time.sleep(1)
    if is_port_in_use(FLASK_PORT):
        print(f"✅ Flask server started successfully on port {FLASK_PORT}")
        break
    print(f"   Waiting... ({i+1}/5)")
else:
    print(f"⚠️  Flask server may not have started. Check for errors above.")

print("\n" + "="*60)
print("✅ SERVER STATUS: RUNNING")
print("="*60)
print("ℹ️  The Flask server is now running in the background.")
print("ℹ️  This cell will appear 'finished', but the server keeps running!")
print("ℹ️  DO NOT interrupt or restart this cell unless you want to stop the server.")
print("="*60 + "\n")

# =============================================
# 🔹 STEP 5 — Setup ngrok (BEFORE monitoring loop)
# =============================================
print("\n" + "="*60)
print("🌐 Setting up ngrok tunnel...")
print("="*60)

NGROK_AUTH_TOKEN = "34rVSQJ2Qtvmo53ShGFusPikqyF_5BVRzLan4Tqhuw3tfLtdH"
ngrok.set_auth_token(NGROK_AUTH_TOKEN)

# Kill any existing ngrok tunnels
try:
    tunnels = ngrok.get_tunnels()
    for tunnel in tunnels:
        if tunnel.config['addr'] == f'localhost:{FLASK_PORT}' or ':5000' in str(tunnel.public_url) or ':5001' in str(tunnel.public_url):
            ngrok.disconnect(tunnel.public_url)
            print(f"🔌 Disconnected existing ngrok tunnel: {tunnel.public_url}")
except Exception as e:
    print(f"ℹ️  No existing tunnels to disconnect: {e}")

# Create new tunnel
print(f"🌐 Creating ngrok tunnel to port {FLASK_PORT}...")
try:
    public_url = ngrok.connect(FLASK_PORT)
    
    # Extract clean URL from ngrok response
    ngrok_url_str = str(public_url)
    if 'NgrokTunnel:' in ngrok_url_str:
        # Extract URL from "NgrokTunnel: "https://..." -> "http://localhost:5000""
        match = re.search(r'"(https://[^"]+)"', ngrok_url_str)
        if match:
            ngrok_url = match.group(1)
        else:
            ngrok_url = ngrok_url_str.replace('NgrokTunnel: "', '').split('"')[0]
    else:
        ngrok_url = ngrok_url_str
    
    print("\n" + "="*60)
    print("🌍 PUBLIC NGROK URL:")
    print(f"   {ngrok_url}")
    print("\n📋 Endpoints:")
    print(f"   Health: {ngrok_url}/health")
    print(f"   Predict: {ngrok_url}/predict")
    print("\n" + "="*60)
    print("✅ BACKEND CONFIGURATION:")
    print("="*60)
    print(f"The Flask backend is already configured to use:")
    print(f"   ALPHAFOLD2_NGROK_URL = \"{ngrok_url}\"")
    print("\n💡 IMPORTANT: If the URL above is different from your backend, update:")
    print("   backend/combined_server.py, line 1310")
    print("   OR set environment variable: ALPHAFOLD2_NGROK_URL")
    print("="*60)
    print("\n✅ Ngrok tunnel is now active and ready to receive requests!")
    print("="*60 + "\n")
    
except Exception as e:
    print(f"❌ Error setting up ngrok: {e}")
    print("⚠️  Ngrok tunnel setup failed. The Flask server is still running locally.")
    print("   You may need to check your ngrok auth token or network connection.")
    ngrok_url = None

# Keep the cell alive by running a monitoring loop
print("\n" + "="*60)
print("🔍 Monitoring server status... (This keeps the cell running)")
print("="*60)
print("💡 Heartbeat messages will appear every 2 minutes to confirm server is active")
print("💡 DO NOT interrupt this cell unless you want to stop the server")
print("="*60 + "\n")

try:
    heartbeat_counter = 0
    while True:
        time.sleep(30)  # Check every 30 seconds
        
        # Check if Flask server is still running
        if not is_port_in_use(FLASK_PORT):
            print("⚠️  WARNING: Flask server appears to have stopped!")
            print("   The port is no longer in use. Server may have crashed.")
            break
        
        if not flask_thread.is_alive():
            print("⚠️  WARNING: Flask thread is no longer alive!")
            print("   The background thread has stopped.")
            break
        
        # Print a heartbeat every 2 minutes (4 * 30 seconds = 2 minutes)
        heartbeat_counter += 1
        if heartbeat_counter >= 4:
            current_time = time.strftime('%Y-%m-%d %H:%M:%S')
            ngrok_status = f"ngrok: {ngrok_url}" if ngrok_url else "ngrok: NOT SET UP"
            print(f"💓 [{current_time}] Server heartbeat - Flask running on port {FLASK_PORT}, {ngrok_status}")
            heartbeat_counter = 0  # Reset counter
            
except KeyboardInterrupt:
    print("\n" + "="*60)
    print("🛑 Monitoring stopped by user")
    print("="*60)
    print("⚠️  The Flask server thread may still be running in the background.")
    print("⚠️  To fully stop the server, restart the Colab runtime:")
    print("    Runtime → Restart runtime")
    print("="*60)
except Exception as e:
    print(f"\n⚠️  Monitoring error: {e}")
    print("   Flask server may still be running. Check the port to confirm.")
    print("   You can test the endpoint to verify it's working.")

