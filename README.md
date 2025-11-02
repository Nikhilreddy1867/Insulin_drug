# 🧬 Insulin Drug Synthesis Platform

A complete machine learning pipeline for protein analysis, structure prediction, and molecular docking with interactive 3D visualization.

## 🌟 Features

- **Protein Classification** - Predict pathogenicity using custom ML models
- **Sequence Generation** - Generate novel protein sequences
- **SMILES Generation** - Convert protein sequences to drug-like molecules
- **AlphaFold2 Integration** - Predict 3D protein structures (via Google Colab)
- **Molecular Docking** - Dock drug molecules to protein structures
- **3D Visualization** - Interactive molecular viewer (3Dmol.js)
- **Complete Pipeline** - End-to-end workflow from sequence to docking

## 📋 Prerequisites

### Required
- **Python 3.8+** (3.9+ recommended)
- **Node.js 16+** and npm
- **Git**

### For Molecular Docking (Optional)
- **OpenBabel**: `brew install open-babel` (Mac) or download from [openbabel.org](https://openbabel.org/wiki/Category:Installation) (Windows)
- **AutoDock Vina**: `brew install vina` (Mac) or download from [vina.scripps.edu](http://vina.scripps.edu/download.html) (Windows)

### For AlphaFold2 (Optional)
- **Google Colab account** (free tier works)
- **Ngrok account** (free tier works) - [ngrok.com](https://ngrok.com/)

## 🚀 Quick Start

### 1. Clone Repository
```bash
git clone https://github.com/Nikhilreddy1867/Insulin_drug.git
cd Insulin_drug
```

### 2. Install Frontend Dependencies

**Windows (PowerShell/CMD):**
```bash
npm install
```

**Mac/Linux:**
```bash
npm install
```

### 3. Install Backend Dependencies

**Windows:**
```powershell
# Create virtual environment
python -m venv venv

# Activate (PowerShell)
.\venv\Scripts\Activate.ps1
# OR (CMD)
venv\Scripts\activate.bat

# Install dependencies
pip install -r backend/requirements.txt
```

**Mac/Linux:**
```bash
# Create virtual environment
python3 -m venv venv

# Activate
source venv/bin/activate

# Install dependencies
pip install -r backend/requirements.txt
```

### 4. Setup Model Files

Place your model files in `backend/models/`:
- `protein_classifier.pt` - Protein classifier model
- `progen.pt` - ProGen2 embeddings model
- `fusion_best.pt` - Fusion model for SMILES generation
- `molt5.pt` - MolT5 decoder model
- `tokenizer.json` - Tokenizer file
- `Sequence_Generator.pt` - Sequence generator model
- `pca_model.pkl` - PCA model
- `label_encoder.pkl` - Label encoder

### 5. Configure Environment

Create `backend/.env` file:
```env
# MongoDB (for authentication)
MONGODB_URI=mongodb://localhost:27017/insulin_drug

# Flask
SECRET_KEY=your-secret-key-here
FLASK_ENV=development

# AlphaFold2 (optional - set after Colab setup)
ALPHAFOLD2_NGROK_URL=https://your-ngrok-url.ngrok-free.app

# Docking (optional)
DOCKING_NGROK_URL=https://your-docking-ngrok-url.ngrok-free.app
```

### 6. Start Servers

**Windows:**
```powershell
# Terminal 1: Backend
cd backend
..\venv\Scripts\Activate.ps1
python combined_server.py

# Terminal 2: Frontend
npm run dev

# Terminal 3: FastAPI Service (for SMILES generation)
cd python-api
..\..\venv\Scripts\Activate.ps1
uvicorn api:app --host 0.0.0.0 --port 8000
```

**Mac/Linux:**
```bash
# Terminal 1: Backend
cd backend
source ../venv/bin/activate
python combined_server.py

# Terminal 2: Frontend
npm run dev

# Terminal 3: FastAPI Service (for SMILES generation)
cd python-api
source ../../venv/bin/activate
uvicorn api:app --host 0.0.0.0 --port 8000
```

### 7. Access Application

- **Frontend**: http://localhost:5173
- **Backend API**: http://localhost:5001
- **FastAPI Service**: http://localhost:8000

## 📖 Usage

### Protein Classification
1. Navigate to Dashboard → Classification tab
2. Enter protein sequence
3. Click "Predict"
4. View pathogenicity prediction and confidence scores

### Sequence Generation
1. Dashboard → Sequence Generation tab
2. Enter seed sequence (optional)
3. Click "Generate Sequences"
4. View generated sequences with device info

### SMILES Generation
1. Dashboard → SMILES Generation tab
2. Enter protein sequence
3. Adjust generation parameters (optional)
4. Click "Generate SMILES"
5. View generated SMILES molecules

### AlphaFold2 Structure Prediction
1. **Setup Colab Notebook** (one-time):
   - Open `backend/colab_flask_helper_alphafold2_fixed.py` in Google Colab
   - Update ngrok token in the code
   - Run all cells
   - Copy the ngrok URL

2. **Update Backend**:
   - Add ngrok URL to `backend/.env`: `ALPHAFOLD2_NGROK_URL=...`
   - Restart Flask backend

3. **Use in Dashboard**:
   - Dashboard → AlphaFold2 tab
   - Enter protein sequence
   - Click "Predict Structure"
   - Wait 5-15 minutes for results
   - Download PDB file

### Molecular Docking

#### Local Docking (Recommended)
**Mac:**
```bash
# Install dependencies
brew install open-babel vina
pip install rdkit
```

**Windows:**
1. Download OpenBabel from [openbabel.org](https://openbabel.org/wiki/Category:Installation)
2. Download Vina from [vina.scripps.edu](http://vina.scripps.edu/download.html)
3. Add to PATH
4. Install RDKit: `pip install rdkit`

**Usage:**
- Dashboard → Docking tab
- Enter SMILES string
- Paste protein PDB content (from AlphaFold2)
- Click "Run Docking"
- View binding affinities and 3D visualization

#### Colab Docking (Alternative)
1. Setup Colab notebook with `backend/colab_flask_helper_docking_fixed.py`
2. Get ngrok URL
3. Add to `backend/.env`: `DOCKING_NGROK_URL=...`

### Complete Pipeline
1. Dashboard → Complete Pipeline tab
2. Enter protein sequence and SMILES
3. Click "Run Complete Pipeline"
4. Automatically runs: Sequence → AlphaFold2 → Docking
5. View combined results with 3D visualization

## 🔧 Configuration

### GPU Support (M1 Mac)
The application automatically detects and uses:
- **M1/M2 Mac**: Metal Performance Shaders (MPS)
- **CUDA**: NVIDIA GPUs
- **CPU**: Fallback if GPU unavailable

Device information is displayed in all API responses.

### MongoDB Setup (for Authentication)

**Mac:**
```bash
brew install mongodb-community
brew services start mongodb-community
```

**Windows:**
1. Download MongoDB from [mongodb.com](https://www.mongodb.com/try/download/community)
2. Install and start MongoDB service

**Create Database:**
```bash
python setup_mongodb.py
```

## 📁 Project Structure

```
Insulin_drug/
├── backend/                 # Flask backend server
│   ├── combined_server.py  # Main server (port 5001)
│   ├── local_docking.py    # Local docking module
│   ├── requirements.txt     # Python dependencies
│   └── models/             # ML model files
├── python-api/             # FastAPI service (port 8000)
│   ├── api.py              # SMILES generation service
│   └── requirements.txt
├── src/                    # React frontend
│   ├── pages/              # Page components
│   └── components/         # Reusable components
├── package.json            # Frontend dependencies
└── README.md               # This file
```

## 🐛 Troubleshooting

### Port Already in Use
**Windows:**
```powershell
# Find process using port
netstat -ano | findstr :5001
# Kill process (replace PID)
taskkill /PID <PID> /F
```

**Mac:**
```bash
# Find process using port
lsof -ti:5001
# Kill process
kill -9 $(lsof -ti:5001)
```

### Module Not Found Errors
- Ensure virtual environment is activated
- Reinstall dependencies: `pip install -r backend/requirements.txt`
- Check Python version: `python --version` (should be 3.8+)

### GPU Not Detected
- Check PyTorch installation: `python -c "import torch; print(torch.cuda.is_available())"`
- For M1 Mac: Ensure PyTorch supports MPS
- Device will fallback to CPU automatically

### Docking Dependencies Missing
**Mac:**
```bash
brew install open-babel vina
pip install rdkit
```

**Windows:**
- Install OpenBabel and Vina manually
- Add to system PATH
- Install RDKit via pip

### AlphaFold2 Connection Errors
- Verify Colab notebook is running
- Check ngrok URL in `backend/.env`
- Test connection: `curl http://localhost:5001/alphafold2/health`

## 🚧 Development

### Adding New Features
1. Backend: Edit `backend/combined_server.py`
2. Frontend: Edit files in `src/pages/`
3. FastAPI: Edit `python-api/api.py`

### Testing
```bash
# Backend health check
curl http://localhost:5001/api/health

# Frontend
# Open http://localhost:5173
```

## 📝 License

This project is for research and educational purposes.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📧 Support

For issues and questions, please open an issue on GitHub.

---

**Made with ❤️ for Insulin Drug Synthesis Research**


