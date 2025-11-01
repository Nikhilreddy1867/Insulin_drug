# Seq2Drug Fusion API - Setup Complete ✅

The SMILES generation module has been successfully integrated to match your working website architecture.

## Architecture

```
Frontend (React, port 5173)
    ↓
Flask Backend (combined_server.py, port 5001)
    ↓
FastAPI Service (python-api/api.py, port 8000)
    ↓
Fusion Model (progen.pt + molt5.pt + fusion_best.pt)
```

## Files Created

1. **`python-api/api.py`** - FastAPI service for SMILES generation (exact match to your working code)
2. **`python-api/requirements.txt`** - Python dependencies for FastAPI service
3. **`python-api/README.md`** - Setup and API documentation

## Changes Made

1. **Flask Backend (`backend/combined_server.py`)**:
   - Updated `/generate-smiles` endpoint to proxy requests to FastAPI service
   - Removed direct SMILES generation code
   - Added proper error handling for FastAPI connection issues

2. **Model Path**: FastAPI service looks for models in `backend/models/` (already correct)

## Setup Instructions

### 1. Install FastAPI Dependencies

```bash
cd python-api
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Start the FastAPI Service

```bash
cd python-api
source venv/bin/activate
python api.py
```

The service will start on **http://localhost:8000**

### 3. Start Flask Backend (as usual)

```bash
cd backend
source venv/bin/activate  # or use the main venv
python combined_server.py
```

### 4. Start Frontend (as usual)

```bash
npm run dev
```

## Testing

### Test FastAPI directly:
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"sequence": "MKTAYIAKQR"}'
```

### Test through Flask (full integration):
```bash
curl -X POST http://localhost:5001/generate-smiles \
  -H "Content-Type: application/json" \
  -d '{"sequence": "MKTAYIAKQR"}'
```

## API Endpoints

### FastAPI Service (http://localhost:8000)

- **POST /predict** - Generate SMILES
  - Request: `{ "sequence": "...", "max_length": 256, "temperature": 1.5, "top_k": 50, "top_p": 0.9, "repetition_penalty": 1.15 }`
  - Response: `{ "sequence": "...", "smiles": "...", "isValid": true, "deviceType": "...", "sampling": {...} }`

- **GET /health** - Check service status

### Flask Backend (http://localhost:5001)

- **POST /generate-smiles** - Proxies to FastAPI service
  - Same request format as before
  - Returns compatible response format

## Model Files Required

Make sure these files exist in `backend/models/`:
- ✅ `progen.pt`
- ✅ `molt5.pt`
- ✅ `fusion_best.pt`
- ✅ `tokenizer.json`

## Default Parameters

- `max_length`: 256 (increased from 128)
- `temperature`: 1.5
- `top_k`: 50
- `top_p`: 0.9
- `repetition_penalty`: 1.15

## Error Handling

If FastAPI service is not running, Flask will return:
```json
{
  "success": false,
  "error": "SMILES generation service unavailable. Please start the Python API service (python-api/api.py) on port 8000."
}
```

## Notes

- The FastAPI service loads models on startup (may take 30-60 seconds)
- Models are cached in memory after first load
- Frontend doesn't need any changes - it already calls the Flask endpoint
- The architecture now matches your working website exactly!

