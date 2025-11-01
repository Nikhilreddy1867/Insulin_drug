# Seq2Drug Fusion API

FastAPI service for SMILES generation from protein sequences.

## Setup

1. Install dependencies:
```bash
cd python-api
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

2. Make sure model files are in `backend/models/`:
   - `progen.pt`
   - `molt5.pt`
   - `fusion_best.pt`
   - `tokenizer.json`

3. Run the service:
```bash
cd python-api
source venv/bin/activate
python api.py
```

The service will start on http://localhost:8000

## API Endpoints

### POST /predict
Generate SMILES from protein sequence.

Request:
```json
{
  "sequence": "MKTAYIAKQR...",
  "max_length": 256,
  "temperature": 1.5,
  "top_k": 50,
  "top_p": 0.9,
  "repetition_penalty": 1.15
}
```

Response:
```json
{
  "sequence": "...",
  "smiles": "CCO...",
  "isValid": true,
  "deviceType": "GPU (Metal/M1)",
  "sampling": {...}
}
```

### GET /health
Check service health and model loading status.

