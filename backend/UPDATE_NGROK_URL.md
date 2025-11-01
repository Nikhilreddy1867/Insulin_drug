# Quick Update: AlphaFold2 Ngrok URL

Your AlphaFold2 notebook is running and generated this ngrok URL:

```
https://muzzleloading-pedro-originally.ngrok-free.dev
```

## Update Flask Backend

Open `backend/combined_server.py` and find line ~1308:

```python
ALPHAFOLD2_NGROK_URL = os.environ.get('ALPHAFOLD2_NGROK_URL', 'http://localhost:8000')
```

**Change it to:**

```python
ALPHAFOLD2_NGROK_URL = os.environ.get('ALPHAFOLD2_NGROK_URL', 'https://muzzleloading-pedro-originally.ngrok-free.dev')
```

**OR** set environment variable before starting Flask:

### Windows PowerShell:
```powershell
$env:ALPHAFOLD2_NGROK_URL="https://muzzleloading-pedro-originally.ngrok-free.dev"
cd backend
python combined_server.py
```

### Linux/Mac:
```bash
export ALPHAFOLD2_NGROK_URL="https://muzzleloading-pedro-originally.ngrok-free.dev"
cd backend
python combined_server.py
```

## Restart Flask Backend

1. Stop the current Flask server (Ctrl+C)
2. Start it again with the updated URL
3. Test in Dashboard → AlphaFold2 tab

## Important Notes

⚠️ **Keep the Colab notebook cell running!** If you stop it, the Flask server and ngrok tunnel will stop.

⚠️ **Ngrok free tier URLs change** when you restart the notebook. If the URL changes, update `combined_server.py` again.

## Testing

1. Go to Dashboard → AlphaFold2 tab
2. Enter a protein sequence (e.g., `MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQFSTLKSTVEAIWAGIKATEAAVSEEFGLAPFLPDQIHFVHSQELLSRYPDLDAKGRERAIAKDLGAVFLVGIGGKLSDGHRHDVRAPDYDDWQTSTSTSLPRADLQLFVDGVRQLEWLSQRLQQPQQKSAFAVQEDFNRSWFRPGHRRNKVFDLPIGVLKSSAQNLMNQEDVHSKQAPGTILKSQGMQVFVLEELDKTLFTLGFHKPAIVQHASSAKDLGPLLDGIWKTTTTKQAAKCLQKNLPSFLGVTSSEFRYLMNSQTRLPDNYLPLLPAIIDRFDNTLPLTGQAQIIFRRFLPLQGKEFQ`)
3. Click "Predict Structure"
4. Wait 5-15 minutes for results
5. Download PDB file when complete

## Troubleshooting

**"Cannot connect to AlphaFold2 service"**
- Check Colab notebook is still running
- Verify ngrok URL in `combined_server.py`
- Check Flask backend logs for connection errors

**"No PDB file generated"**
- Check Colab notebook output for errors
- Ensure sequence is valid (only ACDEFGHIKLMNPQRSTVWY)
- Check AlphaFold2 logs in the job folder

**Timeout errors**
- Increase timeout in `combined_server.py` (currently 900 seconds = 15 minutes)
- Long sequences take longer to process


