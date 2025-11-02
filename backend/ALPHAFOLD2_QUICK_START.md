# AlphaFold2 Colab Integration - Quick Start Guide

## 🎯 Goal
Instead of copying/pasting PDB files manually, your website will automatically:
1. Send protein sequence to Colab notebook via ngrok
2. Run AlphaFold2 prediction in Colab
3. Extract rank 1 PDB automatically
4. Return PDB content directly to your website

## 📝 Step-by-Step Instructions

### Step 1: Open Colab Notebook
1. Go to: https://colab.research.google.com/github/sokrypton/ColabFold/blob/main/AlphaFold2.ipynb
2. **File → Save a copy in Drive** (make your own copy)

### Step 2: Run Normal Prediction First
1. Set your sequence in the input cell
2. Click **Runtime → Run all**
3. Wait for prediction to complete (5-15 minutes)
4. Verify results look good

### Step 3: Add Integration Code
**Add this as a NEW cell at the very END of the notebook:**

```python
#@title Setup Flask API & ngrok Integration
# Copy and paste the code from: ALPHAFOLD2_COLAB_CODE.py
```

Open the file `ALPHAFOLD2_COLAB_CODE.py` in this folder and copy **all the code** into a new Colab cell.

⚠️ **IMPORTANT**: After adding the code, **SAVE THE NOTEBOOK**:
- Click **File → Save** (or press `Ctrl+S` / `Cmd+S`)
- This saves your changes permanently
- If you don't save, the code will disappear when you disconnect!

### Step 4: Get ngrok Token
1. Sign up at https://ngrok.com (free account works)
2. Go to: https://dashboard.ngrok.com/get-started/your-authtoken
3. Copy your auth token

### Step 5: Update Token in Colab
In the code you just pasted, find this line:
```python
NGROK_AUTH_TOKEN = "YOUR_NGROK_TOKEN_HERE"
```
Replace `YOUR_NGROK_TOKEN_HERE` with your actual token.

### Step 6: Run the Integration Cell
1. Run the cell you just added
2. Wait for Flask server to start
3. **Copy the ngrok URL** that appears (e.g., `https://xxxx-xxxx.ngrok-free.app`)

### Step 7: Update Your Backend
1. Open `backend/.env` file
2. Add or update:
   ```
   ALPHAFOLD2_NGROK_URL=https://xxxx-xxxx.ngrok-free.app
   ```
   (Use the URL from Step 6)
3. Save the file
4. Restart your Flask backend server

### Step 8: Test It!
1. Open your website: http://localhost:5173
2. Go to AlphaFold2 page
3. Enter a protein sequence
4. Click "Predict Structure"
5. The PDB will be automatically extracted and sent back - **no manual copying needed!**

## 🔄 How It Works

```
Your Website → Flask Backend → ngrok → Colab Notebook → AlphaFold2
                                                           ↓
Your Website ← Flask Backend ← ngrok ← Colab Notebook ← Rank 1 PDB
```

1. **Website sends** sequence to `/alphafold2/predict`
2. **Backend forwards** to Colab's ngrok URL
3. **Colab runs** AlphaFold2 (takes 5-15 min)
4. **Colab extracts** rank 1 PDB from results automatically
5. **Colab returns** PDB content as JSON
6. **Backend receives** and forwards to website
7. **Website displays** PDB in 3D viewer

## 📋 Files Created

1. **`ALPHAFOLD2_COLAB_INTEGRATION.md`** - Detailed guide with explanations
2. **`ALPHAFOLD2_COLAB_CODE.py`** - Complete code to paste into Colab
3. **`ALPHAFOLD2_EXTRACT_RANK1.py`** - Simple extraction script (if you just want to extract from existing results)

## ⚠️ Important Notes

- **Keep Colab tab open**: The notebook must stay running for the API to work
- **Ngrok free tier**: URLs change when you restart ngrok (upgrade for static URLs)
- **Colab timeouts**: Free Colab sessions timeout after ~12 hours of inactivity
- **GPU required**: Make sure Runtime → Change runtime type → GPU is selected

## 🐛 Troubleshooting

### "Connection refused"
- Make sure the Flask cell ran successfully
- Check that ngrok shows a URL in the output

### "No PDB file found"
- Verify prediction completed successfully
- Check the jobname folder contains PDB files

### Ngrok URL changes
- Re-run the integration cell to get new URL
- Update backend `.env` file with new URL
- Restart Flask backend

### Colab session expired
- Re-run the notebook from the beginning
- Re-run the integration cell
- Get new ngrok URL and update backend

## ✅ Success Checklist

- [ ] Colab notebook copied to Drive
- [ ] Integration code added as new cell
- [ ] ngrok token obtained and set
- [ ] Flask server started (shows "Flask server started on port 5000")
- [ ] ngrok URL copied
- [ ] Backend `.env` updated with ngrok URL
- [ ] Flask backend restarted
- [ ] Tested from website - PDB appears automatically!

## 🎉 You're Done!

Now when you run predictions from your website, the rank 1 PDB will automatically be extracted and sent back - no more manual copying!

