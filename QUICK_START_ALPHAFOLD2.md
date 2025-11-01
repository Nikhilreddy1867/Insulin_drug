# Quick Start: AlphaFold2 Integration

## ✅ Current Status

- **Ngrok URL**: `https://muzzleloading-pedro-originally.ngrok-free.dev`
- **Backend**: Already configured (no changes needed!)
- **Navbar**: AlphaFold2 & Docking removed (only in Dashboard)
- **Frontend**: Ready to use

---

## 🔧 Fix Port Conflict Issue

The notebook shows "Port 5000 is in use". Use this **FIXED notebook code**:

**File**: `backend/ALPHAFOLD2_NOTEBOOK_FIXED.py`

**Or paste this in your Colab cell** (replaces the old code):

```python
# This version automatically handles port conflicts
# Copy the entire file: backend/ALPHAFOLD2_NOTEBOOK_FIXED.py
```

The fixed version:
- ✅ Automatically kills processes on port 5000
- ✅ Uses port 5001 if 5000 is busy
- ✅ Disconnects old ngrok tunnels
- ✅ Starts Flask server properly
- ✅ Runs continuously

---

## 📝 Important Notes

### 1. Ngrok URL Update
**You DON'T need to update the backend manually!**

The backend is already configured with:
```
ALPHAFOLD2_NGROK_URL = 'https://muzzleloading-pedro-originally.ngrok-free.dev'
```

**Only update if:**
- You restart the Colab notebook
- You get a NEW ngrok URL from the notebook
- Then update line 1309 in `backend/combined_server.py`

### 2. Continuous Running
**The notebook WILL run continuously IF:**
- ✅ You use the fixed notebook code
- ✅ You don't interrupt the cell
- ✅ The Colab runtime doesn't disconnect

**If it stops:**
- Re-run the cell
- Update backend with new ngrok URL (if it changed)

### 3. Navbar Cleanup
✅ **AlphaFold2 and Docking are now ONLY in Dashboard**
- They're removed from navbar
- Access via: Dashboard → AlphaFold2 tab / Docking tab

---

## 🚀 Testing

1. **Use the fixed notebook code** (`ALPHAFOLD2_NOTEBOOK_FIXED.py`)
2. **Run it in Colab** - it should start without port errors
3. **Keep the cell running**
4. **Test in Dashboard**:
   - Go to Dashboard → AlphaFold2 tab
   - Enter sequence
   - Click "Predict Structure"
   - Wait 5-15 minutes

---

## 🐛 Troubleshooting

### Port 5000 Error
✅ **Fixed in the new notebook code** - it automatically handles this

### Notebook Stops Running
- Check Colab runtime hasn't disconnected
- Re-run the cell
- Check for errors in output

### Backend Connection Error
- Verify ngrok URL matches backend (line 1309)
- Restart Flask backend
- Test ngrok URL: `curl https://muzzleloading-pedro-originally.ngrok-free.dev/health`

---

**You're all set!** Use the fixed notebook code and everything will work. 🎉



