# ⚠️ Important: Save Your Colab Notebook After Adding Code!

## Why Code Disappears

When you disconnect from Colab runtime or the session expires:
- **New cells you added are NOT automatically saved**
- Only cells that existed when you opened the notebook are preserved
- You need to **manually save** the notebook after adding new cells

## ✅ Solution: Save Your Notebook

### Method 1: Save to Google Drive (Recommended)
1. After adding the integration code cell, click **File → Save**
2. OR press `Ctrl+S` (Windows) or `Cmd+S` (Mac)
3. The notebook will be saved to your Google Drive
4. Your code will persist even after disconnecting

### Method 2: Download and Re-upload
1. After adding code: **File → Download → Download .ipynb**
2. Save the `.ipynb` file to your computer
3. When you need it again: **File → Upload notebook**
4. Upload your saved `.ipynb` file

## 🔄 Best Practice Workflow

### First Time Setup:
1. Open ColabFold notebook: https://colab.research.google.com/github/sokrypton/ColabFold/blob/main/AlphaFold2.ipynb
2. **File → Save a copy in Drive** (THIS IS CRITICAL!)
3. Work in YOUR copy (from Drive)
4. Add the integration code cell from `COLAB_CELL_CODE.txt`
5. **File → Save** (Ctrl+S or Cmd+S) - SAVE IT!
6. Now your code is permanently saved

### Every Time You Use It:
1. Open YOUR saved copy from Google Drive
2. The integration code cell should already be there (if you saved it)
3. If not, add it again and **SAVE THIS TIME!**
4. Run the cells you need

## 💡 Quick Tips

- **Always work in a Drive copy**, not the original notebook
- **Save frequently** - press Ctrl+S/Cmd+S often
- **Check Drive** - Your saved notebooks appear in Google Drive under "Colab Notebooks" folder
- If code is missing, it means the notebook wasn't saved after you added it

## 🔍 How to Verify It's Saved

After adding the code cell:
1. Click **File** menu
2. If you see "Revert to last saved version", it means changes are unsaved
3. Click **Save** to save your changes
4. The "Revert" option will disappear when saved

## 📝 What Happens When You Don't Save

- ✅ Code runs fine while runtime is active
- ❌ Code disappears when you disconnect
- ❌ Code disappears when runtime times out
- ❌ Code disappears when browser closes
- ❌ Have to add code again next time

## ✅ What Happens When You Save

- ✅ Code persists in your Drive copy
- ✅ Code is there next time you open the notebook
- ✅ Only need to add it once
- ✅ Can share your saved notebook with others

## 🚨 Emergency Fix: If You Forgot to Save

If you already disconnected and code is gone:

1. **Open your Drive copy** (File → Open notebook → Drive)
2. Check if the cell is there
3. If not, add it again from `COLAB_CELL_CODE.txt`
4. **IMMEDIATELY SAVE** (Ctrl+S / Cmd+S)
5. Verify it's saved (no "Revert" option in File menu)

## 🎯 Summary

**The golden rule**: After adding ANY new code to Colab, **ALWAYS save the notebook** before disconnecting!

```
Add Code → File → Save → Code is Permanent ✅
Add Code → Disconnect → Code is Lost ❌
```

