# How the AlphaFold2 Notebook Works & Stays Running

## 🔄 Why the Cell Stops & How to Keep It Running

### Problem
When you run the notebook cell, it finishes executing the code and appears "done". However, the Flask server needs to keep running continuously to accept requests.

### Solution
The updated notebook code includes a **monitoring loop** that keeps the cell active and shows that the server is running.

---

## 📋 Step-by-Step Execution Flow

### Step 1: Install Dependencies
```python
!pip install flask flask-cors pyngrok colabfold...
```
- Installs required packages
- Takes 1-2 minutes
- ✅ This completes

### Step 2: Setup Flask App
```python
app = Flask(__name__)
CORS(app)

@app.route('/predict', methods=['POST'])
def predict_structure():
    # AlphaFold2 prediction code
```
- Creates Flask application
- Defines API endpoints
- ✅ This completes

### Step 3: Check & Free Port
```python
if is_port_in_use(5000):
    kill_port(5000)  # Free the port
FLASK_PORT = 5000
```
- Checks if port 5000 is available
- Kills any process using it
- ✅ This completes

### Step 4: Start Flask Server (Background Thread)
```python
flask_thread = threading.Thread(target=run_flask, daemon=True)
flask_thread.start()
```
- Starts Flask in a **background thread**
- Server runs independently
- ✅ This completes

### Step 5: Setup Ngrok
```python
ngrok.set_auth_token("...")
public_url = ngrok.connect(5000)
```
- Creates public tunnel
- Gets ngrok URL
- ✅ This completes

### Step 6: **MONITORING LOOP (Keeps Cell Alive)**
```python
while True:
    time.sleep(30)
    # Check if server is still running
    # Print heartbeat
```
- ⚠️ **This runs FOREVER**
- Keeps the cell "executing"
- Shows server status
- **THIS IS WHY THE CELL KEEPS RUNNING**

---

## 🎯 How It Works

### The Monitoring Loop

The code at the end includes a `while True:` loop that:

1. **Sleeps for 30 seconds**
2. **Checks if Flask server is still running**
3. **Prints a heartbeat message every 2 minutes**
4. **Repeats forever**

This loop keeps the notebook cell "active" and prevents it from finishing. As long as this loop runs, the Flask server continues running in the background.

### Visual Indicators

When running, you'll see:
```
🔍 Monitoring server status... (Press interrupt to stop)
💓 Server heartbeat: 2024-01-15 10:30:00 - Server is running on port 5000
💓 Server heartbeat: 2024-01-15 10:32:00 - Server is running on port 5000
💓 Server heartbeat: 2024-01-15 10:34:00 - Server is running on port 5000
...
```

These heartbeat messages confirm:
- ✅ The cell is still running
- ✅ The Flask server is active
- ✅ The monitoring loop is working

---

## ✅ What You Should See

### When Cell Starts:
```
🚀 Starting Flask server in background...
✅ Flask server started successfully on port 5000
✅ SERVER STATUS: RUNNING
🔍 Monitoring server status... (Press interrupt to stop)
```

### While Running:
```
💓 Server heartbeat: 2024-01-15 10:30:00 - Server is running on port 5000
💓 Server heartbeat: 2024-01-15 10:32:00 - Server is running on port 5000
```

### When Working:
- The cell appears to be "running" (spinner icon)
- Heartbeat messages appear every 2 minutes
- Flask server accepts requests
- Ngrok tunnel stays active

---

## ⚠️ Important Notes

### ✅ DO:
- **Keep the cell running** - Don't interrupt it
- **Watch for heartbeat messages** - Confirms it's working
- **Let it run in background** - It doesn't block other cells
- **Test the endpoint** - Use your Flask backend to send requests

### ❌ DON'T:
- **Don't interrupt the cell** - This stops the Flask server
- **Don't restart the runtime** - This kills everything
- **Don't close the browser tab** - Colab needs to stay open
- **Don't run the cell again** - If it's already running

---

## 🔧 Troubleshooting

### Cell Stops / No Heartbeat Messages

**Problem**: The monitoring loop stopped running

**Solution**:
1. Re-run the cell
2. Check for errors in output
3. Verify Flask server started (look for "✅ Flask server started")

### Server Not Responding

**Problem**: Flask server crashed or stopped

**Solution**:
1. Check the heartbeat messages stopped
2. Look for error messages in the cell output
3. Re-run the cell to restart

### Ngrok URL Changed

**Problem**: Restarted runtime or ngrok reset

**Solution**:
1. Copy the new ngrok URL from output
2. Update `backend/combined_server.py` line 1309
3. Restart your local Flask backend

---

## 🎓 Understanding the Architecture

```
Colab Notebook Cell
    ↓
┌─────────────────────────────────────┐
│ Flask Server (Background Thread)   │ ← Runs continuously
│   - Listens on port 5000           │
│   - Accepts /predict requests      │
│   - Runs AlphaFold2                │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ Ngrok Tunnel                        │ ← Creates public URL
│   - https://xxxx.ngrok-free.dev    │
│   - Forwards to localhost:5000     │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ Monitoring Loop                     │ ← Keeps cell alive
│   - while True:                     │
│   - Checks server status           │
│   - Prints heartbeat               │
└─────────────────────────────────────┘
```

---

## 📝 Summary

**The notebook cell keeps running because of the monitoring loop at the end.**

The loop:
1. ✅ Keeps the cell "executing"
2. ✅ Monitors Flask server status
3. ✅ Shows heartbeat messages
4. ✅ Runs until you interrupt it

**As long as you see heartbeat messages, the server is running and ready to accept requests!**

---

## 🚀 Quick Checklist

Before using the service:
- [ ] Cell shows "✅ Flask server started"
- [ ] Cell shows "✅ SERVER STATUS: RUNNING"
- [ ] Heartbeat messages appear every 2 minutes
- [ ] Ngrok URL is displayed
- [ ] Flask backend has the correct ngrok URL

If all checked ✅, you're ready to use AlphaFold2!



