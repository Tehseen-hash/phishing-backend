"""
Entry point to run the FastAPI backend server.
This script sets up the path and starts the uvicorn server.
"""

import os
import sys
import uvicorn

# Ensure the Backend directory is in the Python path
backend_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Backend")
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

if __name__ == "__main__":
    print(f"Starting FastAPI server from: {backend_dir}")
    # Run the FastAPI app from Backend/main.py
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, app_dir=backend_dir)
