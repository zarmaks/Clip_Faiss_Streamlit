#!/usr/bin/env python3
"""
CLIP-FAISS Image Search Launcher

This script launches the CLIP-FAISS image search application.
"""

import os
import sys
import subprocess

# Instead of trying to import, directly construct the path to main.py
current_dir = os.path.dirname(os.path.abspath(__file__))
main_path = os.path.join(current_dir, "src", "main.py")

if __name__ == "__main__":
    # Verify the main.py file exists
    if not os.path.exists(main_path):
        print(f"Error: Could not find {main_path}")
        sys.exit(1)
        
    print(f"Launching Streamlit app from {main_path}")
    
    # Launch Streamlit using subprocess - the modern way to start Streamlit
    subprocess.run(["streamlit", "run", main_path], check=True)