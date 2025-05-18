# 📁 config.py – path settings for the project
import os
from pathlib import Path

# Default folder with images - will be overridden by the UI
# For local development
DEFAULT_IMAGES_PATH = r"C:\\Users\\zarma\\pexels_imgs"

# For cloud deployment - use environment variable if available
IMAGES_PATH = os.environ.get("IMAGES_PATH", DEFAULT_IMAGES_PATH)

# Create upload directory for cloud deployment
UPLOAD_DIR = "uploaded_images"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# FAISS index output file
OUTPUT_INDEX_PATH = "vector_index/vector.index"