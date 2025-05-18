# 📁 config.py – path settings for the project
import os
from pathlib import Path

# Create upload directory for all environments
UPLOAD_DIR = "uploaded_images"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Default folder with images - will be set to uploaded_images by default
# This works both locally and on the cloud
DEFAULT_IMAGES_PATH = os.path.abspath(UPLOAD_DIR)

# For cloud deployment - use environment variable if available
IMAGES_PATH = os.environ.get("IMAGES_PATH", DEFAULT_IMAGES_PATH)

# FAISS index output file
OUTPUT_INDEX_PATH = "vector_index/vector.index"