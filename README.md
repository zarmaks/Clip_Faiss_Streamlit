# CLIP-FAISS Image Search 🔍

An image search application that uses OpenAI CLIP (Contrastive Language-Image Pretraining) and FAISS (Facebook AI Similarity Search) to find similar images based on text descriptions, other images, or metadata.

## ✨ Features

- **Natural language search**: Find images by describing them with simple text
- **Image-based search**: Upload an image and find similar ones
- **Metadata filtering**: Filter by date, location or camera
- **t-SNE visualization**: Visualize the semantic relationships between images
- **Interactive UI**: User-friendly interface with Streamlit

## 🛠️ System Requirements

- Python 3.10 or newer
- 4GB RAM minimum (8GB recommended)
- At least 1GB of free disk space

## 📥 Installation

There are three ways to install:

### A. Using Poetry (Recommended)

1. **Install Poetry** if you don't have it already:
   ```bash
   pip install poetry
   ```

2. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/clip-faiss-search.git
   cd clip-faiss-search
   ```

3. **Install dependencies**:
   ```bash
   poetry install
   ```

4. **Activate the environment**:
   ```bash
   poetry shell
   ```

### B. Using Conda

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/clip-faiss-search.git
   cd clip-faiss-search
   ```

2. **Create and activate the environment from the environment.yml file**:
   ```bash
   conda env create -f environment.yml
   conda activate clip_faiss_env
   ```

3. **Install the additional dependencies**:
   ```bash
   pip install -e .
   ```

### C. Using Pip

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/clip-faiss-search.git
   cd clip-faiss-search
   ```

2. **Create a virtual environment** (optional but recommended):
   ```bash
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install the package and its dependencies**:
   ```bash
   pip install -e .
   ```

## 🚀 Starting the Application

1. **Configure the images folder** in the `config.py` file:
   ```python
   # Change this to your images folder path
   IMAGES_PATH = r"C:\\path\\to\\your\\images"
   ```

2. **Run the application**:
   ```bash
   streamlit run main.py
   ```

   The application will automatically open in your default browser, typically at http://localhost:8501

## 🔧 Configuration

- **Changing the images folder**: You can change the images folder either by editing the `config.py` file or via the UI in the sidebar.
  
- **Setting search parameters**:
  - Similarity threshold
  - Number of results
  - Softmax scaling

- **Metadata filters**:
  - Date range
  - Location (name or coordinates)
  - Camera make and model

## 🌐 Uploading to GitHub

1. **Create a new repository** on GitHub
2. **Initialize Git** in your project folder:
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git branch -M main
   git remote add origin https://github.com/yourusername/clip-faiss-search.git
   git push -u origin main
   ```

## 📝 Note about index files

The FAISS index is automatically created during the first use and stored in the `vector_index/` folder. If you change the set of images, you can regenerate the index using the "Force Reindex Images" button in the UI.

## 📚 Project Structure

```
clip_faiss_search/
├── config.py            # Application settings
├── main.py              # Main file with Streamlit UI
├── faiss_utils.py       # Helper functions for CLIP+FAISS
├── environment.yml      # Conda environment
├── pyproject.toml       # Python package settings
└── vector_index/        # FAISS index files
```

## 📋 Requirements

For a complete list of dependencies, see the `pyproject.toml` file.

## 📄 License

[MIT](LICENSE)