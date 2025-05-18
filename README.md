<<<<<<< HEAD
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

### Using Conda

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

3. **Install the project and its dependencies**:
   ```bash
   pip install -e .
   ```

### Using Pip

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

3. **Install the project and its dependencies**:
   ```bash
   pip install -e .
   ```



## 🚀 Starting the Application

1. **Configure the images folder** in the `src/config.py` file:
   ```python
   # Change this to your images folder path
   IMAGES_PATH = r"C:\\path\\to\\your\\images"
   ```

2. **Run the application**:
   ```bash
   # Option 1: Using the run.py script (recommended)
   python run.py
   
   # Option 2: Running the Streamlit app directly
   streamlit run src/main.py
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
   git remote add origin https://github.com/yourusername/clip-faiss-image-search.git
   git push -u origin main
   ```

## 🚀 Deploying to Streamlit Cloud

1. **Create a Streamlit Cloud account** at [https://streamlit.io/cloud](https://streamlit.io/cloud)

2. **Deploy the app**:
   - Log in to Streamlit Cloud
   - Click "New app"
   - Select your GitHub repository
   - Set the main file path to `src/main.py`
   - Configure advanced settings:
     - Python version: 3.10 (recommended)
     - Requirements: Use your pyproject.toml

3. **Configure your app**:
   - Set any required secrets in the Streamlit Cloud dashboard
   - Adjust the compute resources as needed

## 📝 Note about index files

The FAISS index is automatically created during the first use and stored in the `vector_index/` folder. If you change the set of images, you can regenerate the index using the "Force Reindex Images" button in the UI.

## 📚 Project Structure

```
clip-faiss-search/
├── src/
│   ├── config.py            # Application settings
│   ├── main.py              # Main file with Streamlit UI
│   ├── faiss_utils.py       # Helper functions for CLIP+FAISS
├── run.py                   # Launcher script for the application
├── environment.yml          # Conda environment specification
├── pyproject.toml           # Project metadata and dependencies
├── README.md
├── .gitignore
```

> **Note:**  
> The `vector_index/` folder, where the FAISS index files are stored, is created automatically during the first use and is excluded from version control via `.gitignore`.

## 📋 Requirements

The project requires the following main dependencies:

- faiss-cpu (1.7.4)
- sentence-transformers (2.2.2)
- matplotlib
- pillow
- numpy
- scikit-learn
- streamlit
- tqdm
- requests

For a complete list of dependencies, see the `pyproject.toml` file.

## 📄 License

[MIT](LICENSE)
=======
# Clip_Faiss_Streamlit
>>>>>>> 5cd40f5020e87029df82710c16b49f68edf28cc5
