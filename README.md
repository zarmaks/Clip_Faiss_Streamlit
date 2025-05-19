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

## 📥 Installation & Local Development

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/clip-faiss-search.git
   cd clip-faiss-search
   ```

2. **Create a virtual environment with Python 3.10**:
   ```bash
   python -m venv venv
   # or, if you have multiple Python versions:
   py -3.10 -m venv venv
   ```

3. **Activate the virtual environment**:
   - On Windows (Command Prompt):
     ```cmd
     venv\Scripts\activate.bat
     ```
   - On Windows (PowerShell):
     ```powershell
     .\venv\Scripts\Activate
     ```
   - On macOS/Linux:
     ```bash
     source venv/bin/activate
     ```

4. **Install the project and its dependencies**:
   ```bash
   pip install -e .
   # or, if you use requirements.txt:
   pip install -r requirements.txt
   ```

## 🚀 Starting the Application

1. **Configure the images folder** in the `src/config.py` file:
   ```python
   # Change this to your images folder path
   IMAGES_PATH = r"C:\\path\\to\\your\\images"
   ```

2. **Run the application**:
   ```bash
   python run.py
   # or
   streamlit run src/main.py
   ```

   The application will automatically open in your default browser, typically at http://localhost:8501

## 🔧 Configuration

- **Changing the images folder**: You can change the images folder either by editing the `config.py` file or via the UI in the sidebar.
- **Setting search parameters**: Similarity threshold, number of results, softmax scaling
- **Metadata filters**: Date range, location (name or coordinates), camera make and model


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
├── environment.yml          # (Optional) Conda environment specification
├── pyproject.toml           # Project metadata and dependencies
├── requirements.txt         # Project dependencies (for pip/Streamlit Cloud)
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

For a complete list of dependencies, see the `pyproject.toml` or `requirements.txt` file.

## 📄 License

[MIT](LICENSE)
