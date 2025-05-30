# CLIP-FAISS Image Search 🔍

A research tool for exploring vector embeddings and semantic search with images, powered by OpenAI CLIP and FAISS.
live demo: https://semantic-image-clip-faiss.streamlit.app/
---

## 🧭 Project Overview

**clip-faiss-search** is designed to help researchers and practitioners understand and experiment with vector embeddings for images. It enables semantic image search using both text and image queries, with advanced filtering and visualization options—all through an interactive Streamlit UI.

---

## ✨ Key Features

- **Text-based Search:** Enter natural language queries to find semantically similar images using CLIP embeddings.
- **Image-based Search:** Upload an image and retrieve visually similar images from your dataset.
- **Metadata Filtering:** Filter results by metadata such as date, location, and camera information.
- **Softmax Scaling:** Adjust the contrast of similarity scores to make thresholds more meaningful (e.g., set threshold close to 0.8).
- **Threshold & Top-K Filtering:** Control the number of results (top-K) or set a similarity threshold.
- **t-SNE Visualization:** Visualize the entire image dataset, grouped by semantic similarity, using t-SNE dimensionality reduction.
- **Streamlit UI:** User-friendly, interactive interface for all operations.

---

## 🛠️ Usage Modes

- **Text Query Search:** Enter a description to find matching images.
- **Image Query Search:** Upload an image to find similar ones.
- **Metadata-only Search:** Filter images using metadata without a text or image query.
- **Semantic Visualization:** Explore the dataset’s structure with t-SNE clustering.

---

## ⚙️ Technical Highlights

- **CLIP Embeddings:** Used for both text and image queries, enabling cross-modal search.
- **FAISS Indexing:** Fast nearest-neighbor search over high-dimensional vectors.
- **Softmax Scaling:** Option to apply softmax scaling to similarity scores for better contrast.
- **Flexible Filtering:** Supports both threshold-based and top-K result filtering.
- **Metadata Extraction:** Extracts and utilizes EXIF metadata for advanced filtering.
- **Visualization:** t-SNE plots for exploring semantic clusters in your dataset.

---

## 🧪 Research Context

This tool is part of a broader research effort (see [LightlyGPT](https://github.com/zarmaks/Lightl-yGPT)) to build agentic AI systems that allow users to interact with image datasets using natural language—no SQL, no dashboards, just questions and answers. The approach combines:

- **LangChain** for reasoning,
- **CLIP** for visual understanding,
- **ChromaDB** (in the capstone) or **FAISS** (in this tool) for fast retrieval.

---

## 📚 Project Structure

```
clip-faiss-search/
├── src/
│   ├── config.py            # Application settings
│   ├── main.py              # Main Streamlit UI and logic
│   ├── faiss_utils.py       # Helper functions for CLIP+FAISS
├── run.py                   # Launcher script
├── pyproject.toml           # Project metadata and dependencies
├── requirements.txt         # Project dependencies
├── README.md
├── .gitignore
```

---

## 📋 Requirements

Main dependencies:

- faiss-cpu (1.7.4)
- sentence-transformers (2.2.2)
- matplotlib
- pillow
- numpy
- scikit-learn
- streamlit
- tqdm
- requests

For a complete list, see `pyproject.toml` or `requirements.txt`.

---
## 🚀 Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/zarmaks/Clip_Faiss_Streamlit.git
   cd Clip_Faiss_Streamlit
   ```

2. **(Recommended) Create a virtual environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Streamlit app:**
   ```bash
   streamlit run src/main.py
   ```

> For advanced users, you can also use `pyproject.toml` with Poetry.

---

Add this section after the "Requirements" and before the "License" in your README.md for a complete and user-friendly documentation!

## 📝 Note about index files

The FAISS index is automatically created during the first use and stored in the `vector_index/` folder. If you change the set of images, you can regenerate the index using the "Force Reindex Images" button in the UI.

> **Note:**  
> The `vector_index/` folder is excluded from version control via `.gitignore`.

---

## 📄 License

[MIT](LICENSE)
