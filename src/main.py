# 📁 main.py - Main execution file with Streamlit UI

from faiss_utils import generate_clip_embeddings, create_faiss_index, retrieve_similar_images, geocode_location, search_by_metadata_only
from sentence_transformers import SentenceTransformer
import streamlit as st
import os
import time
import shutil
from config import IMAGES_PATH, OUTPUT_INDEX_PATH, UPLOAD_DIR
from PIL import Image
import json
import datetime
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import io
import base64
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import uuid

# Detect if running in Streamlit Cloud environment
IS_STREAMLIT_CLOUD = os.environ.get('STREAMLIT_RUNTIME_ENV') == 'cloud'

# Create necessary directories if they don't exist
os.makedirs(os.path.dirname(OUTPUT_INDEX_PATH), exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Debug information for Streamlit Cloud
if IS_STREAMLIT_CLOUD:
    st.sidebar.markdown("### Debugging Info")
    st.sidebar.write(f"Running in Streamlit Cloud: {IS_STREAMLIT_CLOUD}")
    st.sidebar.write(f"Python version: {os.sys.version}")
    st.sidebar.write(f"Current working directory: {os.getcwd()}")
    st.sidebar.write(f"UPLOAD_DIR absolute path: {os.path.abspath(UPLOAD_DIR)}")
    st.sidebar.write(f"Files in UPLOAD_DIR: {os.listdir(UPLOAD_DIR) if os.path.exists(UPLOAD_DIR) else 'Directory not found'}")

# Initialize session state variables to persist data between reruns
if "custom_images_path" not in st.session_state:
    # Για το Streamlit Cloud, πάντα χρησιμοποίησε το UPLOAD_DIR
    if IS_STREAMLIT_CLOUD:
        st.session_state.custom_images_path = os.path.abspath(UPLOAD_DIR)
    else:
        # Για τοπική εκτέλεση, χρησιμοποίησε το IMAGES_PATH από το config.py
        st.session_state.custom_images_path = os.path.abspath(IMAGES_PATH)
if "force_reindex" not in st.session_state:
    st.session_state.force_reindex = False

# Load model with caching
@st.cache_resource(show_spinner=False)
def load_model():
    with st.spinner("Loading CLIP model..."):
        try:
            # Set a longer timeout for downloads in cloud environment
            if IS_STREAMLIT_CLOUD:
                import huggingface_hub
                huggingface_hub.constants.HF_HUB_DOWNLOAD_TIMEOUT = 300  # 5 minutes
            
            # Load the model
            return SentenceTransformer('clip-ViT-B-32')
        except Exception as e:
            st.error(f"Error loading CLIP model: {str(e)}")
            st.info("This could be due to network issues or memory constraints. Please try refreshing the page.")
            # Return a placeholder to avoid breaking everything
            return None

# Function to delete index files to force reindexing
def delete_index_files():
    """Delete index files to force reindexing"""
    try:
        if os.path.exists(OUTPUT_INDEX_PATH):
            os.remove(OUTPUT_INDEX_PATH)
            st.info(f"Removed index file: {OUTPUT_INDEX_PATH}")
        if os.path.exists(OUTPUT_INDEX_PATH + '.paths'):
            os.remove(OUTPUT_INDEX_PATH + '.paths')
            st.info(f"Removed paths file: {OUTPUT_INDEX_PATH + '.paths'}")
        if os.path.exists(OUTPUT_INDEX_PATH + '.metadata'):
            os.remove(OUTPUT_INDEX_PATH + '.metadata')
            st.info(f"Removed metadata file: {OUTPUT_INDEX_PATH + '.metadata'}")
    except Exception as e:
        st.error(f"Error removing index files: {str(e)}")

# Create embeddings and index (if it doesn't exist)
# NO caching decorator to ensure it runs every time
def prepare_index(_images_path=None):
    """
    Prepare FAISS index for image search
    
    Parameters:
    _images_path: Path to image directory
    """
    # Use custom path if provided
    images_path = _images_path or st.session_state.custom_images_path
    
    # Check if index already exists and no reindex is requested
    if not st.session_state.force_reindex and os.path.exists(OUTPUT_INDEX_PATH) and os.path.exists(OUTPUT_INDEX_PATH + '.paths'):
        model = load_model()
        
        # Load the existing index
        import faiss
        index = faiss.read_index(OUTPUT_INDEX_PATH)
        
        # Load image paths from the saved file
        with open(OUTPUT_INDEX_PATH + '.paths', 'r') as f:
            image_paths = [line.strip() for line in f.readlines()]
            
        # Load metadata if available
        metadata_list = []
        if os.path.exists(OUTPUT_INDEX_PATH + '.metadata'):
            try:
                with open(OUTPUT_INDEX_PATH + '.metadata', 'r') as f:
                    metadata_list = json.load(f)
                st.success(f"Loaded metadata for {len(metadata_list)} images")
            except Exception as e:
                st.warning(f"Error loading metadata: {e}")
                
        return model, index, image_paths, metadata_list
    else:
        # Create new index if it doesn't exist or reindex is requested
        with st.spinner("Preparing image index. This may take a while..."):
            if st.session_state.force_reindex:
                st.session_state.force_reindex = False  # Reset the flag
                
            model = load_model()
            
            # Check if images directory exists
            if not os.path.exists(images_path):
                st.error(f"Images directory not found: {images_path}")
                st.info("Please select a valid images directory in the sidebar.")
                
                # Try to create the directory
                try:
                    os.makedirs(images_path, exist_ok=True)
                    st.success(f"Created directory: {images_path}")
                except Exception as e:
                    st.error(f"Unable to create directory: {str(e)}")
                
                if IS_STREAMLIT_CLOUD:
                    # In cloud environment, fall back to UPLOAD_DIR
                    backup_path = os.path.abspath(UPLOAD_DIR)
                    st.info(f"Falling back to upload directory: {backup_path}")
                    images_path = backup_path
                else:
                    return model, None, [], []
            
            try:
                embeddings, image_paths, metadata_list = generate_clip_embeddings(images_path, model)
                
                if not image_paths:
                    st.warning(f"No images found in {images_path}")
                    if IS_STREAMLIT_CLOUD:
                        st.info("Please upload some images using the Image Search tab.")
                    return model, None, [], []
            except Exception as e:
                st.error(f"Error generating embeddings: {str(e)}")
                return model, None, [], []
                
            index = create_faiss_index(embeddings, image_paths, OUTPUT_INDEX_PATH, metadata_list)
            return model, index, image_paths, metadata_list

# Display images in a grid
def display_images(results, scores=None, metadata_list=None, image_paths=None):
    if not results:
        st.warning("😢 No similar images found.")
        return
    
    # Display results in a grid
    cols = 3
    rows = (len(results) + cols - 1) // cols
    
    for row in range(rows):
        cols_container = st.columns(cols)
        for col in range(cols):
            idx = row * cols + col
            if idx < len(results):
                with cols_container[col]:
                    try:
                        img = Image.open(results[idx])
                        st.image(img, caption=f"Score: {scores[idx]:.2f}" if scores else "", use_container_width=True)
                        st.write(f"📂 {os.path.basename(results[idx])}")
                        
                        # Show metadata if available
                        if metadata_list and image_paths:
                            # Find the metadata for this image by matching the path
                            image_path = results[idx]
                            meta_idx = None
                            for i, path in enumerate(image_paths):
                                if path == image_path and i < len(metadata_list):
                                    meta_idx = i
                                    break
                            
                            if meta_idx is not None and meta_idx < len(metadata_list):
                                meta = metadata_list[meta_idx]
                                with st.expander("📋 Image Metadata"):
                                    if 'date_taken' in meta:
                                        st.write(f"📅 Date: {meta['date_taken']}")
                                    elif 'file_date' in meta:
                                        st.write(f"📅 File Date: {meta['file_date']}")
                                    
                                    if 'camera_make' in meta or 'camera_model' in meta:
                                        camera_info = []
                                        if 'camera_make' in meta:
                                            camera_info.append(meta['camera_make'])
                                        if 'camera_model' in meta:
                                            camera_info.append(meta['camera_model'])
                                        st.write(f"📷 Camera: {' '.join(camera_info)}")
                                    
                                    if 'latitude' in meta and 'longitude' in meta:
                                        st.write(f"📍 Location: {meta['latitude']:.6f}, {meta['longitude']:.6f}")
                                        # Create a simple map link to view location
                                        map_url = f"https://www.openstreetmap.org/?mlat={meta['latitude']}&mlon={meta['longitude']}&zoom=15"
                                        st.markdown(f"[View on Map]({map_url})")
                                        
                                    # Display dimensions
                                    if 'width' in meta and 'height' in meta:
                                        st.write(f"📐 Dimensions: {meta['width']}x{meta['height']}")
                    except Exception as e:
                        st.error(f"Error loading image: {e}")

# Function to extract embeddings from FAISS index or regenerate them
def get_embeddings_for_tsne(model, index, image_paths):
    """
    Get embeddings for t-SNE visualization, either by regenerating them from images
    
    Args:
        model: CLIP model for encoding
        index: FAISS index (not used for regeneration but kept for API compatibility)
        image_paths: List of image paths
    
    Returns:
        Numpy array of embeddings
    """
    # We'll regenerate embeddings from the images directly
    # This avoids the 'reconstruct not implemented' error with certain FAISS indices
    embeddings = []
    valid_paths = []
    
    with st.spinner(f"Processing images for t-SNE visualization (0/{len(image_paths)})..."):
        for i, img_path in enumerate(image_paths):
            try:
                # Update progress message
                if i % 10 == 0:
                    st.spinner(f"Processing images for t-SNE visualization ({i}/{len(image_paths)})...")
                
                # Generate embedding
                image = Image.open(img_path)
                embedding = model.encode(image)
                
                # Store embedding and path
                embeddings.append(embedding)
                valid_paths.append(img_path)
            except Exception as e:
                pass  # Skip problematic images silently
    
    # Convert to numpy array
    if embeddings:
        embeddings_array = np.array(embeddings)
        return embeddings_array, valid_paths
    else:
        raise ValueError("No valid embeddings could be generated")

# Function to create t-SNE visualization of all images
def create_tsne_visualization(embeddings, image_paths, perplexity=15, n_iter=1000, thumbnail_size=(50, 50)):
    """
    Create a t-SNE visualization of all images based on their embeddings
    
    Args:
        embeddings: Image embeddings extracted from FAISS index
        image_paths: List of image paths corresponding to embeddings
        perplexity: t-SNE perplexity parameter
        n_iter: Number of iterations for t-SNE
        thumbnail_size: Size of image thumbnails in the plot
    
    Returns:
        Matplotlib figure with t-SNE visualization
    """
    # Apply t-SNE dimensionality reduction
    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, random_state=42)
    tsne_results = tsne.fit_transform(embeddings)
    
    # Create the figure and axis
    fig, ax = plt.subplots(figsize=(12, 12))
    
    # Plot each point and add image thumbnail
    for i, (x, y) in enumerate(tsne_results):
        try:
            # Open and resize image
            img = Image.open(image_paths[i])
            img = img.resize(thumbnail_size, Image.Resampling.LANCZOS)
            
            # Convert PIL image to a numpy array
            img_arr = np.array(img)
            
            # Create an OffsetImage with the thumbnail
            imagebox = OffsetImage(img_arr, zoom=1)
            
            # Create an AnnotationBbox with the image
            ab = AnnotationBbox(imagebox, (x, y), frameon=True, pad=0.1)
            
            # Add the annotation box to the plot
            ax.add_artist(ab)
            
        except Exception as e:
            # If we can't load the image, just plot a point
            ax.scatter(x, y, c='red', s=10, alpha=0.5)
            
    # Set plot properties
    ax.set_title('t-SNE Visualization of Images')
    ax.set_xlabel('t-SNE dimension 1')
    ax.set_ylabel('t-SNE dimension 2')
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # Adjust axis limits to provide some padding
    ax.set_xlim(tsne_results[:, 0].min() - 10, tsne_results[:, 0].max() + 10)
    ax.set_ylim(tsne_results[:, 1].min() - 10, tsne_results[:, 1].max() + 10)
    
    # Make the plot tight
    plt.tight_layout()
    
    return fig

# Function to convert matplotlib figure to Streamlit-compatible image
def fig_to_image(fig):
    """Convert matplotlib figure to image for display in Streamlit"""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    buf.seek(0)
    return buf

# Streamlit UI
st.title("🔍 CLIP Image Search with FAISS")
st.markdown("Search images using natural language or upload an image for similarity search.")

# Show current folder info and image count
try:
    image_count = sum(1 for f in os.listdir(st.session_state.custom_images_path) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png')))
    
    if image_count > 0:
        st.info(f"💾 Current images folder: {st.session_state.custom_images_path} ({image_count} images)")
    else:
        st.warning(f"💾 Current images folder: {st.session_state.custom_images_path} (No images found)")
        
        if IS_STREAMLIT_CLOUD:
            st.info("👉 Please upload some images using the Image Search tab.")
except Exception as e:
    st.info(f"💾 Current images folder: {st.session_state.custom_images_path}")
    st.warning(f"Unable to access image folder: {str(e)}")

# Add sidebar for additional options
with st.sidebar:
    st.header("⚙️ Search Options")
    
    # Directory selection
    st.subheader("📁 Image Directory")
    current_dir = st.text_input("Current image folder", value=st.session_state.custom_images_path)
      # Change directory button
    if st.button("📂 Set Folder Path") and current_dir:
        try:
            # Create the directory if it doesn't exist
            os.makedirs(current_dir, exist_ok=True)
            
            # Delete index files to force regeneration
            delete_index_files()
            
            # Update path
            st.session_state.custom_images_path = current_dir
            st.session_state.force_reindex = True
            st.success(f"Image folder set to: {current_dir}")
            st.rerun()
        except Exception as e:
            st.error(f"Error setting folder path: {str(e)}")
    
    st.markdown("---")
    top_k = st.slider("Number of results", min_value=1, max_value=50, value=9)
    threshold = st.slider("🎯 Similarity threshold", min_value=0.0, max_value=1.0, value=0.2, step=0.01)
    
    # Add scaling options
    enable_scaling = st.checkbox("📈 Enable softmax scaling", value=False)
    scaling_method = 'none'
    
    if enable_scaling:
        scaling_method = 'softmax'
        st.info("Softmax scaling normalizes the similarity distribution and can change ranking.")
        
        # Add temperature control for softmax
        softmax_temp = st.slider(
            "Softmax temperature", 
            min_value=0.01, 
            max_value=0.5,
            value=0.1, 
            step=0.01, 
            help="Lower values (0.01-0.05) create sharper contrast between results. Higher values give more uniform distribution."
        )
    
    # Add metadata filtering UI
    st.markdown("---")
    st.subheader("🏷️ Metadata Filters")
    
    # Enable metadata filtering
    use_metadata = st.checkbox("Enable metadata filtering", value=False)
    
    metadata_filters = {}
    
    if use_metadata:
        # Date filter
        st.write("📅 Date Range")
        col1, col2 = st.columns(2)
        start_date = col1.date_input("Start date", value=None)
        end_date = col2.date_input("End date", value=None)
        
        if start_date and end_date:
            metadata_filters['date_range'] = {
                'start': datetime.datetime.combine(start_date, datetime.time.min),
                'end': datetime.datetime.combine(end_date, datetime.time.max)
            }
        
        # Location filter
        st.write("📍 Location")
        use_location = st.checkbox("Filter by location", value=False)
        
        if use_location:
            # Allow user to enter location name instead of coordinates
            location_input_type = st.radio(
                "Location input method",
                ["Location Name", "Coordinates"],
                horizontal=True
            )
            
            if location_input_type == "Location Name":
                location_name = st.text_input("Enter location (e.g., Athens, Greece)")
                radius = st.slider("Radius (km)", min_value=1, max_value=1000, value=10)
                
                # Only attempt geocoding if a location name is provided
                if location_name:
                    coordinates = geocode_location(location_name)
                    if coordinates:
                        lat, lon = coordinates
                        # Display the resolved coordinates
                        st.success(f"Found coordinates: {lat:.6f}, {lon:.6f}")
                        
                        metadata_filters['location_radius'] = {
                            'lat': lat,
                            'lon': lon,
                            'km': radius
                        }
                    else:
                        st.error(f"Could not find coordinates for '{location_name}'. Try a different location name.")
            else:
                # Original coordinate input
                col1, col2 = st.columns(2)
                lat = col1.number_input("Latitude", value=0.0, format="%.6f")
                lon = col2.number_input("Longitude", value=0.0, format="%.6f")
                radius = st.slider("Radius (km)", min_value=1, max_value=1000, value=10)
                
                # Only add location filter if coordinates are not default (0,0)
                if lat != 0.0 or lon != 0.0:
                    metadata_filters['location_radius'] = {
                        'lat': lat,
                        'lon': lon,
                        'km': radius
                    }
        
        # Camera filter
        st.write("📷 Camera")
        camera_make = st.text_input("Camera make (e.g. Canon, Nikon)")
        camera_model = st.text_input("Camera model")
        
        if camera_make or camera_model:
            metadata_filters['camera'] = {}
            if camera_make:
                metadata_filters['camera']['make'] = camera_make
            if camera_model:
                metadata_filters['camera']['model'] = camera_model
    
    # Add option to force reindex
    if st.button("🔄 Force Reindex Images"):
        # Delete index files to force regeneration
        delete_index_files()
        st.session_state.force_reindex = True
        st.success("Reindexing images...")
        st.rerun()

# Create tabs for different search modes
tab1, tab2, tab3, tab4 = st.tabs(["📝 Text Search", "🖼️ Image Search", "🏷️ Metadata-only Search", "🔬 t-SNE Visualization"])

# Text search tab
with tab1:
    query_text = st.text_input("📎 Enter a text query (e.g. 'a cat'):", value="")
    
    col1, col2 = st.columns(2)
    search_button = col1.button("🔎 Search by Text", key="text_search")
    clear_button = col2.button("🧹 Clear Results", key="clear_text")
    
    if search_button and query_text:
        with st.spinner("Searching for similar images..."):
            try:
                model, index, image_paths, metadata_list = prepare_index()
                
                if index is not None and image_paths:
                    # Get metadata filters if enabled
                    filters = metadata_filters if use_metadata else None
                    
                    # Get the softmax temperature if enabled
                    softmax_temperature = softmax_temp if enable_scaling else 0.1
                    
                    query, results, scores = retrieve_similar_images(
                        query_text, 
                        model, 
                        index, 
                        image_paths, 
                        metadata_list=metadata_list,
                        similarity_threshold=threshold, 
                        top_k=top_k,
                        scaling_method=scaling_method,
                        metadata_filters=filters,
                        softmax_temperature=softmax_temperature
                    )
                    
                    if results:
                        st.success(f"Found {len(results)} images above the threshold.")
                        display_images(results, scores, metadata_list, image_paths)
                    else:
                        st.warning("No similar images found.")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
    
    if clear_button:
        st.rerun()

# Image search tab
with tab2:
    st.write("### 📤 Upload Images")
    
    # Add option to upload multiple images for the database
    if IS_STREAMLIT_CLOUD:
        st.info("Running in Streamlit Cloud. You can upload images directly here to build your image database.")
        
        uploaded_files = st.file_uploader("Upload images to your database:", 
                                         type=["jpg", "jpeg", "png"], 
                                         accept_multiple_files=True)
        
        if uploaded_files:
            with st.spinner(f"Adding {len(uploaded_files)} images to your database..."):
                for uploaded_file in uploaded_files:
                    # Create a unique filename
                    unique_id = str(uuid.uuid4())
                    file_ext = uploaded_file.name.split('.')[-1]
                    save_path = os.path.join(UPLOAD_DIR, f"{unique_id}.{file_ext}")
                    
                    # Save the file
                    with open(save_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                
                # Force reindexing after adding new images
                st.session_state.force_reindex = True
                st.success(f"Added {len(uploaded_files)} images to the database.")
                st.info("The system will automatically reindex your images.")
    
    st.write("### 🔎 Search by Image")
    query_img = st.file_uploader("Upload an image to search for similar images:", type=["jpg", "jpeg", "png"], key="search_image")
    
    col1, col2 = st.columns(2)
    search_button = col1.button("🔎 Search by Image", key="img_search")
    clear_button = col2.button("🧹 Clear Results", key="clear_img")
    
    if search_button and query_img:
        with st.spinner("Searching for similar images..."):
            try:
                # Show the query image first, so it appears even if search fails
                query_image = Image.open(query_img)
                st.image(query_image, caption="🔍 Query Image", width=300)
                
                # Prepare the index
                model, index, image_paths, metadata_list = prepare_index()
                
                # Check if we have any images indexed
                if not image_paths:
                    if IS_STREAMLIT_CLOUD:
                        st.warning("No images in the database. Please upload some images first.")
                    else:
                        st.warning(f"No images found in {st.session_state.custom_images_path}")
                    st.stop()
                
                if index is not None:
                    # Create a unique filename for the uploaded image
                    unique_id = str(uuid.uuid4())
                    file_ext = query_img.name.split('.')[-1] if '.' in query_img.name else 'jpg'
                    temp_img_path = os.path.join(UPLOAD_DIR, f"{unique_id}.{file_ext}")
                    
                    # Save the uploaded file temporarily
                    with open(temp_img_path, "wb") as f:
                        f.write(query_img.getbuffer())
                    
                    # Get metadata filters if enabled
                    filters = metadata_filters if use_metadata else None
                    
                    # Get the softmax temperature if enabled
                    softmax_temperature = softmax_temp if enable_scaling else 0.1
                    
                    # Perform the search
                    query, results, scores = retrieve_similar_images(
                        query_image, 
                        model, 
                        index, 
                        image_paths, 
                        metadata_list=metadata_list,
                        similarity_threshold=threshold, 
                        top_k=top_k,
                        scaling_method=scaling_method,
                        metadata_filters=filters,
                        softmax_temperature=softmax_temperature
                    )
                    
                    # Show results
                    if results:
                        st.success(f"Found {len(results)} similar images above the threshold.")
                        display_images(results, scores, metadata_list, image_paths)
                    else:
                        st.warning("No similar images found that meet your criteria.")
                        st.info("Try adjusting the similarity threshold or metadata filters.")
                else:
                    st.error("Failed to prepare the image index.")
            except Exception as e:
                st.error(f"An error occurred during the search: {str(e)}")
                st.info("If you're seeing this error for the first time, try searching again.")
    
    if clear_button:
        st.rerun()

# Metadata-only search tab
with tab3:
    st.info("🏷️ Search images using only metadata without text or image query")
    
    if not use_metadata:
        st.warning("Enable 'Metadata filtering' in the sidebar and select your desired filters.")
    
    col1, col2 = st.columns(2)
    search_button = col1.button("🔎 Search by Metadata", key="metadata_search")
    clear_button = col2.button("🧹 Clear Results", key="clear_metadata")
    
    if search_button:
        with st.spinner("Searching for images based on metadata..."):
            try:
                # Load the model, index and metadata but not needed
                # for the search, only for accessing the information
                model, index, image_paths, metadata_list = prepare_index()
                
                if image_paths and metadata_list:
                    if use_metadata and metadata_filters:
                        # Search using only metadata
                        filtered_paths, filtered_scores = search_by_metadata_only(
                            image_paths, 
                            metadata_list, 
                            metadata_filters,
                            top_k=top_k
                        )
                        
                        if filtered_paths:
                            st.success(f"Found {len(filtered_paths)} images that match the selected filters.")
                            display_images(filtered_paths, filtered_scores, metadata_list, image_paths)
                        else:
                            st.warning("No images found matching the selected filters.")
                    else:
                        st.error("You must enable 'Metadata filtering' and select at least one filter.")
            except Exception as e:
                st.error(f"An error occurred during the search: {str(e)}")
    
    if clear_button:
        st.rerun()

# t-SNE visualization tab
with tab4:
    st.info("🔬 Visualize image embeddings using t-SNE")
    
    # Controls for t-SNE parameters
    col1, col2 = st.columns(2)
    perplexity = col1.slider("Perplexity", min_value=5, max_value=50, value=30,
                          help="t-SNE perplexity parameter - influences how local vs global structure is preserved")
    n_iter = col2.slider("Iterations", min_value=250, max_value=2000, value=1000,
                      help="Number of iterations for t-SNE optimization")
    
    # Thumbnail size control
    thumbnail_size = st.slider("Thumbnail Size", min_value=20, max_value=100, value=50,
                            help="Size of image thumbnails in pixels")
    
    # Generate visualization button
    if st.button("Generate t-SNE Visualization"):
        with st.spinner("Generating t-SNE visualization..."):
            try:
                model, index, image_paths, metadata_list = prepare_index()
                
                if index is not None and image_paths:
                    # Get embeddings - we'll regenerate them from the images to avoid FAISS issues
                    embeddings, valid_paths = get_embeddings_for_tsne(model, index, image_paths)
                    
                    # Create visualization
                    fig = create_tsne_visualization(
                        embeddings, 
                        valid_paths, 
                        perplexity=perplexity, 
                        n_iter=n_iter, 
                        thumbnail_size=(thumbnail_size, thumbnail_size)
                    )
                    
                    # Display the visualization
                    st.pyplot(fig)
                    st.success(f"Visualization created with {len(valid_paths)} images")
                    st.info("Note: Image positions reflect semantic similarity based on CLIP embeddings")
                else:
                    st.warning("No index or images available for visualization.")
            except Exception as e:
                st.error(f"An error occurred during t-SNE visualization: {str(e)}")
                st.error("Try reducing the number of images or changing the parameters")

# Add footer with project information
st.markdown("---")
st.markdown("### 💡 About CLIP-FAISS")
st.markdown("""
This application uses OpenAI's CLIP (Contrastive Language-Image Pretraining) model to understand both text and images,
and FAISS (Facebook AI Similarity Search) for efficient similarity search among large collections of vectors.
""")