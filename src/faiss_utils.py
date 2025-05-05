# 📦 faiss_utils.py - Utility functions for FAISS + CLIP

import os
import numpy as np
from glob import glob
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import faiss
from typing import List, Tuple, Union, Optional, Literal, Dict, Any
import logging
from PIL.ExifTags import TAGS, GPSTAGS
import datetime
import json
import requests

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 🔧 Normalize vectors (L2)
def l2_normalize(vecs: np.ndarray) -> np.ndarray:
    """
    Normalize vectors to unit length using L2 normalization
    
    Args:
        vecs: Input vectors to normalize
        
    Returns:
        L2-normalized vectors
    """
    return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)

# 🧮 Apply softmax to scores
def apply_softmax_scaling(scores: np.ndarray, temperature: float = 0.1) -> np.ndarray:
    """
    Apply softmax scaling to similarity scores
    
    Args:
        scores: Input similarity scores
        temperature: Controls the "sharpness" of the distribution (lower values = sharper contrast)
            
    Returns:
        Softmax-scaled scores (sum to 1.0)
    """
    # Apply temperature to increase contrast (lower temp = higher contrast)
    # Subtract max for numerical stability
    exp_scores = np.exp((scores - np.max(scores)) / temperature)
    # Softmax calculation
    return exp_scores / np.sum(exp_scores)

# 🚀 Create CLIP embeddings for images
def generate_clip_embeddings(images_path: str, model, use_tqdm: bool = True) -> Tuple[List[np.ndarray], List[str], List[Dict[str, Any]]]:
    """
    Generate CLIP embeddings for all images in a directory
    
    Args:
        images_path: Path to the directory containing images
        model: CLIP model to use for encoding
        use_tqdm: Whether to show progress bar
        
    Returns:
        Tuple of (list of embeddings, list of image paths, list of metadata dicts)
    """
    # Find all image files
    image_paths = glob(os.path.join(images_path, '**/*.jpg'), recursive=True)
    image_paths.extend(glob(os.path.join(images_path, '**/*.jpeg'), recursive=True))
    image_paths.extend(glob(os.path.join(images_path, '**/*.png'), recursive=True))
    
    if not image_paths:
        raise ValueError(f"No images found in {images_path}")
    
    logger.info(f"Found {len(image_paths)} images to process")
    
    # Process all images
    embeddings = []
    valid_paths = []
    metadata_list = []
    
    # Use tqdm for progress indication if requested
    iterator = tqdm(image_paths, desc="Generating embeddings") if use_tqdm else image_paths
    
    for img_path in iterator:
        try:
            # Extract metadata first
            metadata = extract_image_metadata(img_path)
            
            # Generate embedding
            image = Image.open(img_path)
            embedding = model.encode(image)
            
            # Store all data
            embeddings.append(embedding)
            valid_paths.append(img_path)
            metadata_list.append(metadata)
            
        except Exception as e:
            logger.warning(f"Error processing {img_path}: {e}")
    
    logger.info(f"Successfully generated embeddings for {len(embeddings)} images")
    return embeddings, valid_paths, metadata_list

# 💾 Create FAISS index with normalized vectors
def create_faiss_index(embeddings: List[np.ndarray], image_paths: List[str], 
                     output_path: str, metadata_list: Optional[List[Dict[str, Any]]] = None) -> faiss.IndexIDMap:
    """
    Create and save a FAISS index for fast similarity search
    
    Args:
        embeddings: List of image embeddings
        image_paths: List of corresponding image paths
        output_path: Path to save the FAISS index
        metadata_list: Optional list of metadata dictionaries for each image
        
    Returns:
        FAISS index object
    """
    dimension = len(embeddings[0])
    index = faiss.IndexFlatIP(dimension)  # Inner product index for cosine similarity
    index = faiss.IndexIDMap(index)
    
    # Prepare vectors and normalize
    vectors = np.array(embeddings).astype(np.float32)
    vectors = l2_normalize(vectors)
    
    # Add vectors to the index
    index.add_with_ids(vectors, np.arange(len(vectors)))
    
    # Save the index to disk
    faiss.write_index(index, output_path)
    
    # Save image paths separately
    with open(output_path + '.paths', 'w') as f:
        for path in image_paths:
            f.write(path + '\n')
    
    # Save metadata separately if provided
    if metadata_list:
        with open(output_path + '.metadata', 'w') as f:
            json.dump(metadata_list, f)
    
    logger.info(f"✅ Index saved to {output_path}")
    return index

# 🔍 Search with query (text or image) and similarity threshold
def retrieve_similar_images(
    query: Union[str, Image.Image], 
    model, 
    index: faiss.Index, 
    image_paths: List[str], 
    metadata_list: Optional[List[Dict[str, Any]]] = None,
    similarity_threshold: float = 0.3,
    top_k: int = 50,
    scaling_method: Literal['none', 'softmax'] = 'none',
    metadata_filters: Optional[Dict[str, Any]] = None,
    softmax_temperature: float = 0.1
) -> Tuple[Union[str, Image.Image], List[str], List[float]]:
    """
    Retrieve similar images using text or image query with optional metadata filtering
    
    Args:
        query: Text query or image
        model: CLIP model for encoding
        index: FAISS index to search in
        image_paths: List of image paths corresponding to the index
        metadata_list: List of metadata dictionaries (must be same length as image_paths)
        similarity_threshold: Minimum similarity score to include in results
        top_k: Maximum number of results to return
        scaling_method: Method to use for scaling scores ('none' or 'softmax')
        metadata_filters: Dictionary of metadata filters to apply
        softmax_temperature: Controls the "sharpness" of the softmax distribution (lower = sharper)
        
    Returns:
        Tuple of (query, filtered_paths, filtered_scores)
    """
    # Handle file path query
    if isinstance(query, str) and query.lower().endswith(('.jpg', '.jpeg', '.png')):
        query = Image.open(query)
    
    # Encode query
    query_vector = model.encode(query)
    query_vector = query_vector.astype(np.float32).reshape(1, -1)
    query_vector = l2_normalize(query_vector)
    
    # Search in the index - get many more results if we'll filter by metadata
    max_results = len(image_paths)
    search_k = min(max_results, top_k * 10) if metadata_filters and metadata_list else min(max_results, top_k)
    D, I = index.search(query_vector, search_k)  # D: distances, I: indices
    
    # Apply score scaling if specified
    original_scores = D[0].copy()  # Save original scores for logging
    
    if scaling_method == 'softmax':
        logger.info(f"Applying softmax scaling with temperature {softmax_temperature}")
        # Apply softmax scaling with user-defined temperature
        softmax_scores = apply_softmax_scaling(D[0], temperature=softmax_temperature)
        scaling_factor = 1.0 / max(softmax_scores) if max(softmax_scores) > 0 else 1.0
        D[0] = softmax_scores * scaling_factor
    
    # Log scores
    logger.info("\n🎯 Similarity scores (top 10):")
    for i, (orig, scaled, idx) in enumerate(zip(original_scores[:10], D[0][:10], I[0][:10])):
        if idx < len(image_paths):
            if scaling_method != 'none':
                logger.info(f"Original: {orig:.4f} → Scaled: {scaled:.4f} → {image_paths[idx]}")
            else:
                logger.info(f"{orig:.4f} → {image_paths[idx]}")
    
    # Apply metadata filters if specified
    filtered_indices = []
    for i, idx in enumerate(I[0]):
        if idx >= len(image_paths):
            continue
            
        if metadata_filters and metadata_list:
            if idx >= len(metadata_list):
                continue
                
            metadata = metadata_list[idx]
            include = _apply_metadata_filters(metadata, metadata_filters)
            
            if include and D[0][i] >= similarity_threshold:
                filtered_indices.append((idx, D[0][i]))
        else:
            if D[0][i] >= similarity_threshold:
                filtered_indices.append((idx, D[0][i]))
    
    # Sort by similarity score and limit to top_k
    filtered_indices.sort(key=lambda x: x[1], reverse=True)
    filtered_indices = filtered_indices[:top_k]
    
    # Extract paths and scores
    filtered_paths = [image_paths[idx] for idx, _ in filtered_indices]
    filtered_scores = [float(score) for _, score in filtered_indices]
    
    return query, filtered_paths, filtered_scores

# Helper function to apply metadata filters
def _apply_metadata_filters(metadata: Dict[str, Any], filters: Dict[str, Any]) -> bool:
    """
    Apply metadata filters to an image's metadata
    
    Returns True if the metadata matches all filters
    """
    # If no metadata or filters, include by default
    if not metadata or not filters:
        return True
        
    # Date range filter
    if 'date_range' in filters:
        date_filter = filters['date_range']
        date_key = 'date_taken' if 'date_taken' in metadata else 'file_date'
        
        if date_key in metadata:
            try:
                # Parse the date string
                if isinstance(metadata[date_key], str):
                    # Try different date formats
                    for fmt in ('%Y-%m-%d %H:%M:%S', '%Y:%m:%d %H:%M:%S'):
                        try:
                            img_date = datetime.datetime.strptime(metadata[date_key], fmt)
                            break
                        except ValueError:
                            continue
                else:
                    img_date = metadata[date_key]  # Already a datetime object
                    
                # Parse filter dates
                start_date = datetime.datetime.fromisoformat(date_filter['start']) if isinstance(date_filter['start'], str) else date_filter['start']
                end_date = datetime.datetime.fromisoformat(date_filter['end']) if isinstance(date_filter['end'], str) else date_filter['end']
                
                if img_date < start_date or img_date > end_date:
                    return False
            except Exception as e:
                logger.warning(f"Error comparing dates: {e}")
                return False
        else:
            # If the date filter is applied but image has no date metadata, exclude it
            return False
    
    # Location radius filter
    if 'location_radius' in filters:
        # Check if image has location data - exclude images without location data
        if not ('latitude' in metadata and 'longitude' in metadata):
            return False
            
        try:
            filter_lat = filters['location_radius']['lat']
            filter_lon = filters['location_radius']['lon']
            radius_km = filters['location_radius']['km']
            
            # Calculate distance using Haversine formula
            from math import sin, cos, sqrt, atan2, radians
            R = 6371  # Earth radius in km
            
            lat1, lon1 = radians(filter_lat), radians(filter_lon)
            lat2, lon2 = radians(metadata['latitude']), radians(metadata['longitude'])
            
            dlon = lon2 - lon1
            dlat = lat2 - lat1
            
            a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
            c = 2 * atan2(sqrt(a), sqrt(1-a))
            distance = R * c
            
            if distance > radius_km:
                return False
        except Exception as e:
            logger.warning(f"Error calculating location distance: {e}")
            return False
    
    # Camera make/model filter
    if 'camera' in filters:
        camera_filter = filters['camera']
        
        if 'make' in camera_filter:
            if 'camera_make' not in metadata:
                return False
            if camera_filter['make'].lower() not in metadata['camera_make'].lower():
                return False
                
        if 'model' in camera_filter:
            if 'camera_model' not in metadata:
                return False
            if camera_filter['model'].lower() not in metadata['camera_model'].lower():
                return False
    
    # All filters passed
    return True

# 🖼️ Visualize results in a grid
def visualize_results(query: Union[str, Image.Image], retrieved_images: List[str], scores: Optional[List[float]] = None):
    """
    Visualize query and retrieved images in a grid layout
    
    Args:
        query: Text query or image
        retrieved_images: List of image paths to visualize
        scores: Optional list of similarity scores
    """
    if not retrieved_images:
        print("😢 No similar images found.")
        return
    
    total = len(retrieved_images)
    cols = 5
    rows = max(2, (total + 1) // cols + 1)
    plt.figure(figsize=(cols * 5, rows * 5))
    
    # Show the query
    plt.subplot(rows, cols, 1)
    if isinstance(query, Image.Image):
        plt.imshow(query)
        plt.title("Query Image")
    else:
        plt.text(0.5, 0.5, f"Query:\n\n'{query}'", fontsize=14, ha='center', va='center')
    plt.axis('off')
    
    # Show results
    for i, img_path in enumerate(retrieved_images):
        plt.subplot(rows, cols, i + 2)
        try:
            img = Image.open(img_path)
            plt.imshow(img)
            title = f"Match {i + 1}"
            if scores and i < len(scores):
                title += f"\nScore: {scores[i]:.2f}"
            plt.title(title)
        except Exception as e:
            plt.text(0.5, 0.5, f"Error loading image:\n{str(e)}", ha='center', va='center', color='red')
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# 📋 Extract metadata from images
def extract_image_metadata(image_path: str) -> Dict[str, Any]:
    """
    Extract metadata from an image including geo location, date taken, camera info
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Dictionary containing metadata fields
    """
    metadata = {}
    
    try:
        # Open image to extract EXIF data
        with Image.open(image_path) as img:
            # Basic image properties
            metadata['width'] = img.width
            metadata['height'] = img.height
            metadata['format'] = img.format
            
            # Extract EXIF data if available
            if hasattr(img, '_getexif') and img._getexif():
                exif_data = img._getexif()
                
                # Process standard EXIF tags
                for tag_id, value in exif_data.items():
                    tag = TAGS.get(tag_id, tag_id)
                    
                    # Handle specific tags of interest
                    if tag == 'DateTimeOriginal':
                        metadata['date_taken'] = value
                    elif tag == 'Make':
                        metadata['camera_make'] = value
                    elif tag == 'Model':
                        metadata['camera_model'] = value
                    elif tag == 'GPSInfo':
                        # Process GPS data
                        gps_data = {}
                        for gps_tag_id, gps_value in value.items():
                            gps_tag = GPSTAGS.get(gps_tag_id, gps_tag_id)
                            gps_data[gps_tag] = gps_value
                        
                        # Extract coordinates if available
                        if all(key in gps_data for key in ['GPSLatitude', 'GPSLatitudeRef', 'GPSLongitude', 'GPSLongitudeRef']):
                            lat = _convert_to_decimal_degrees(gps_data['GPSLatitude'])
                            lon = _convert_to_decimal_degrees(gps_data['GPSLongitude'])
                            
                            # Apply reference direction
                            if gps_data['GPSLatitudeRef'] == 'S':
                                lat = -lat
                            if gps_data['GPSLongitudeRef'] == 'W':
                                lon = -lon
                                
                            metadata['latitude'] = lat
                            metadata['longitude'] = lon
            
            # Get file creation/modification time as fallback for date
            if 'date_taken' not in metadata:
                try:
                    timestamp = os.path.getmtime(image_path)
                    metadata['file_date'] = datetime.datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
                except:
                    pass
                
    except Exception as e:
        logger.warning(f"Error extracting metadata from {image_path}: {e}")
    
    return metadata

# Helper function for GPS coordinate conversion
def _convert_to_decimal_degrees(dms):
    """Convert GPS coordinates in (degrees, minutes, seconds) to decimal degrees"""
    degrees = dms[0]
    minutes = dms[1] / 60.0
    seconds = dms[2] / 3600.0
    return degrees + minutes + seconds

# 🗺️ Geocoding: Convert location name to coordinates
def geocode_location(location_name: str) -> Optional[Tuple[float, float]]:
    """
    Convert a location name to geographical coordinates (latitude, longitude)
    using the OpenStreetMap Nominatim API
    
    Args:
        location_name: The name of the location (e.g., "Athens, Greece")
    
    Returns:
        Tuple of (latitude, longitude) if successful, None otherwise
    """
    try:
        # Use OpenStreetMap Nominatim API for geocoding
        # Note: According to Nominatim usage policy, include a valid user-agent and limit requests
        headers = {
            'User-Agent': 'CLIP-FAISS-ImageSearch/1.0',
        }
        
        params = {
            'q': location_name,
            'format': 'json',
            'limit': 1
        }
        
        url = "https://nominatim.openstreetmap.org/search"
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        
        data = response.json()
        
        if data and len(data) > 0:
            lat = float(data[0]['lat'])
            lon = float(data[0]['lon'])
            logger.info(f"Geocoded '{location_name}' to coordinates: {lat}, {lon}")
            return (lat, lon)
        else:
            logger.warning(f"Could not geocode location: {location_name}")
            return None
            
    except Exception as e:
        logger.error(f"Error geocoding location: {str(e)}")
        return None

# 🏷️ Search images based only on metadata without semantic query
def search_by_metadata_only(
    image_paths: List[str], 
    metadata_list: List[Dict[str, Any]], 
    metadata_filters: Dict[str, Any],
    top_k: int = 50
) -> Tuple[List[str], List[float]]:
    """
    Search for images using only metadata filters without a semantic query
    
    Args:
        image_paths: List of image paths
        metadata_list: List of metadata dictionaries for each image
        metadata_filters: Dictionary of metadata filters to apply
        top_k: Maximum number of results to return
        
    Returns:
        Tuple of (filtered_paths, filtered_scores)
    """
    if not metadata_filters:
        # If no filters provided, return all images with equal scores
        return image_paths[:top_k], [1.0] * min(len(image_paths), top_k)
    
    # Apply metadata filters
    filtered_indices = []
    
    for idx, metadata in enumerate(metadata_list):
        if idx >= len(image_paths):
            continue
            
        include = _apply_metadata_filters(metadata, metadata_filters)
        if include:
            filtered_indices.append(idx)
    
    # Limit to top_k
    filtered_indices = filtered_indices[:top_k]
    
    # Extract paths and assign equal scores of 1.0
    filtered_paths = [image_paths[idx] for idx in filtered_indices]
    filtered_scores = [1.0] * len(filtered_paths)
    
    return filtered_paths, filtered_scores