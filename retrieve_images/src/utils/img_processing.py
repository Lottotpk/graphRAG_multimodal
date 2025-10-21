"""
Shared image processing utilities
Includes the preprocessing functions from the original InternVL implementation
"""

import os
import glob
from typing import List, Tuple
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from multiprocessing import cpu_count
import threading

import logging
from utils.logging_config import setup_logger

setup_logger()
logger = logging.getLogger(__name__)

# Constants from InternVL preprocessing
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    """Build image transformation pipeline"""
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    """Find the closest aspect ratio for dynamic preprocessing"""
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    """Dynamically preprocess image into tiles"""
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    """Load and preprocess image for InternVL model"""
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

def find_image_files(image_directory: str, supported_extensions: List[str]) -> List[str]:
    """
    Find all supported image files in directory (recursive)
    
    Args:
        image_directory: Directory to search
        supported_extensions: List of supported file extensions
        
    Returns:
        List[str]: List of image file paths
    """
    if not os.path.exists(image_directory):
        raise ValueError(f"Directory does not exist: {image_directory}")
    
    image_files = []
    
    for ext in supported_extensions:
        pattern = os.path.join(image_directory, '**', f'*{ext}')
        files = glob.glob(pattern, recursive=True)
        image_files.extend(files)
        
        # Also search without recursive for files directly in the directory
        pattern = os.path.join(image_directory, f'*{ext}')
        files = glob.glob(pattern)
        image_files.extend(files)
    
    # Remove duplicates and sort
    image_files = sorted(list(set(image_files)))
    
    return image_files

def load_image_with_error_handling(image_path: str, max_tiles: int = 12, input_size: int = 448) -> Tuple[torch.Tensor, dict]:
    """
    Load image with error handling and metadata collection
    
    Args:
        image_path: Path to image file
        max_tiles: Maximum number of tiles
        input_size: Input size for processing
        
    Returns:
        Tuple of (pixel_values, metadata)
    """
    try:
        # Use original load_image function
        pixel_values = load_image(image_path, max_num=max_tiles, input_size=input_size)
        
        # Collect metadata
        with Image.open(image_path) as img:
            width, height = img.size
            mode = img.mode
            
        metadata = {
            'image_path': image_path,
            'original_size': (width, height),
            'image_mode': mode,
            'num_tiles': pixel_values.shape[0],
            'tile_size': (input_size, input_size),
            'file_size_bytes': os.path.getsize(image_path),
            'load_success': True,
            'error': None
        }
        
        return pixel_values, metadata
        
    except Exception as e:
        # Return dummy data and error info
        dummy_pixel_values = torch.zeros((1, 3, input_size, input_size))
        metadata = {
            'image_path': image_path,
            'original_size': None,
            'image_mode': None,
            'num_tiles': 0,
            'tile_size': (input_size, input_size),
            'file_size_bytes': 0 if not os.path.exists(image_path) else os.path.getsize(image_path),
            'load_success': False,
            'error': str(e)
        }
        
        return dummy_pixel_values, metadata

def batch_image_paths(image_paths: List[str], batch_size: int) -> List[List[str]]:
    """
    Split image paths into batches
    
    Args:
        image_paths: List of image paths
        batch_size: Size of each batch
        
    Returns:
        List of batches
    """
    batches = []
    for i in range(0, len(image_paths), batch_size):
        batch = image_paths[i:i + batch_size]
        batches.append(batch)
    return batches

def validate_image_file(image_path: str) -> bool:
    """
    Validate if image file can be loaded
    
    Args:
        image_path: Path to image file
        
    Returns:
        bool: True if valid image file
    """
    try:
        with Image.open(image_path) as img:
            img.verify()
        return True
    except Exception:
        return False

def load_images_parallel(image_paths: List[str], max_tiles: int = 12, input_size: int = 448, 
                        max_workers: int = None) -> List[Tuple[torch.Tensor, dict]]:
    """
    Load multiple images in parallel using CPU threads
    
    Args:
        image_paths: List of image paths to load
        max_tiles: Maximum number of tiles per image
        input_size: Input size for processing
        max_workers: Number of worker threads (default: CPU count)
        
    Returns:
        List of (pixel_values, metadata) tuples
    """
    if max_workers is None:
        max_workers = min(cpu_count(), len(image_paths), 8)  # Cap at 8 to avoid overload
    
    def load_single_image(path):
        return load_image_with_error_handling(path, max_tiles, input_size)
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(load_single_image, image_paths))
    
    return results

def process_images_batch_parallel(image_paths: List[str], extractor, tracker, 
                                 max_workers: int = None) -> List[dict]:
    """
    Process a batch of images with parallel loading and GPU extraction
    
    Args:
        image_paths: List of image paths
        extractor: Embedding extractor instance
        tracker: Performance tracker
        max_workers: Number of worker threads for loading
        
    Returns:
        List of result dictionaries with embeddings and metadata
    """
    import time
    
    if max_workers is None:
        max_workers = min(cpu_count(), len(image_paths), 8)
    
    batch_start_time = time.time()
    
    # Phase 1: Parallel image loading on CPU
    logger.info(f"Loading {len(image_paths)} images with {max_workers} CPU threads...")
    load_start = time.time()
    
    loaded_results = load_images_parallel(image_paths, max_workers=max_workers)
    
    load_time = time.time() - load_start
    logger.info(f"Image loading completed in {load_time:.2f}s ({len(image_paths)/load_time:.1f} images/s)")
    
    # Separate successful and failed loads
    successful_tensors = []
    successful_metadata = []
    results = []
    
    for pixel_values, metadata in loaded_results:
        if metadata['load_success']:
            successful_tensors.append(pixel_values)
            successful_metadata.append(metadata)
            tracker.record_image_processing(True, loading_time=load_time / len(image_paths))
        else:
            logger.info(f"Failed to load: {metadata['image_path']} - {metadata['error']}")
            tracker.record_image_processing(False, loading_time=load_time / len(image_paths))
    
    # Phase 2: GPU batch extraction
    if successful_tensors:
        logger.info(f"Extracting embeddings for {len(successful_tensors)} images on GPU...")
        extract_start = time.time()

        if hasattr(extractor, 'outputs_token_level') and extractor.outputs_token_level():
            batch_outputs = extractor.extract_batch_token_embeddings(successful_tensors)
            extract_time = time.time() - extract_start
            logger.info(f"GPU token extraction completed in {extract_time:.2f}s ({len(successful_tensors)/extract_time:.1f} images/s)")
            per_image_extract_time = extract_time / max(1, len(successful_tensors))
            for out, metadata in zip(batch_outputs, successful_metadata):
                # Record per-image embeddings count and dim
                num_tokens = int(out['embeddings'].shape[0])
                embedding_dim = int(out['embeddings'].shape[1]) if out['embeddings'].ndim == 2 else 0
                tracker.record_embeddings_per_image(num_tokens, embedding_dim)
                pooled_embedding = None
                # Only compute pooled when extractor declares include_pooled
                if getattr(extractor, 'include_pooled', False):
                    # Compute pooled mean vector from tokens (no extra model pass)
                    pooled_embedding = out['embeddings'].mean(dim=0)
                results.append({
                    'embeddings': out['embeddings'],  # Tensor [num_tokens, dim]
                    'token_indices': out['token_indices'],
                    'metadata': metadata,
                    'token_level': True,
                    'extraction_time': per_image_extract_time,
                    'pooled_embedding': pooled_embedding
                })
                # Record extraction time per image
                tracker.record_image_processing(True, extraction_time=per_image_extract_time)

            # Save example structure once
            if successful_metadata and not tracker.has_example():
                # Derive structure from first output
                first_out = batch_outputs[0]
                num_tokens = int(first_out['embeddings'].shape[0])
                embedding_dim = int(first_out['embeddings'].shape[1]) if first_out['embeddings'].ndim == 2 else 0
                # Reconstruct tiles/positions from indices if available
                token_indices = first_out.get('token_indices', [])
                num_tiles = max([idx[0] for idx in token_indices]) + 1 if token_indices else 0
                sequence_length = max([idx[1] for idx in token_indices]) + 1 if token_indices else 0
                tracker.set_example({
                    'token_level': True,
                    'num_tokens': num_tokens,
                    'embedding_dim': embedding_dim,
                    'num_tiles': num_tiles,
                    'sequence_length': sequence_length
                })
        else:
            # Try batch first; fallback to per-image extraction with skipping on failure
            try:
                embeddings = extractor.extract_batch_embeddings(successful_tensors)
                extract_time = time.time() - extract_start
                logger.info(f"GPU extraction completed in {extract_time:.2f}s ({len(successful_tensors)/extract_time:.1f} images/s)")
                per_image_extract_time = extract_time / len(successful_tensors)
                for embedding, metadata in zip(embeddings, successful_metadata):
                    tracker.record_embeddings_per_image(1, int(embedding.shape[0]) if hasattr(embedding, 'shape') else 0)
                    results.append({
                        'embedding': embedding,
                        'metadata': metadata,
                        'token_level': False,
                        'extraction_time': per_image_extract_time
                    })
                    tracker.record_image_processing(True, extraction_time=per_image_extract_time)
            except Exception as e:
                logger.info(f"Batch extraction error: {e}. Falling back to per-image extraction and skipping failures.")
                for tensor, metadata in zip(successful_tensors, successful_metadata):
                    try:
                        start_single = time.time()
                        embedding = extractor.extract_single_embedding(tensor)
                        single_time = time.time() - start_single
                        tracker.record_embeddings_per_image(1, int(embedding.shape[0]) if hasattr(embedding, 'shape') else 0)
                        results.append({
                            'embedding': embedding,
                            'metadata': metadata,
                            'token_level': False,
                            'extraction_time': single_time
                        })
                        tracker.record_image_processing(True, extraction_time=single_time)
                    except Exception as ie:
                        logger.info(f"Extraction failed for {metadata['image_path']}: {ie}")
                        tracker.record_image_processing(False, extraction_time=0)
                        continue

            if successful_metadata and not tracker.has_example():
                tracker.set_example({
                    'token_level': False,
                    'num_tokens': 1,
                    'embedding_dim': int(embeddings[0].shape[0]) if hasattr(embeddings[0], 'shape') else 0,
                })
    
    # Record batch metrics
    batch_time = time.time() - batch_start_time
    tracker.record_batch_processing(len(image_paths), batch_time)
    
    # Sample memory usage
    tracker.sample_memory_usage()
    
    logger.info(f"Batch completed in {batch_time:.2f}s total ({len(image_paths)/batch_time:.1f} images/s)")
    
    return results

def get_image_statistics(image_paths: List[str]) -> dict:
    """
    Get statistics about a collection of images
    
    Args:
        image_paths: List of image paths
        
    Returns:
        dict: Statistics about the images
    """
    if not image_paths:
        return {"error": "No image paths provided"}
    
    total_size = 0
    sizes = []
    modes = []
    extensions = []
    valid_count = 0
    
    for path in image_paths:
        if os.path.exists(path):
            try:
                with Image.open(path) as img:
                    sizes.append(img.size)
                    modes.append(img.mode)
                    valid_count += 1
                    
                total_size += os.path.getsize(path)
                ext = os.path.splitext(path)[1].lower()
                extensions.append(ext)
                
            except Exception:
                continue
    
    if valid_count == 0:
        return {"error": "No valid images found"}
    
    # Calculate statistics
    widths = [size[0] for size in sizes]
    heights = [size[1] for size in sizes]
    
    stats = {
        'total_images': len(image_paths),
        'valid_images': valid_count,
        'invalid_images': len(image_paths) - valid_count,
        'total_size_mb': total_size / (1024 * 1024),
        'average_size_mb': (total_size / len(image_paths)) / (1024 * 1024),
        
        'image_dimensions': {
            'width_range': (min(widths), max(widths)),
            'height_range': (min(heights), max(heights)),
            'average_width': sum(widths) / len(widths),
            'average_height': sum(heights) / len(heights),
        },
        
        'image_modes': {mode: modes.count(mode) for mode in set(modes)},
        'file_extensions': {ext: extensions.count(ext) for ext in set(extensions)},
    }
    
    return stats