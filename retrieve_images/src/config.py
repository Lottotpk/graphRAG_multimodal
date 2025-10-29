"""
Configuration settings for embedding extraction system
"""

import os
from datetime import datetime

# Model Configuration
MODEL_PATH = 'OpenGVLab/InternVL3_5-8B'
DEVICE_MAP = "auto"
TORCH_DTYPE = "bfloat16"

# Processing Configuration (Optimized for RTX 4090)  
MAX_TILES = 12
INPUT_SIZE = 448
BATCH_SIZE = 4  # Increased for RTX 4090: 8 images per batch for better GPU utilization

# Database Configuration
IMAGE_DATABASE_BASE_DIR = "./img_embedding_databases/"
# Align with workspace folder naming for text databases
TEXT_DATABASE_BASE_DIR = "./description_embedding_databases/"
# Where generated image descriptions will be saved
IMAGE_DESCRIPTION_BASE_DIR = "./generated_img_description/"
# Images summary, entities, relations
IMAGE_SUMMARY_BASE_DIR = "./generated_img_summary/"
AUTO_TIMESTAMP_FORMAT = "%Y-%m-%d_%H-%M-%S"
# Only image summary
IMAGE_ONLY_SUMMARY_DIR = "./generated_only_summary/"
# Formated prompt
FORMAT_PROMPT_DIR = "./generated_format_prompt/"
# Retrieval result
RETRIEVAL_RESULT = "./retrieval_result/"
# Store all generated content
IMAGE_ALL_BASE_DIR = "./generated_all/"

# Supported extraction strategies
EXTRACTION_STRATEGIES = {
    'vision_tokens': {
        'name': 'Vision Tokens (Encoder Last Hidden States)',
        'description': 'All tokens from vision encoder last_hidden_state (no pooling)',
        'expected_dim': 1024,
        'token_level': True,
        'alias_of': 'vision_encoder_token_level'
    },
    'vision_encoder_token_level': {
        'name': 'Vision Encoder Token-Level',
        'description': 'Token-level from model.vision_model last_hidden_state',
        'expected_dim': 1024,
        'token_level': True
    },
    'vision_encoder_mean_pooling': {
        'name': 'Vision Encoder Mean Pooling',
        'description': 'Mean pooling over tokens from vision encoder',
        'expected_dim': 1024,
        'token_level': False
    },
    'vision_encoder_combine': {
        'name': 'Vision Encoder Combine',
        'description': 'Token-level + pooled mean stored per image',
        'expected_dim': 1024,
        'token_level': True
    },
    'feature_tokens': {
        'name': 'Feature Head Tokens', 
        'description': 'All tokens from model.extract_feature output (no pooling)',
        'expected_dim': 4096,
        'token_level': True,
        'alias_of': 'feature_head_token_level'
    },
    'feature_head_token_level': {
        'name': 'Feature Head Token-Level',
        'description': 'Token-level from projection head (extract_feature)',
        'expected_dim': 4096,
        'token_level': True
    },
    'feature_head_combine': {
        'name': 'Feature Head Combine',
        'description': 'Token-level + pooled mean stored per image',
        'expected_dim': 4096,
        'token_level': True
    },
    'extract_feature': {
        'name': 'Extract Feature Strategy', 
        'description': 'Use model.extract_feature method with pooling',
        'expected_dim': 4096,
        'token_level': False,
        'alias_of': 'feature_head_mean_pooling'
    },
    'feature_head_mean_pooling': {
        'name': 'Feature Head Mean Pooling',
        'description': 'Mean pooling across tiles and positions from feature head',
        'expected_dim': 4096,
        'token_level': False
    }
}

# FAISS Configuration
FAISS_INDEX_TYPES = {
    'exact': 'IndexFlatIP',      # Exact cosine similarity
    'approximate': 'IndexIVFFlat' # Approximate search
}

# Metrics Configuration
METRICS_TO_COLLECT = [
    'extraction_time_per_image',
    'batch_processing_time', 
    'memory_usage',
    'embedding_size_bytes',
    'database_size_mb'
]

# File Extensions
SUPPORTED_IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']

def get_database_folder_name(strategy_name, custom_name=None, mode='new'):
    """
    Generate database folder name based on strategy and mode
    
    Args:
        strategy_name: 'cls_token' or 'extract_feature'
        custom_name: Custom folder name if provided
        mode: 'new' or 'append'
    """
    if custom_name and mode == 'append':
        return custom_name
    
    timestamp = datetime.now().strftime(AUTO_TIMESTAMP_FORMAT)
    
    if custom_name:
        return f"{timestamp}_{custom_name}_{strategy_name}"
    else:
        return f"{timestamp}_{strategy_name}_auto"

def get_database_path(folder_name):
    """Get full path to database folder"""
    return os.path.join(IMAGE_DATABASE_BASE_DIR, folder_name)

def ensure_database_dir():
    """Create database directory if it doesn't exist"""
    os.makedirs(IMAGE_DATABASE_BASE_DIR, exist_ok=True)

# ----------------------------
# Text database helpers
# ----------------------------

def get_text_database_folder_name(strategy_name, custom_name=None, mode='new'):
    """
    Generate text database folder name based on strategy and mode
    """
    if custom_name and mode == 'append':
        return custom_name
    timestamp = datetime.now().strftime(AUTO_TIMESTAMP_FORMAT)
    if custom_name:
        return f"{timestamp}_{custom_name}_{strategy_name}"
    else:
        return f"{timestamp}_{strategy_name}_auto"

def get_text_database_path(folder_name):
    """Get full path to TEXT database folder"""
    return os.path.join(TEXT_DATABASE_BASE_DIR, folder_name)

def ensure_text_database_dir():
    """Create TEXT database directory if it doesn't exist"""
    os.makedirs(TEXT_DATABASE_BASE_DIR, exist_ok=True)

def ensure_prompt_description_dir():
    """Create formated prompt directory if it doesn't exist"""
    os.makedirs(FORMAT_PROMPT_DIR, exist_ok=True)

def ensure_retrieval_result_dir():
    """Create retrieval result directory if it doesn't exist"""
    os.makedirs(RETRIEVAL_RESULT, exist_ok=True)

# ----------------------------
# Image description helpers
# ----------------------------

def ensure_image_description_dir(type: str = None):
    if type == "summary":
        os.makedirs(IMAGE_SUMMARY_BASE_DIR, exist_ok=True)
    elif type == "only_summary":
        os.makedirs(IMAGE_ONLY_SUMMARY_DIR, exist_ok=True)
    elif type == "all":
        os.makedirs(IMAGE_ALL_BASE_DIR, exist_ok=True)
    else:
        os.makedirs(IMAGE_DESCRIPTION_BASE_DIR, exist_ok=True)

def get_description_filename(prompt_slug: str = "desc"):
    """
    Build a timestamped filename for image descriptions JSON.
    """
    timestamp = datetime.now().strftime(AUTO_TIMESTAMP_FORMAT)
    safe_slug = prompt_slug.replace(" ", "_").replace("/", "-")[:50]
    return f"{timestamp}_{safe_slug}.json"

def get_description_path(filename: str, type: str = None):
    if type == "summary":
        return os.path.join(IMAGE_SUMMARY_BASE_DIR, filename)
    elif type == "only_summary":
        return os.path.join(IMAGE_ONLY_SUMMARY_DIR, filename)
    elif type == "prompt":
        return os.path.join(FORMAT_PROMPT_DIR, filename)
    elif type == "result":
        return os.path.join(RETRIEVAL_RESULT, filename)
    elif type == "all":
        return os.path.join(IMAGE_ALL_BASE_DIR, filename)
    else:
        return os.path.join(IMAGE_DESCRIPTION_BASE_DIR, filename)