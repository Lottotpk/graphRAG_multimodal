"""
FAISS Vector Database Storage
Handles embedding storage, indexing, and metadata management
"""

import faiss
import numpy as np
import json
import os
import pickle
import time
import torch
import logging
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
from utils.logging_config import setup_logger

setup_logger()
logger = logging.getLogger(__name__)

class FAISSEmbeddingDatabase:
    """
    FAISS-based vector database for image embeddings with metadata
    
    Supports both exact and approximate search with metadata persistence
    """
    
    def __init__(self, database_folder: str, embedding_dimension: int, 
                 index_type: str = 'exact', create_new: bool = False):
        """
        Initialize FAISS database
        
        Args:
            database_folder: Path to database folder
            embedding_dimension: Dimension of embeddings
            index_type: 'exact' (IndexFlatIP) or 'approximate' (IndexIVFFlat)
            create_new: Whether to create new database or load existing
        """
        self.database_folder = database_folder
        self.embedding_dimension = embedding_dimension
        self.index_type = index_type
        
        # File paths
        self.index_path = os.path.join(database_folder, 'embeddings.index')
        self.metadata_path = os.path.join(database_folder, 'metadata.json')
        self.config_path = os.path.join(database_folder, 'config.json')
        self.stats_path = os.path.join(database_folder, 'stats.json')
        
        # Initialize database
        os.makedirs(database_folder, exist_ok=True)
        
        if create_new or not self._database_exists():
            self._create_new_database()
        else:
            self._load_existing_database()
            
        # Metrics
        self.storage_metrics = {
            'total_embeddings': 0,
            'total_storage_time': 0.0,
            'index_build_time': 0.0,
            'last_updated': None
        }
        
    def _database_exists(self) -> bool:
        """Check if database files exist"""
        return (os.path.exists(self.index_path) and 
                os.path.exists(self.metadata_path) and
                os.path.exists(self.config_path))
    
    def _create_new_database(self):
        """Create new FAISS database"""
        logger.info(f"Creating new FAISS database in {self.database_folder}")
        
        # Create FAISS index
        if self.index_type == 'exact':
            self.index = faiss.IndexFlatIP(self.embedding_dimension)
        elif self.index_type == 'approximate':
            # IVF index for approximate search
            quantizer = faiss.IndexFlatIP(self.embedding_dimension)
            self.index = faiss.IndexIVFFlat(quantizer, self.embedding_dimension, 100)  # 100 clusters
        else:
            raise ValueError(f"Unknown index type: {self.index_type}")
        
        # Initialize metadata
        self.metadata = {}
        self.embedding_counter = 0
        
        # Save configuration
        config = {
            'embedding_dimension': self.embedding_dimension,
            'index_type': self.index_type,
            'created_at': datetime.now().isoformat(),
            'version': '1.0'
        }
        
        with open(self.config_path, 'w') as f:
            json.dump(config, f, indent=2)
            
        # Initialize empty metadata
        with open(self.metadata_path, 'w') as f:
            json.dump({}, f, indent=2)
    
    def _load_existing_database(self):
        """Load existing FAISS database"""
        logger.info(f"Loading existing FAISS database from {self.database_folder}")
        
        # Load configuration
        with open(self.config_path, 'r') as f:
            config = json.load(f)
            
        # Verify compatibility
        if config['embedding_dimension'] != self.embedding_dimension:
            raise ValueError(f"Dimension mismatch: expected {self.embedding_dimension}, "
                           f"got {config['embedding_dimension']}")
        
        if config['index_type'] != self.index_type:
            logger.info(f"Warning: Index type mismatch. Expected {self.index_type}, "
                  f"got {config['index_type']}. Using existing type.")
            self.index_type = config['index_type']
        
        # Load FAISS index
        self.index = faiss.read_index(self.index_path)
        
        # Load metadata
        with open(self.metadata_path, 'r') as f:
            self.metadata = json.load(f)
            
        self.embedding_counter = len(self.metadata)
        logger.info(f"Loaded database with {self.embedding_counter} embeddings")
    
    def add_embeddings(self, embeddings: List[np.ndarray], image_paths: List[str], 
                      additional_metadata: Optional[List[Dict]] = None) -> List[str]:
        """
        Add embeddings to the database
        
        Args:
            embeddings: List of embedding vectors
            image_paths: List of corresponding image paths
            additional_metadata: Optional additional metadata for each embedding
            
        Returns:
            List[str]: List of embedding IDs
        """
        if len(embeddings) != len(image_paths):
            raise ValueError("Number of embeddings must match number of image paths")
        
        start_time = time.time()
        
        # Convert embeddings to numpy array (handle dtype and shape robustness)
        numpy_embeddings: List[np.ndarray] = []
        bad_indices: List[int] = []
        for idx, emb in enumerate(embeddings):
            try:
                # Torch tensor handling
                if 'torch' in str(type(emb)):
                    # Avoid importing torch here; rely on duck-typing
                    if hasattr(emb, 'dtype') and 'bfloat16' in str(emb.dtype):
                        emb = emb.float()
                    arr = emb.detach().cpu().numpy() if hasattr(emb, 'detach') else emb.cpu().numpy()
                else:
                    arr = np.array(emb)

                # Ensure float32
                arr = arr.astype('float32', copy=False)

                # Squeeze singleton dims and coerce to 1D vector
                arr = np.squeeze(arr)

                # If 2D like [1, D] or [D, 1], reshape to [D]
                if arr.ndim == 2:
                    if arr.shape[0] == 1 and arr.shape[1] == self.embedding_dimension:
                        arr = arr.reshape(self.embedding_dimension)
                    elif arr.shape[1] == 1 and arr.shape[0] == self.embedding_dimension:
                        arr = arr.reshape(self.embedding_dimension)
                    else:
                        # Flatten if total size matches expected dimension
                        if arr.size == self.embedding_dimension:
                            arr = arr.reshape(self.embedding_dimension)
                        else:
                            raise ValueError(f"Unexpected 2D shape {arr.shape}")

                # Final validation
                if arr.ndim != 1 or arr.shape[0] != self.embedding_dimension:
                    raise ValueError(f"Invalid embedding shape {arr.shape}, expected ({self.embedding_dimension},)")

                numpy_embeddings.append(arr)
            except Exception:
                bad_indices.append(idx)
                continue

        if bad_indices:
            raise ValueError(
                f"Found {len(bad_indices)} malformed embeddings (e.g., indices {bad_indices[:5]}...). "
                f"All embeddings must be 1D vectors of length {self.embedding_dimension}."
            )

        embeddings_array = np.vstack(numpy_embeddings)
        
        # Normalize for cosine similarity (IndexFlatIP expects normalized vectors)
        faiss.normalize_L2(embeddings_array)
        
        # Train index if it's IVF and not trained yet
        if self.index_type == 'approximate' and not self.index.is_trained:
            logger.info("Training IVF index...")
            train_start = time.time()
            self.index.train(embeddings_array)
            train_time = time.time() - train_start
            logger.info(f"Index training completed in {train_time:.2f} seconds")
            self.storage_metrics['index_build_time'] += train_time
        
        # Add embeddings to index
        start_idx = self.embedding_counter
        self.index.add(embeddings_array)
        
        # Generate embedding IDs and store metadata
        embedding_ids = []
        for i, (image_path, embedding) in enumerate(zip(image_paths, embeddings)):
            embedding_id = f"emb_{self.embedding_counter:06d}"
            embedding_ids.append(embedding_id)
            
            # Store metadata (use converted float32 embeddings for calculations)
            converted_embedding = embeddings_array[i]  # Use the float32 version
            metadata_entry = {
                'id': embedding_id,
                'image_path': image_path,
                'index_position': self.embedding_counter,
                'added_at': datetime.now().isoformat(),
                'embedding_norm': float(np.linalg.norm(converted_embedding)),
            }
            
            # Add additional metadata if provided
            if additional_metadata and i < len(additional_metadata):
                metadata_entry.update(additional_metadata[i])
            
            self.metadata[embedding_id] = metadata_entry
            self.embedding_counter += 1
        
        # Update metrics
        storage_time = time.time() - start_time
        self.storage_metrics['total_embeddings'] = self.embedding_counter
        self.storage_metrics['total_storage_time'] += storage_time
        self.storage_metrics['last_updated'] = datetime.now().isoformat()
        
        logger.info(f"Added {len(embeddings)} embeddings in {storage_time:.2f} seconds")
        
        return embedding_ids
    
    def save_database(self):
        """Save database to disk"""
        logger.info("Saving database to disk...")
        start_time = time.time()
        
        # Save FAISS index
        faiss.write_index(self.index, self.index_path)
        
        # Save metadata
        with open(self.metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        
        # No longer write stats.json (metrics are summarized in a human-readable txt)
            
        save_time = time.time() - start_time
        logger.info(f"Database saved in {save_time:.2f} seconds")
    
    def get_database_info(self) -> Dict[str, Any]:
        """Get information about the database"""
        index_size_mb = os.path.getsize(self.index_path) / (1024 * 1024) if os.path.exists(self.index_path) else 0
        metadata_size_mb = os.path.getsize(self.metadata_path) / (1024 * 1024) if os.path.exists(self.metadata_path) else 0
        
        return {
            'database_folder': self.database_folder,
            'embedding_dimension': self.embedding_dimension,
            'index_type': self.index_type,
            'total_embeddings': self.index.ntotal if hasattr(self.index, 'ntotal') else 0,
            'is_trained': self.index.is_trained if hasattr(self.index, 'is_trained') else True,
            'storage_metrics': self.storage_metrics,
            'file_sizes': {
                'index_size_mb': index_size_mb,
                'metadata_size_mb': metadata_size_mb,
                'total_size_mb': index_size_mb + metadata_size_mb
            }
        }
    
    def search_similar(self, query_embedding: np.ndarray, k: int = 5) -> Tuple[List[float], List[str]]:
        """
        Search for similar embeddings (for future retrieval functionality)
        
        Args:
            query_embedding: Query embedding vector
            k: Number of results to return
            
        Returns:
            Tuple of (similarity_scores, embedding_ids)
        """
        if self.index.ntotal == 0:
            return [], []
        
        # Normalize query for cosine similarity
        query_normalized = query_embedding.reshape(1, -1).astype('float32')
        faiss.normalize_L2(query_normalized)
        
        # Search
        if self.index_type == 'approximate' and hasattr(self.index, 'nprobe'):
            self.index.nprobe = min(10, self.index.nlist)  # Search 10 clusters
            
        scores, indices = self.index.search(query_normalized, k)
        
        # Convert indices to embedding IDs
        embedding_ids = []
        for idx in indices[0]:
            if idx != -1:  # Valid index
                # Find embedding ID by index position
                for emb_id, metadata in self.metadata.items():
                    if metadata['index_position'] == idx:
                        embedding_ids.append(emb_id)
                        break
        
        return scores[0].tolist(), embedding_ids
    
    def search_colbert(self, query_embedding: np.ndarray, k: int = 5) -> Tuple[List[float], List[str]]:
        # 2 dimensional vector on query_embedding only
        if self.index.ntotal == 0:
            return [], []
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1).astype('float32') # faiss.normalize_L2 does not receive (D,)
        faiss.normalize_L2(query_embedding)
        
        candidate_img = set()
        for i in range(len(query_embedding)):
            _, ids = self.search_similar(query_embedding[i], k=k)
            for emb_id in ids:
                candidate_img.add(self.metadata[emb_id]['image_path'])
        candidate_img = list(candidate_img)
        logger.info(f"ColBERT searching: Found {len(candidate_img)} candidate image(s) from {k*len(query_embedding)} searches.")

        # Combine the corresponding vector from the metadata
        all_candidates = []
        for target_img in candidate_img:
            flag = False # Start appending vector flag since the same img will be cluster in one range only.
            candidate_embedding = []
            for _, val in self.metadata.items():
                if val['image_path'] == target_img:
                    candidate_embedding.append(self.index.reconstruct_n(val['index_position'], 1)[0])
                    flag |= True
                elif val['image_path'] != target_img and flag:
                    break
            all_candidates.append(np.stack(candidate_embedding))

        # Compute MaxSim
        similarity_scores = []
        for embedding in all_candidates:
            scores = np.matmul(query_embedding, embedding.T)
            max_row = scores.max(axis=1)
            sim_result = max_row.sum()
            similarity_scores.append(sim_result)

        # Report top k candidates
        result = torch.topk(torch.Tensor(similarity_scores), k)
        return result.values, [candidate_img[i] for i in result.indices]

    def search_hierarchy(self, query_embedding: List[np.ndarray], weight: List[float], k: int = 5) -> Tuple[List[float], List[str]]:

        if self.index.ntotal == 0:
            return [], []
        for i in range(len(query_embedding)):
            if query_embedding[i].ndim == 1:
                query_embedding[i] = query_embedding[i].reshape(1, -1).astype('float32') # faiss.normalize_L2 does not receive (D,)
            faiss.normalize_L2(query_embedding[i])
        
        # 2 dimensional vector on query_embedding only
        ndim = query_embedding[0].shape[1]
        print(ndim)

        candidate_img = set()
        for i in range(len(query_embedding)):
            for j in range(len(query_embedding[i])):
                _, ids = self.search_similar(query_embedding[i][j], k=k)
                for emb_id in ids:
                    candidate_img.add(self.metadata[emb_id]['image_path'])
        candidate_img = list(candidate_img)
        logger.info(f"ColBERT searching: Found {len(candidate_img)} candidate image(s) from {(k)*sum([len(x) for x in query_embedding])} searches.")

        # Combine the corresponding vector from the metadata
        all_candidates = []
        for target_img in candidate_img:
            flag = False # Start appending vector flag since the same img will be cluster in one range only.
            components = ['summary', 'entities', 'relations', 'abstract']
            candidate_embedding = [np.empty((0, ndim)), np.empty((0, ndim)), np.empty((0, ndim)), np.empty((0, ndim))] # [summary, entities, relations, abstract]
            for _, val in self.metadata.items():
                if val['image_path'] == target_img:
                    idx = components.index(val['type'])
                    candidate_embedding[idx] = np.append(candidate_embedding[idx], self.index.reconstruct_n(val['index_position'], 1), axis=0)
                    flag |= True
                elif val['image_path'] != target_img and flag:
                    break
            all_candidates.append(candidate_embedding)

        # Compute MaxSim
        similarity_scores = []
        for candidate in all_candidates:
            total_score = []
            for i in range(len(candidate)):
                scores = np.matmul(query_embedding[i], candidate[i].T)
                max_row = scores.max(axis=1)
                total_score.append(max_row.sum() * weight[i])
            # May adjust scoring here. Note: [summary, entities, relations, abstract]
            similarity_scores.append(sum(total_score))

        # Report top k candidates
        result = torch.topk(torch.Tensor(similarity_scores), k)
        return result.values, [candidate_img[i] for i in result.indices]
    
    def get_embedding_by_id(self, embedding_id: str) -> Optional[Dict]:
        """Get embedding metadata by ID"""
        return self.metadata.get(embedding_id)
    
    def list_all_embeddings(self) -> List[str]:
        """Get all embedding IDs"""
        return list(self.metadata.keys())
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get detailed storage statistics"""
        if not self.metadata:
            return {"error": "No embeddings in database"}
            
        image_paths = [meta['image_path'] for meta in self.metadata.values()]
        embedding_norms = [meta.get('embedding_norm', 0) for meta in self.metadata.values()]
        
        return {
            'total_embeddings': len(self.metadata),
            'unique_images': len(set(image_paths)),
            'database_size_mb': self.get_database_info()['file_sizes']['total_size_mb'],
            'average_embedding_norm': sum(embedding_norms) / len(embedding_norms) if embedding_norms else 0,
            'storage_efficiency': {
                'bytes_per_embedding': self.embedding_dimension * 4,  # float32
                'metadata_overhead_per_embedding': 
                    self.get_database_info()['file_sizes']['metadata_size_mb'] * 1024 * 1024 / len(self.metadata)
            }
        }