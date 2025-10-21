"""
Performance metrics tracking for embedding extraction
Collects time, memory, and storage metrics
"""

import time
import psutil
import os
import json
from datetime import datetime
from typing import Dict, Any, List
import torch
import logging
from utils.logging_config import setup_logger

setup_logger()
logger = logging.getLogger(__name__)

class PerformanceTracker:
    """
    Track performance metrics during embedding extraction
    """
    
    def __init__(self, experiment_name: str = "embedding_extraction"):
        """
        Initialize performance tracker
        
        Args:
            experiment_name: Name of the experiment for reporting
        """
        self.experiment_name = experiment_name
        self.start_time = None
        self.end_time = None
        
        # Metrics storage
        self.metrics = {
            'experiment_info': {
                'name': experiment_name,
                'start_time': None,
                'end_time': None,
                'duration_seconds': 0
            },
            'processing_metrics': {
                'total_images': 0,
                'successful_images': 0,
                'failed_images': 0,
                'total_batches': 0,
                'images_per_second_overall': 0,
            },
            'time_metrics': {
                'image_loading_times': [],
                'embedding_extraction_times': [],
                'database_storage_times': [],
                'batch_processing_times': [],
                # Derived convenience metrics
                'per_image_totals': []  # load + extract + store per image
            },
            'memory_metrics': {
                'memory_samples': [],
                'peak_memory_mb': 0,
                'average_memory_mb': 0,
            },
            'storage_metrics': {
                'embedding_sizes_bytes': [],
                'database_size_mb': 0,
                'metadata_size_mb': 0,
            },
            'embedding_metrics': {
                'num_embeddings_per_image': [],
                'embedding_dim_per_image': []
            },
            'example_info': None,
            'gpu_metrics': {
                'gpu_available': torch.cuda.is_available(),
                'gpu_name': None,
                'gpu_memory_samples': [],
                'peak_gpu_memory_mb': 0,
            }
        }
        
        # Initialize GPU info
        if torch.cuda.is_available():
            self.metrics['gpu_metrics']['gpu_name'] = torch.cuda.get_device_name(0)
    
    def start_experiment(self):
        """Start the experiment timer"""
        self.start_time = time.time()
        self.metrics['experiment_info']['start_time'] = datetime.now().isoformat()
        logger.info(f"Started experiment: {self.experiment_name}")
    
    def end_experiment(self, success: bool = True):
        """End the experiment and calculate final metrics
        
        Args:
            success: Whether the experiment completed successfully
        """
        if self.start_time is None:
            raise ValueError("Experiment not started. Call start_experiment() first.")
            
        self.end_time = time.time()
        duration = self.end_time - self.start_time
        
        self.metrics['experiment_info']['end_time'] = datetime.now().isoformat()
        self.metrics['experiment_info']['duration_seconds'] = duration
        self.metrics['experiment_info']['success'] = success
        
        # Calculate overall throughput
        if duration > 0 and self.metrics['processing_metrics']['successful_images'] > 0:
            self.metrics['processing_metrics']['images_per_second_overall'] = \
                self.metrics['processing_metrics']['successful_images'] / duration
        
        # Calculate memory statistics
        memory_samples = self.metrics['memory_metrics']['memory_samples']
        if memory_samples:
            self.metrics['memory_metrics']['peak_memory_mb'] = max(memory_samples)
            self.metrics['memory_metrics']['average_memory_mb'] = sum(memory_samples) / len(memory_samples)
        
        # Calculate GPU memory statistics
        gpu_samples = self.metrics['gpu_metrics']['gpu_memory_samples']
        if gpu_samples:
            self.metrics['gpu_metrics']['peak_gpu_memory_mb'] = max(gpu_samples)
        
        # print appropriate completion message
        status = "successfully" if success else "with errors"
        logger.info(f"Experiment {status}: {self.experiment_name} in {duration:.2f} seconds")
        
    def record_image_processing(self, success: bool, loading_time: float = 0, 
                              extraction_time: float = 0, storage_time: float = 0):
        """
        Record metrics for processing a single image
        
        Args:
            success: Whether processing was successful
            loading_time: Time taken to load image
            extraction_time: Time taken to extract embedding
            storage_time: Time taken to store in database
        """
        # Only increment image counters on initial loading phase to avoid double counting
        if loading_time > 0:
            self.metrics['processing_metrics']['total_images'] += 1
            if success:
                self.metrics['processing_metrics']['successful_images'] += 1
            else:
                self.metrics['processing_metrics']['failed_images'] += 1

        # Record times (can be called multiple times per image across phases)
        if success:
            if loading_time > 0:
                self.metrics['time_metrics']['image_loading_times'].append(loading_time)
            if extraction_time > 0:
                self.metrics['time_metrics']['embedding_extraction_times'].append(extraction_time)
            if storage_time > 0:
                self.metrics['time_metrics']['database_storage_times'].append(storage_time)
    
    def record_batch_processing(self, batch_size: int, batch_time: float):
        """
        Record metrics for batch processing
        
        Args:
            batch_size: Number of images in batch
            batch_time: Time taken to process batch
        """
        self.metrics['processing_metrics']['total_batches'] += 1
        self.metrics['time_metrics']['batch_processing_times'].append({
            'batch_size': batch_size,
            'batch_time': batch_time,
            'images_per_second': batch_size / batch_time if batch_time > 0 else 0
        })

    def record_embeddings_per_image(self, num_embeddings: int, embedding_dim: int):
        """Track number and dimension of embeddings per image"""
        self.metrics['embedding_metrics']['num_embeddings_per_image'].append(num_embeddings)
        self.metrics['embedding_metrics']['embedding_dim_per_image'].append(embedding_dim)

    def has_example(self) -> bool:
        return self.metrics.get('example_info') is not None

    def set_example(self, example_dict):
        self.metrics['example_info'] = example_dict
    
    def sample_memory_usage(self):
        """Sample current memory usage"""
        # CPU memory
        cpu_memory_mb = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
        self.metrics['memory_metrics']['memory_samples'].append(cpu_memory_mb)
        
        # GPU memory if available
        if torch.cuda.is_available():
            try:
                gpu_memory_mb = torch.cuda.memory_allocated(0) / 1024 / 1024
                self.metrics['gpu_metrics']['gpu_memory_samples'].append(gpu_memory_mb)
            except Exception:
                pass  # GPU memory sampling failed, continue
    
    def record_storage_metrics(self, embedding_size_bytes: int, database_size_mb: float, 
                             metadata_size_mb: float):
        """
        Record storage-related metrics
        
        Args:
            embedding_size_bytes: Size of individual embedding in bytes
            database_size_mb: Total database size in MB
            metadata_size_mb: Metadata size in MB
        """
        self.metrics['storage_metrics']['embedding_sizes_bytes'].append(embedding_size_bytes)
        self.metrics['storage_metrics']['database_size_mb'] = database_size_mb
        self.metrics['storage_metrics']['metadata_size_mb'] = metadata_size_mb
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary of all metrics
        
        Returns:
            Dict: Summary of metrics
        """
        # Calculate time statistics
        extraction_times = self.metrics['time_metrics']['embedding_extraction_times']
        loading_times = self.metrics['time_metrics']['image_loading_times']
        storage_times = self.metrics['time_metrics']['database_storage_times']
        batch_times = [b['batch_time'] for b in self.metrics['time_metrics']['batch_processing_times']]
        
        # Derived embedding metrics
        nums = self.metrics['embedding_metrics']['num_embeddings_per_image']
        dims = self.metrics['embedding_metrics']['embedding_dim_per_image']
        avg_num_embeddings_per_image = (sum(nums) / len(nums)) if nums else 0
        avg_embedding_dim = int((sum(dims) / len(dims))) if dims else 0
        avg_embedding_size_bytes = avg_embedding_dim * 4

        summary = {
            'experiment_overview': self.metrics['experiment_info'],
            'processing_summary': self.metrics['processing_metrics'],
            
            'timing_analysis': {
                'average_extraction_time_per_image': sum(extraction_times) / len(extraction_times) if extraction_times else 0,
                'average_loading_time_per_image': sum(loading_times) / len(loading_times) if loading_times else 0,
                'average_storage_time_per_image': sum(storage_times) / len(storage_times) if storage_times else 0,
                'average_total_time_per_image': (
                    (sum(loading_times) / len(loading_times) if loading_times else 0) +
                    (sum(extraction_times) / len(extraction_times) if extraction_times else 0) +
                    (sum(storage_times) / len(storage_times) if storage_times else 0)
                ),
                'average_batch_time': sum(batch_times) / len(batch_times) if batch_times else 0,
                
                'min_extraction_time': min(extraction_times) if extraction_times else 0,
                'max_extraction_time': max(extraction_times) if extraction_times else 0,
            },
            
            'resource_usage': {
                'memory': {
                    'peak_cpu_memory_mb': self.metrics['memory_metrics']['peak_memory_mb'],
                    'average_cpu_memory_mb': self.metrics['memory_metrics']['average_memory_mb'],
                },
                'gpu': self.metrics['gpu_metrics']
            },
            
            'storage_analysis': {
                'total_database_size_mb': self.metrics['storage_metrics']['database_size_mb'],
                'metadata_overhead_mb': self.metrics['storage_metrics']['metadata_size_mb'],
                'average_embedding_size_bytes': avg_embedding_size_bytes,
                'average_num_embeddings_per_image': avg_num_embeddings_per_image,
                'average_storage_size_per_image_bytes': avg_num_embeddings_per_image * avg_embedding_size_bytes
            },
            'example_info': self.metrics.get('example_info')
        }
        
        return summary
    
    def save_metrics(self, output_path: str):
        """
        Save metrics to a human-readable text file
        """
        s = self.get_summary()
        exp = s['experiment_overview']
        proc = s['processing_summary']
        timing = s['timing_analysis']
        storage = s['storage_analysis']
        ex = s.get('example_info')

        lines = []
        lines.append(f"Experiment: {exp['name']}")
        lines.append(f"Duration: {exp['duration_seconds']:.1f}s")
        lines.append(f"Success/Total: {proc['successful_images']}/{proc['total_images']}")
        lines.append(f"Avg time per image: total {timing['average_total_time_per_image']:.3f}s | load {timing['average_loading_time_per_image']:.3f}s | model {timing['average_extraction_time_per_image']:.3f}s | store {timing['average_storage_time_per_image']:.3f}s")
        lines.append(f"Avg storage per image: {storage['average_storage_size_per_image_bytes'] / (1024*1024):.3f} MB")
        lines.append(f"Avg embeddings per image: {storage['average_num_embeddings_per_image']:.1f} | Avg size per embedding: {storage['average_embedding_size_bytes']} bytes")
        if ex:
            if ex.get('token_level'):
                lines.append(f"Example (first image): tokens {ex.get('num_tokens')} | dim {ex.get('embedding_dim')} | tiles {ex.get('num_tiles')} | seq_len {ex.get('sequence_length')}")
            else:
                lines.append(f"Example (first image): single vector | dim {ex.get('embedding_dim')}")

        txt_path = output_path if output_path.endswith('.txt') else output_path.replace('.json', '.txt')
        with open(txt_path, 'w') as f:
            f.write('\n'.join(lines) + '\n')
        logger.info(f"Metrics saved to {txt_path}")
    
    def print_summary(self):
        """print a simplified, easy-to-read summary of metrics"""
        s = self.get_summary()
        exp = s['experiment_overview']
        proc = s['processing_summary']
        timing = s['timing_analysis']
        res = s['resource_usage']
        storage = s['storage_analysis']

        logger.info("\n--- Metrics ---")
        logger.info(f"Avg time per image: total {timing['average_total_time_per_image']:.3f}s | load {timing['average_loading_time_per_image']:.3f}s | model {timing['average_extraction_time_per_image']:.3f}s | store {timing['average_storage_time_per_image']:.3f}s")
        logger.info(f"Avg storage per image: {storage['average_storage_size_per_image_bytes'] / (1024*1024):.3f} MB")
        logger.info(f"Avg embeddings per image: {storage['average_num_embeddings_per_image']:.1f} | Avg size per embedding: {storage['average_embedding_size_bytes']} bytes")
        ex = s.get('example_info')
        if ex:
            if ex.get('token_level'):
                logger.info(f"Example (first image): tokens {ex.get('num_tokens')} | dim {ex.get('embedding_dim')} | tiles {ex.get('num_tiles')} | seq_len {ex.get('sequence_length')}")
            else:
                logger.info(f"Example (first image): single vector | dim {ex.get('embedding_dim')}")