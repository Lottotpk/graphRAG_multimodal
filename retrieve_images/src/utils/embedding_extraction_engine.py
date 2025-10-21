"""
Embedding Extraction Engine
Shared batching, device/dtype handling, and metrics for vision embedding extractors.
"""

from abc import ABC, abstractmethod
import torch
import time
from typing import List, Dict, Any
import psutil
import os
import logging
from utils.logging_config import setup_logger

setup_logger()
logger = logging.getLogger(__name__)


class EmbeddingExtractionEngine(ABC):
    """
    Core engine for embedding extractors.
    Provides GPU batching, metrics, and optional token-level APIs.
    """

    def __init__(self, model, tokenizer, device='cuda'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()

        self.metrics = {
            'total_images_processed': 0,
            'total_extraction_time': 0.0,
            'extraction_times': [],
            'memory_usage_samples': [],
            'batch_times': []
        }

    def outputs_token_level(self) -> bool:
        return False

    @abstractmethod
    def extract_single_embedding(self, pixel_values: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def get_embedding_dimension(self) -> int:
        pass

    @abstractmethod
    def get_strategy_name(self) -> str:
        pass

    def extract_batch_embeddings(self, pixel_values_batch: List[torch.Tensor]) -> List[torch.Tensor]:
        batch_start_time = time.time()

        if not pixel_values_batch:
            return []

        with torch.no_grad():
            gpu_tensors = []
            for pixel_values in pixel_values_batch:
                if pixel_values.device != torch.device(self.device):
                    pixel_values = pixel_values.to(self.device, non_blocking=True)
                if pixel_values.dtype != torch.bfloat16:
                    pixel_values = pixel_values.to(torch.bfloat16)
                gpu_tensors.append(pixel_values)

            try:
                embeddings = self._extract_batch_parallel(gpu_tensors)
                avg_time_per_image = (time.time() - batch_start_time) / len(gpu_tensors)
                for _ in range(len(gpu_tensors)):
                    self.metrics['extraction_times'].append(avg_time_per_image)
                    self.metrics['total_extraction_time'] += avg_time_per_image
                    self.metrics['total_images_processed'] += 1
            except (RuntimeError, NotImplementedError, torch.cuda.OutOfMemoryError) as e:
                logging.info(f"Batch parallel processing failed ({e}), falling back to sequential...")
                embeddings = []
                for pixel_values in gpu_tensors:
                    start_time = time.time()
                    embedding = self.extract_single_embedding(pixel_values)
                    embedding = embedding.cpu()
                    embeddings.append(embedding)
                    extraction_time = time.time() - start_time
                    self.metrics['extraction_times'].append(extraction_time)
                    self.metrics['total_extraction_time'] += extraction_time
                    self.metrics['total_images_processed'] += 1

            if len(self.metrics['memory_usage_samples']) < 100 or \
               self.metrics['total_images_processed'] % 10 == 0:
                memory_mb = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
                self.metrics['memory_usage_samples'].append(memory_mb)

        batch_time = time.time() - batch_start_time
        self.metrics['batch_times'].append(batch_time)
        return embeddings

    def _extract_batch_parallel(self, gpu_tensors: List[torch.Tensor]) -> List[torch.Tensor]:
        embeddings = []
        for tensor in gpu_tensors:
            embedding = self.extract_single_embedding(tensor).cpu()
            embeddings.append(embedding)
        return embeddings

    def get_metrics_summary(self) -> Dict[str, Any]:
        if self.metrics['total_images_processed'] == 0:
            return {"error": "No images processed yet"}

        extraction_times = self.metrics['extraction_times']
        memory_samples = self.metrics['memory_usage_samples']
        batch_times = self.metrics['batch_times']

        summary = {
            'strategy_name': self.get_strategy_name(),
            'embedding_dimension': self.get_embedding_dimension(),
            'total_images_processed': self.metrics['total_images_processed'],
            'total_extraction_time': self.metrics['total_extraction_time'],
            'avg_extraction_time_per_image': sum(extraction_times) / len(extraction_times),
            'min_extraction_time': min(extraction_times),
            'max_extraction_time': max(extraction_times),
            'images_per_second': self.metrics['total_images_processed'] / self.metrics['total_extraction_time'],
            'avg_memory_usage_mb': sum(memory_samples) / len(memory_samples) if memory_samples else 0,
            'peak_memory_usage_mb': max(memory_samples) if memory_samples else 0,
            'avg_batch_time': sum(batch_times) / len(batch_times) if batch_times else 0,
            'total_batches_processed': len(batch_times),
            'embedding_size_bytes': self.get_embedding_dimension() * 4,
            'total_embeddings_size_mb': (self.metrics['total_images_processed'] * self.get_embedding_dimension() * 4) / (1024 * 1024)
        }
        return summary

    def reset_metrics(self):
        self.metrics = {
            'total_images_processed': 0,
            'total_extraction_time': 0.0,
            'extraction_times': [],
            'memory_usage_samples': [],
            'batch_times': []
        }

    # Optional token-level APIs
    def extract_single_token_embeddings(self, pixel_values: torch.Tensor):
        raise NotImplementedError

    def extract_batch_token_embeddings(self, pixel_values_batch: List[torch.Tensor]):
        raise NotImplementedError