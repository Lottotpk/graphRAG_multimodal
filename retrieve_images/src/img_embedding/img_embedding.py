#!/usr/bin/env python3
"""
Build embedding database directly from image dataset.

Example:
  python -m img_embedding.img_embedding \
    --img_dir /research/d7/fyp24/tpipatpajong2/graphRAG_multimodal/dataset/stanford40 \
    --strategy last_layer_mean_pooling \
    --index_type exact \
    --verbose
"""

import os
import sys
import time
import argparse
from datetime import datetime
from typing import List

import torch
from transformers import AutoModel, Qwen3VLForConditionalGeneration

from img_embedding.img_embedder import ImgEmbedder
from vector_db.faiss_storage import FAISSEmbeddingDatabase
from metrics.performance_tracker import PerformanceTracker
from config import (
    MODEL_PATH, DEVICE_MAP, TORCH_DTYPE, SUPPORTED_IMAGE_EXTENSIONS, MAX_TILES, INPUT_SIZE,
    ensure_database_dir, get_database_path, get_text_database_folder_name
)
import logging
from utils.img_processing import find_image_files, load_images_parallel
from utils.logging_config import setup_logger

setup_logger()
logger = logging.getLogger(__name__)


STRATEGY_CHOICES = [
    'last_layer_token_level',
    'last_layer_cls_token',
    'last_layer_mean_pooling',
    '-2_layer_token_level',
    '-2_layer_cls_token',
    '-2_layer_mean_pooling',
]


# List of sources; each entry can be a directory or a single image file path
IMAGE_SOURCE_PATHS: List[str] = [
    '/research/d7/fyp24/tpipatpajong2/graphRAG_multimodal/dataset/test_imgs',
]


def _collect_images_from_sources(sources: List[str]) -> List[str]:
    images: List[str] = []
    for src in sources:
        if not src:
            continue
        if os.path.isdir(src):
            found = find_image_files(src, SUPPORTED_IMAGE_EXTENSIONS)
            images.extend(found)
        elif os.path.isfile(src):
            ext = os.path.splitext(src)[1].lower()
            if ext in SUPPORTED_IMAGE_EXTENSIONS:
                images.append(src)
    # Deduplicate and sort for stable order
    images = sorted(list(set(images)))
    return images


def _load_model():
    torch_dtype = torch.bfloat16 if TORCH_DTYPE == 'bfloat16' else torch.float16
    if MODEL_PATH.split("/")[0] == "Qwen":
        logger.info("Qwen detected, Loading Qwen3VLForConditionalGeneration...")
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            MODEL_PATH,
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
        )
    else:
        model = AutoModel.from_pretrained(
            MODEL_PATH,
            dtype=torch_dtype,
            load_in_8bit=False,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            trust_remote_code=True,
            device_map=DEVICE_MAP
        ).eval()
    return model


def main(args):
    images = _collect_images_from_sources(args.img_dir)
    if not images:
        logger.info('No images found from --img_dir nor IMAGE_SOURCE_PATHS; nothing to do.')
        return None

    if args.verbose:
        logger.info(f'Found {len(images)} images. Loading model...')

    model = _load_model()
    embedder = ImgEmbedder(model, device='cuda')

    if args.verbose:
        logger.info('Loading images in parallel on CPU...')
    loaded = load_images_parallel(images, max_tiles=MAX_TILES, input_size=INPUT_SIZE)

    # Separate successes and failures
    successes = []  # list of (pixel_values, path, num_tiles)
    def _to_cuda_bf16(t: torch.Tensor) -> torch.Tensor:
        if t.dtype != torch.bfloat16:
            t = t.to(torch.bfloat16)
        if t.device.type != 'cuda':
            t = t.cuda()
        return t
    for (pixel_values, meta) in loaded:
        if meta['load_success']:
            successes.append((pixel_values, meta['image_path'], meta['num_tiles']))
        else:
            logger.info(f"Fail to load image from {meta['image_path']} because of {meta['error']}")

    if not successes:
        logger.info('No valid images to process after loading.')
        sys.exit(0)

    # Determine embedding dimension using a small probe
    dim = embedder.get_embedding_dimension(_to_cuda_bf16(successes[0][0]))

    # Prepare DB
    ensure_database_dir()
    db_folder_name = get_text_database_folder_name(args.strategy, args.db_name, args.mode)
    db_path = get_database_path(db_folder_name)
    database = FAISSEmbeddingDatabase(
        database_folder=db_path,
        embedding_dimension=dim,
        index_type=args.index_type,
        create_new=(args.mode == 'new')
    )

    logger.info(f'Writing to text DB: {db_path}')

    # Metrics tracking (mirror image embedder style)
    experiment_name = f"{args.strategy}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    tracker = PerformanceTracker(experiment_name)
    tracker.start_experiment()

    start = time.time()
    
    # Embed in mini-batches to fit memory
    total = len(successes)
    for idx in range(0, total):
        if idx == total // 10:
            logger.info(f"Index: {idx}/{total}")

        batch_start = time.time()
        embeddings, more_meta = embedder.embed(_to_cuda_bf16(successes[idx][0]), args.strategy)
        extract_time = time.time() - batch_start
        per_item_extract = extract_time #/ max(1, len(batch))

        # For token-level, we may need to flatten
        flat_embeddings = []
        flat_meta = []
        if args.strategy.endswith('token_level'):
            for patch_idx in range(len(embeddings)):
                t_emb = embeddings[patch_idx] # t_emb is (T, D)
                for token_idx in range(t_emb.shape[0]):
                    flat_embeddings.append(t_emb[token_idx])
                    m = {"image_path": successes[idx][1]}
                    m.update({'patch_index': patch_idx, 'token_index': token_idx, 'token_level': True})
                    flat_meta.append(m)
            tracker.record_embeddings_per_image(int(len(embeddings) * embeddings[0].shape[0]), int(embeddings[0].shape[1]))
        else:
            for patch_idx in range(len(embeddings)):
                flat_embeddings.append(embeddings[patch_idx])
                m = {"image_path": successes[idx][1]}
                m.update({'patch_index': patch_idx, 'token_level': False, 'pooled': more_meta['pooled']})
                flat_meta.append(m)
            if embeddings:
                tracker.record_embeddings_per_image(len(embeddings), int(embeddings[0].shape[0]))

        store_start = time.time()
        database.add_embeddings(
            embeddings=flat_embeddings,
            image_paths=[m.get('source_image_path', '') for m in flat_meta],
            additional_metadata=flat_meta
        )
        store_time = time.time() - store_start
        # Record per-item times (loading time not applicable here; treat as 0)
        tracker.record_image_processing(True, loading_time=0, extraction_time=per_item_extract, storage_time=store_time / max(1, len(flat_embeddings)))

    database.save_database()
    # After save, record storage sizes
    db_info = database.get_database_info()
    tracker.record_storage_metrics(
        embedding_size_bytes=dim * 4,
        database_size_mb=db_info['file_sizes']['total_size_mb'],
        metadata_size_mb=db_info['file_sizes']['metadata_size_mb']
    )
    tracker.end_experiment()
    # Save metrics.txt next to index
    metrics_path = os.path.join(db_path, 'metrics.txt')
    tracker.save_metrics(metrics_path)
    elapsed = time.time() - start
    logger.info(f'Text DB built in {elapsed:.2f}s')
    logger.info(f'Results saved to: {db_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build image embedding database from dataset')
    parser.add_argument("--img_dir", nargs='*', default=IMAGE_SOURCE_PATHS, help='Path to image source')
    parser.add_argument('--strategy', required=True, choices=STRATEGY_CHOICES, help='Text embedding strategy')
    parser.add_argument('--db_name', help='Optional custom DB name')
    parser.add_argument('--mode', choices=['new', 'append'], default='new', help='Create new or append')
    parser.add_argument('--index_type', choices=['exact', 'approximate'], default='exact')
    parser.add_argument('--verbose', action='store_true')

    args = parser.parse_args()
    main(args)