#!/usr/bin/env python3
"""
Build text embedding database from a descriptions JSON produced by description_generator.

Example:
  python -m text_embedding.text_embedding \
    --descriptions ./image_description_databases/2025-09-23_15-38-56_desc.json \
    --strategy last_layer_mean_pooling \
    --index_type exact \
    --verbose
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime
from typing import List

import torch
from transformers import AutoModel, AutoTokenizer

from text_embedding.text_embedder import TextEmbedder
from vector_db.faiss_storage import FAISSEmbeddingDatabase
from metrics.performance_tracker import PerformanceTracker
from config import (
    MODEL_PATH, DEVICE_MAP, TORCH_DTYPE, IMAGE_DESCRIPTION_BASE_DIR, IMAGE_SUMMARY_BASE_DIR, IMAGE_ALL_BASE_DIR,
    ensure_text_database_dir, get_text_database_folder_name, get_text_database_path,
)
import logging
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


LATEST_IMAGE_DESC_PATH: str = os.path.join(IMAGE_DESCRIPTION_BASE_DIR, os.listdir(IMAGE_DESCRIPTION_BASE_DIR)[-1])
LATEST_IMAGE_SUMMARY_PATH: str = os.path.join(IMAGE_SUMMARY_BASE_DIR, os.listdir(IMAGE_SUMMARY_BASE_DIR)[-1])
LATEST_IMAGE_ALL_PATH: str = os.path.join(IMAGE_ALL_BASE_DIR, os.listdir(IMAGE_ALL_BASE_DIR)[-1])


def construct_prompt(desc: dict, smry: dict):
    abstract = ""
    for _, value in desc.items():
        for adj in value:
            if adj["confidence"] < 0.7:
                continue
            abstract += adj["keyword"] + ","
        abstract += "\n"
    prompt = smry["summary"] + "\n" + smry["entities"] + "\n" + smry["relations"] + "\n" + abstract
    # logger.info(prompt)
    return prompt


def load_model():
    torch_dtype = torch.bfloat16 if TORCH_DTYPE == 'bfloat16' else torch.float16
    model = AutoModel.from_pretrained(
        MODEL_PATH,
        dtype=torch_dtype,
        load_in_8bit=False,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True,
        device_map=DEVICE_MAP
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, use_fast=False)
    return model, tokenizer


def main(args):
    # if not os.path.exists(args.descriptions):
    #     logger.info(f"Error: descriptions file not found: {args.descriptions}")
    #     sys.exit(1)
    # if not os.path.exists(args.summary):
    #     logger.info(f"Error: summary file not found: {args.summary}")
    #     sys.exit(1)

    # with open(args.descriptions, 'r') as f, open(args.summary, 'r') as f2:
    #     payload = json.load(f)
    #     payload_smry = json.load(f2)
    payload = None
    if args.input == None or not os.path.exists(args.input):
        logger.info("Cannot find directory, or directory is None, use the latest file instead")
        with open(LATEST_IMAGE_ALL_PATH, 'r') as f:
            payload = json.load(f)
    else:
        with open(args.descriptions, 'r') as f:
            payload = json.load(f)

    # Collect valid texts and metadata
    # items = payload.get('records', [])
    # items_smry = payload_smry.get('records', [])
    # texts: List[str] = []
    # meta: List[dict] = []
    # for it, it2 in zip(items, items_smry):
    #     if it.get('description') and not it.get('error') and it2.get('description') and not it2.get('error'): 
    #         texts.append(construct_prompt(it['description'], it2['description']))
    #         meta.append({'source_image_path': it.get('image_path')})
    items = payload.get('records', [])
    texts: List[str] = []
    meta: List[dict] = []
    for it in items:
        if not it.get('error'):
            texts.append(construct_prompt(it["abstract"], {k: it[k] for k in ("summary", "entities", "relations")}))
            meta.append({'source_image_path': it.get('image_path')})

    if not texts:
        logger.info('No valid descriptions found in JSON')
        sys.exit(0)

    logger.info(f'Found {len(texts)} descriptions. Loading model...')
    model, tokenizer = load_model()
    embedder = TextEmbedder(model, tokenizer, device='cuda')

    # Determine embedding dimension using a small probe
    dim = embedder.get_embedding_dimension(texts)

    # Prepare DB
    ensure_text_database_dir()
    db_folder_name = get_text_database_folder_name(args.strategy, args.db_name, args.mode)
    db_path = get_text_database_path(db_folder_name)
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
    batch_size = 16
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        batch_meta = meta[i:i+batch_size]
        logger.info(f"Itr: {i}/{len(texts)}")
        # logger.info(batch_texts)

        batch_start = time.time()
        embeddings, more_meta = embedder.embed(batch_texts, args.strategy)
        extract_time = time.time() - batch_start
        per_item_extract = extract_time / max(1, len(batch_texts))

        # For token-level, we may need to flatten
        flat_embeddings = []
        flat_meta = []
        if args.strategy.endswith('token_level'):
            for base_meta, t_emb, t_meta in zip(batch_meta, embeddings, more_meta):
                # t_emb is (T, D)
                for token_idx in range(t_emb.shape[0]):
                    flat_embeddings.append(t_emb[token_idx])
                    m = dict(base_meta)
                    m.update({'token_index': token_idx, 'token_level': True})
                    flat_meta.append(m)
                tracker.record_embeddings_per_image(int(t_emb.shape[0]), int(t_emb.shape[1]))
        else:
            flat_embeddings = embeddings
            for base in batch_meta:
                m = dict(base)
                m.update({'token_level': False})
                flat_meta.append(m)
            if embeddings:
                tracker.record_embeddings_per_image(1, int(embeddings[0].shape[0]))

        store_start = time.time()
        database.add_embeddings(
            embeddings=flat_embeddings,
            image_paths=[m.get('source_image_path', '') for m in flat_meta],
            additional_metadata=flat_meta
        )
        store_time = time.time() - store_start
        # Record per-item times (loading time not applicable here; treat as 0)
        for _ in batch_texts:
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
    parser = argparse.ArgumentParser(description='Build text embedding database from descriptions JSON')
    # parser.add_argument('--descriptions', required=True, help='Path to descriptions JSON output')
    # parser.add_argument('--summary', required=True, help='Path to summary JSON output')
    parser.add_argument('--input', help='path to the generated image elements')
    parser.add_argument('--strategy', required=True, choices=STRATEGY_CHOICES, help='Text embedding strategy')
    parser.add_argument('--db_name', help='Optional custom DB name')
    parser.add_argument('--mode', choices=['new', 'append'], default='new', help='Create new or append')
    parser.add_argument('--index_type', choices=['exact', 'approximate'], default='exact')
    parser.add_argument('--verbose', action='store_true')

    args = parser.parse_args()
    main(args)