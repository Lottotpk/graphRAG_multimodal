#!/usr/bin/env python3
"""
Image Description Generator (uses local InternVL3_5-8B chat API)

Edit the variables in this file to specify image sources and the prompt.
Outputs a timestamped JSON file under ./image_description_databases/ with:
{
  "created_at": iso,
  "prompt": str,
  "model_path": str,
  "records": [ {"image_path": str, "description": str|None, "error": str|None} ]
}

Addition: This script is for extracting the abstract representation of an image.
"""

import os
import sys
import json
import time
from datetime import datetime
from typing import List

import torch
from transformers import AutoModel, AutoTokenizer
import logging
from utils.logging_config import setup_logger
from utils.json_processing import combine_abstract
from prompt import ABSTRACT_PROMPT, SYSTEM_PROMPT
NUM_TOPIC = 2

setup_logger()
logger = logging.getLogger(__name__)

# Ensure package imports work when run from anywhere
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from utils.img_processing import (
    find_image_files, load_image_with_error_handling, load_images_parallel
)
from config import (
    MODEL_PATH, DEVICE_MAP, TORCH_DTYPE, MAX_TILES, INPUT_SIZE,
    BATCH_SIZE,
    IMAGE_DESCRIPTION_BASE_DIR, ensure_image_description_dir,
    get_description_filename, get_description_path,
    SUPPORTED_IMAGE_EXTENSIONS
)


# -----------------------------
# Edit these values
# -----------------------------

# List of sources; each entry can be a directory or a single image file path
IMAGE_SOURCE_PATHS: List[str] = [
    '/research/d7/fyp24/tpipatpajong2/graphRAG_multimodal/dataset/test_imgs',
]

# Prompt written at the top of the output JSON and used for generation
# PROMPT: str = 'Please describe the image in detail.'

# Generation parameters
MAX_NEW_TOKENS: int = 256
DO_SAMPLE: bool = True
TEMPERATURE: float = 0.6
TOP_P: float = 0.9
VERBOSE: bool = True

# Caption batch size (GPU batch for chat). Defaults to config.BATCH_SIZE
# Increase to better utilize GPU memory; decrease if you hit OOM.
CAPTION_BATCH_SIZE: int = 3


def _load_model():
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


def _describe_single(model, tokenizer, pixel_values: torch.Tensor, prompt: str, generation_config: dict) -> str:
    if pixel_values.dtype != torch.bfloat16:
        pixel_values = pixel_values.to(torch.bfloat16)
    if pixel_values.device.type != 'cuda':
        pixel_values = pixel_values.cuda()
    question = f'<image>\n{prompt}'
    response = model.chat(tokenizer, pixel_values, question, generation_config)
    return response


def generate_descriptions():
    ensure_image_description_dir()

    images = _collect_images_from_sources(IMAGE_SOURCE_PATHS)
    if not images:
        logger.info('No images found from IMAGE_SOURCE_PATHS; nothing to do.')
        return None

    if VERBOSE:
        logger.info(f'Found {len(images)} images. Loading model...')

    model, tokenizer = _load_model()

    gen_cfg = dict(
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=DO_SAMPLE,
        temperature=TEMPERATURE,
        top_p=TOP_P,
    )

    records = []
    start = time.time()

    # 1) Parallel CPU image loading
    if VERBOSE:
        logger.info('Loading images in parallel on CPU...')
    loaded = load_images_parallel(images, max_tiles=MAX_TILES, input_size=INPUT_SIZE)

    # Separate successes and failures
    successes = []  # list of (pixel_values, path, num_tiles)
    for (pixel_values, meta) in loaded:
        if meta['load_success']:
            successes.append((pixel_values, meta['image_path'], meta['num_tiles']))
        else:
            records.append({
                'image_path': meta['image_path'],
                'description': None,
                'error': meta['error']
            })

    if not successes:
        logger.info('No valid images to process after loading.')
        # Still write an output file with failures only
        filename = get_description_filename(prompt_slug='desc')
        out_path = get_description_path(filename)
        payload = {
            'created_at': datetime.now().isoformat(),
            'system_prompt': SYSTEM_PROMPT,
            'prompt': ABSTRACT_PROMPT(range(8)),
            'model_path': MODEL_PATH,
            'records': records
        }
        with open(out_path, 'w') as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        logger.info(f'Descriptions saved to: {out_path}')
        return out_path

    # 2) GPU batched captioning using batch_chat when available
    total = len(successes)
    if VERBOSE:
        logger.info(f'Generating captions on GPU in batches of {CAPTION_BATCH_SIZE}...')

    def _to_cuda_bf16(t: torch.Tensor) -> torch.Tensor:
        if t.dtype != torch.bfloat16:
            t = t.to(torch.bfloat16)
        if t.device.type != 'cuda':
            t = t.cuda()
        return t

    for start_idx in range(0, total, CAPTION_BATCH_SIZE):
        batch = successes[start_idx:start_idx + CAPTION_BATCH_SIZE]
        batch_paths = [p for _, p, _ in batch]
        batch_tensors = [_to_cuda_bf16(t) for t, _, _ in batch]
        num_patches_list = [int(t.shape[0]) for t in batch_tensors]

        for j in range(0, 8, NUM_TOPIC):
            PROMPT = ABSTRACT_PROMPT(range(j, min(j + NUM_TOPIC, 8)))
            questions = [f'<image>\n{PROMPT}' for _ in batch_tensors]
            model.system_message = SYSTEM_PROMPT

            try:
                # Concatenate along tiles dimension per talk_test.py example
                pixel_values_cat = torch.cat(batch_tensors, dim=0)
                if hasattr(model, 'batch_chat'):
                    responses = model.batch_chat(
                        tokenizer,
                        pixel_values_cat,
                        num_patches_list=num_patches_list,
                        questions=questions,
                        generation_config=gen_cfg
                    )
                    for pth, resp in zip(batch_paths, responses):
                        records.append({
                            'image_path': pth,
                            'description': json.loads(resp.strip("```json")),
                            'error': None
                        })
                else:
                    # Fallback: per-image chat
                    for t, pth in zip(batch_tensors, batch_paths):
                        try:
                            resp = model.chat(tokenizer, t, questions[0], gen_cfg)
                            records.append({
                                'image_path': pth,
                                'description': json.loads(resp.strip("```json")),
                                'error': None
                            })
                        except Exception as e:
                            records.append({
                                'image_path': pth,
                                'description': None,
                                'error': str(e)
                            })
            except Exception as be:
                # Batch failed (e.g., OOM). Fallback to per-image sequential within this batch
                if VERBOSE:
                    logger.info(f'Batch captioning failed, falling back per-image: {be}')
                for t, pth in zip(batch_tensors, batch_paths):
                    try:
                        resp = model.chat(tokenizer, t, f'<image>\n{PROMPT}', gen_cfg)
                        records.append({
                            'image_path': pth,
                            'description': json.loads(resp.strip("```json")),
                            'error': None
                        })
                    except Exception as e:
                        records.append({
                            'image_path': pth,
                            'description': None,
                            'error': str(e)
                        })

    elapsed = time.time() - start
    logger.info(f'Generated {len(records)} descriptions in {elapsed:.2f}s')

    filename = get_description_filename(prompt_slug='desc')
    out_path = get_description_path(filename)
    payload = {
        'created_at': datetime.now().isoformat(),
        'system_prompt': SYSTEM_PROMPT,
        'prompt': ABSTRACT_PROMPT(range(8)),
        'model_path': MODEL_PATH,
        'records': records
    }
    with open(out_path, 'w') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    combine_abstract(out_path)

    logger.info(f'Descriptions saved to: {out_path}')
    return out_path


if __name__ == '__main__':
    generate_descriptions()