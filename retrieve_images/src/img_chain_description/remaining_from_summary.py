#!/usr/bin/env python3
"""
Entities and relations Generator (uses local InternVL3_5-8B chat API)

Edit the variables in this file to specify image sources and the prompt.
Outputs a timestamped JSON file under ./image_description_databases/ with:
{
  "created_at": iso,
  "prompt": str,
  "model_path": str,
  "records": [ {"image_path": str, "description": str|None, "error": str|None} ]
}
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
import logging
from utils.logging_config import setup_logger
from img_chain_description.prompt_gen_remaining import ABSTRACT_PROMPT, SYSTEM_PROMPT
from img_chain_description.img_summary_gen import _generate_summary

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
    IMAGE_ONLY_SUMMARY_DIR, ensure_image_description_dir,
    get_description_filename, get_description_path,
    SUPPORTED_IMAGE_EXTENSIONS
)


# -----------------------------
# Edit these values
# -----------------------------

# List of sources; each entry can be a directory or a single image file path
IMAGE_SOURCE_PATHS: List[str] = [
    '/research/d7/fyp24/tpipatpajong2/graphRAG_multimodal/dataset/stanford_example',
]

# Image summary directory
# IMAGE_SUMMARY_PATH: str = os.path.join(IMAGE_ONLY_SUMMARY_DIR, "2025-10-12_22-20-48_desc.json")
LATEST_IMAGE_SUMMARY_PATH: str = os.path.join(IMAGE_ONLY_SUMMARY_DIR, os.listdir(IMAGE_ONLY_SUMMARY_DIR)[-1])


# Prompt written at the top of the output JSON and used for generation

# Generation parameters
MAX_NEW_TOKENS: int = 1024
DO_SAMPLE: bool = True
TEMPERATURE: float = 0.1
TOP_P: float = 0.9
# VERBOSE: bool = True

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


def generate_summary(args):
    ensure_image_description_dir("summary")

    images = _collect_images_from_sources(args.img_dir)
    if not images:
        logger.info('No images found from IMAGE_SOURCE_PATHS; nothing to do.')
        return None

    # if VERBOSE:
    if args.verbose:
        logger.info(f'Found {len(images)} images. Loading model...')

    model, tokenizer = _load_model()

    gen_cfg = dict(
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=DO_SAMPLE,
        temperature=TEMPERATURE,
        top_p=TOP_P,
    )

    with open(args.summ_dir, 'r') as f:
        summary = json.load(f)
        summary = summary["records"]
    
    PROMPT = ABSTRACT_PROMPT("")

    records = []
    start = time.time()

    # 1) Parallel CPU image loading
    # if VERBOSE:
    if args.verbose:
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
        out_path = get_description_path(filename, "summary")
        payload = {
            'created_at': datetime.now().isoformat(),
            'system_prompt': SYSTEM_PROMPT,
            'prompt': PROMPT,
            'model_path': MODEL_PATH,
            'records': records
        }
        with open(out_path, 'w') as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        logger.info(f'Descriptions saved to: {out_path}')
        return out_path

    # 2) GPU batched captioning using batch_chat when available
    total = len(successes)
    # if VERBOSE:
    if args.verbose:
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

        questions = [f'<image>\n{ABSTRACT_PROMPT(summary[path]["description"]["summary"])}' for path in batch_paths]
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
                    remaining = json.loads(resp.strip("```json"))
                    records.append({
                        'image_path': pth,
                        'description': dict(summary[pth]["description"], **remaining),
                        'error': None
                    })
            else:
                # Fallback: per-image chat
                for t, pth in zip(batch_tensors, batch_paths):
                    try:
                        resp = model.chat(tokenizer, t, questions[0], gen_cfg)
                        remaining = json.loads(resp.strip("```json"))
                        records.append({
                            'image_path': pth,
                            'description': dict(summary[pth]["description"], **remaining),
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
            # if VERBOSE:
            if args.verbose:
                logger.info(f'Batch captioning failed, falling back per-image: {be}')
            for t, pth in zip(batch_tensors, batch_paths):
                try:
                    resp = model.chat(tokenizer, t, f'<image>\n{ABSTRACT_PROMPT(summary[pth]["description"]["summary"])}', gen_cfg)
                    with open(f"json_error/summ_error_{datetime.now().isoformat()}.log", "a") as wfile:
                        wfile.write(str(resp) + '\n')
                    remaining = json.loads(resp.strip("```json"))
                    records.append({
                        'image_path': pth,
                        'description': dict(summary[pth]["description"], **remaining),
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
    out_path = get_description_path(filename, "summary")
    payload = {
        'created_at': datetime.now().isoformat(),
        'system_prompt': SYSTEM_PROMPT,
        'prompt': PROMPT,
        'model_path': MODEL_PATH,
        'records': records
    }
    with open(out_path, 'w') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    logger.info(f'Descriptions saved to: {out_path}')
    return out_path


def _generate_remaining(args, model, tokenizer, summary = None):
    # TODO: Improve scalability (see 2 previous files, do the same)

    images = _collect_images_from_sources(args.img_dir)
    if not images:
        logger.info('No images found from IMAGE_SOURCE_PATHS; nothing to do.')
        return None

    # if VERBOSE:
    if args.verbose:
        logger.info(f'Found {len(images)} images. Loading model...')

    gen_cfg = dict(
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=DO_SAMPLE,
        temperature=TEMPERATURE,
        top_p=TOP_P,
    )

    if summary == None:
        with open(args.summ_dir, 'r') as f:
            summary = json.load(f)
    summary = summary["records"]
    
    PROMPT = ABSTRACT_PROMPT("")

    records = []
    start = time.time()

    # 1) Parallel CPU image loading
    # if VERBOSE:
    if args.verbose:
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
        out_path = get_description_path(filename, "summary")
        payload = {
            'created_at': datetime.now().isoformat(),
            'system_prompt': SYSTEM_PROMPT,
            'prompt': PROMPT,
            'model_path': MODEL_PATH,
            'records': records
        }
        with open(out_path, 'w') as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        logger.info(f'Descriptions saved to: {out_path}')
        return out_path

    # 2) GPU batched captioning using batch_chat when available
    total = len(successes)
    # if VERBOSE:
    if args.verbose:
        logger.info(f'Generating captions on GPU in batches of {CAPTION_BATCH_SIZE}...')

    def _to_cuda_bf16(t: torch.Tensor) -> torch.Tensor:
        if t.dtype != torch.bfloat16:
            t = t.to(torch.bfloat16)
        if t.device.type != 'cuda':
            t = t.cuda()
        return t
    
    errors = []
    for start_idx in range(0, total, CAPTION_BATCH_SIZE):
        batch = successes[start_idx:start_idx + CAPTION_BATCH_SIZE]
        batch_paths = [p for _, p, _ in batch]
        batch_tensors = [_to_cuda_bf16(t) for t, _, _ in batch]
        num_patches_list = [int(t.shape[0]) for t in batch_tensors]

        questions = [f'<image>\n{ABSTRACT_PROMPT(summary[path]["description"]["summary"])}' for path in batch_paths]
        model.system_message = SYSTEM_PROMPT
        record = []

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
                    desc = resp.strip("```json")
                    begin = desc.find("{")
                    remaining = json.loads(desc[begin:])
                    record.append({
                        'image_path': pth,
                        'description': dict(summary[pth]["description"], **remaining),
                        'error': None
                    })
            else:
                # Fallback: per-image chat
                idx = 0
                for t, pth in zip(batch_tensors, batch_paths):
                    try:
                        resp = model.chat(tokenizer, t, questions[idx], gen_cfg)
                        desc = resp.strip("```json")
                        begin = desc.find("{")
                        remaining = json.loads(desc[begin:])
                        record.append({
                            'image_path': pth,
                            'description': dict(summary[pth]["description"], **remaining),
                            'error': None
                        })
                    except Exception as e:
                        record.append({
                            'image_path': pth,
                            'description': None,
                            'error': str(e)
                        })
                        errors.append(str(e))
                        logger.info(f"Description: None at {pth}: {str(e)} from {resp.strip("```json")}")
                    idx += 1

        except Exception as be:
            # Batch failed (e.g., OOM). Fallback to per-image sequential within this batch
            # if VERBOSE:
            if args.verbose:
                logger.info(f'Batch captioning failed, falling back per-image: {be}')
            record.clear()
            for t, pth in zip(batch_tensors, batch_paths):
                try:
                    resp = model.chat(tokenizer, t, f'<image>\n{ABSTRACT_PROMPT(summary[pth]["description"]["summary"])}', gen_cfg)
                    with open(f"json_error/summ_error_{datetime.now().isoformat()}.log", "a") as wfile:
                        wfile.write(str(resp) + '\n')
                    desc = resp.strip("```json")
                    begin = desc.find("{")
                    remaining = json.loads(desc[begin:])
                    record.append({
                        'image_path': pth,
                        'description': dict(summary[pth]["description"], **remaining),
                        'error': None
                    })
                except Exception as e:
                    record.append({
                        'image_path': pth,
                        'description': None,
                        'error': str(e)
                    })
                    errors.append(str(e))
                    logger.info(f"Description: None at {pth}: {str(e)} from {resp.strip("```json")}")
        records.extend(record)

    elapsed = time.time() - start
    logger.info(f'Generated {len(records)} descriptions in {elapsed:.2f}s')

    return {
        'created_at': datetime.now().isoformat(),
        'system_prompt': SYSTEM_PROMPT,
        'prompt': PROMPT,
        'model_path': MODEL_PATH,
        'records': records,
        'error': errors if errors else None
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Description based on image input and its text summary')
    parser.add_argument("--img_dir", nargs='*', default=IMAGE_SOURCE_PATHS, help='Path to image source')
    parser.add_argument("--summ_dir", default=LATEST_IMAGE_SUMMARY_PATH, help='Path to summary directory')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()
    model, tokenizer = _load_model()
    summary = _generate_summary(args, model, tokenizer)
    _generate_remaining(args, model, tokenizer, summary)
    # generate_summary(args)