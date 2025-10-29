import os
import sys
import json
import argparse
import logging
import torch
from datetime import datetime
from transformers import AutoModel, AutoTokenizer
from utils.logging_config import setup_logger
from config import TORCH_DTYPE, MODEL_PATH, DEVICE_MAP, get_description_filename, get_description_path, ensure_image_description_dir

from img_chain_description.img_summary_gen import _generate_summary
from img_chain_description.desc_from_summary import _generate_abstract
from img_chain_description.remaining_from_summary import _generate_remaining

setup_logger()
logger = logging.getLogger(__name__)

# Ensure package imports work when run from anywhere
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)


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
    # TODO: Do JSON Stream for scalability (see json-stream library) since we cannot hold all data in-memory
    # TODO: Improve wrapper functionality to make it more customizable, e.g., select (or skip) step(s), select pre-defined summary, etc. 

    model, tokenizer = load_model()
    summary = _generate_summary(args, model, tokenizer)
    abstract = _generate_abstract(args, model, tokenizer, summary)
    remaining = _generate_remaining(args, model, tokenizer, summary)

    # TODO: Do JSON Stream here as well
    abst, rema = abstract['records'], remaining['records']
    if not (len(abst) == len(rema)):
        raise ValueError(f"The lengths of abstract and remaining are not the same ({len(abst)}, {len(rema)}), something went wrong.")
    records = []
    for i in range(len(abst)):
        records.append({
            'id': i,
            'image_path': rema[i]['image_path'],
            'summary': rema[i]['description']['summary'],
            'entities': rema[i]['description']['entities'],
            'relations': rema[i]['description']['relations'],
            'abstract': abst[i]['description']
        })
    
    ensure_image_description_dir('all')
    filename = get_description_filename(prompt_slug='all')
    out_path = get_description_path(filename, "all")
    payload = {
        'created_at': datetime.now().isoformat(),
        'summary_model_path': summary['model_path'], 
        'summary_system_prompt': summary['system_prompt'],
        'summary_prompt': summary['prompt'],
        'abstract_model_path': abstract['model_path'],
        'abstract_system_prompt': abstract['system_prompt'],
        'abstract_prompt': abstract['prompt'],
        'remaining_model_path': remaining['model_path'],
        'remaining_system_prompt': remaining['system_prompt'],
        'remaining_prompt': remaining['prompt'],
        'records': records,
        'error': None if not summary['error'] and not remaining['error'] else 'Something went wrong (check logs).'
    }
    with open(out_path, 'w') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    logger.info(f'Descriptions saved to: {out_path}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="The wrapper script to generate all elements of images")
    parser.add_argument("--img_dir", nargs='*', required=True, help='Path(s) to image source')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()
    main(args)