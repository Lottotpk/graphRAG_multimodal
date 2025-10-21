"""
Retrieve (the top-k) images from the query. The query should be about describing certain specific components of the image.
"""

import os
import torch
import argparse
import json
import logging
import time
from datetime import datetime
from typing import List, Dict, Any, Tuple
from transformers import AutoModel, AutoTokenizer
from text_embedding.text_embedder import TextEmbedder
from vector_db.faiss_storage import FAISSEmbeddingDatabase
from config import MODEL_PATH, DEVICE_MAP, TORCH_DTYPE, TEXT_DATABASE_BASE_DIR, ensure_prompt_description_dir, ensure_retrieval_result_dir, get_description_filename, get_description_path
from retrieval.prompt import ABSTRACT_PROMPT, SUMMARY_PROMPT, SYSTEM_PROMPT

from utils.logging_config import setup_logger

setup_logger()
logger = logging.getLogger(__name__)


# -----------------------------
# Edit these values
# -----------------------------

# Generation parameters
MAX_NEW_TOKENS: int = 1024
DO_SAMPLE: bool = True
TEMPERATURE: float = 1e-4
TOP_P: float = 0.9
# VERBOSE: bool = True

PRESENT_DIR = "./retrieval/"
LATEST_IMAGE_DB_PATH: str = os.path.join(TEXT_DATABASE_BASE_DIR, os.listdir(TEXT_DATABASE_BASE_DIR)[-1])

# QUERY = """A man cycling in the middle of the rain\n
# Person: A male, cycling on the street.\nBicycle: Rided by the cyclist.\nWeather: Raining outside.\n
# Person is riding the bicycle in the rain.\n
# \n
# """

QUERY = [
    "A male chef in a white suit cooking in the high-end restaurant image"
]

STRATEGY_CHOICES = [
    'last_layer_token_level',
    'last_layer_cls_token',
    'last_layer_mean_pooling',
    '-2_layer_token_level',
    '-2_layer_cls_token',
    '-2_layer_mean_pooling',
]



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


def describe_single(model, tokenizer, question: str, type: str = "", verbose: bool = False) -> str:
    start = time.time()
    gen_cfg = dict(
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=DO_SAMPLE,
        temperature=TEMPERATURE,
        top_p=TOP_P,
    )
    response = model.chat(tokenizer, None, question, gen_cfg)
    if verbose:
        elapsed = time.time() - start
        logger.info(f'Generated {type} description in {elapsed:.2f}s')
    return response


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


def load_database_info(database_folder: str) -> Dict[str, Any]:
    """Load basic database information"""
    metadata_path = os.path.join(database_folder, 'metadata.json')
    config_path = os.path.join(database_folder, 'config.json')
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    return metadata, config


def load_benchmark(filename: str, ds_name: str) -> Tuple[List[str]]:
    """Load existing benchmark"""
    with open(filename, 'r') as f:
        ds = json.load(f)

    questions = []
    answers = []
    for qa in ds["qa_pairs"]:
        questions.append(qa["question"])
        answers.append(os.path.join(ds_name, qa["answer"]) + ".jpg")

    return questions, answers, ds["total_questions"]


def process_query(model, tokenizer, queries: List[str], verbose: bool = False) -> List[str]:
    ensure_prompt_description_dir()
    prompt = []

    for query in queries:
        model.system_message = SYSTEM_PROMPT
        summary = describe_single(model, tokenizer, SUMMARY_PROMPT(query), "summary", verbose)
        summary = json.loads(summary)
        abstract = describe_single(model, tokenizer, ABSTRACT_PROMPT(range(8), query), "abstract", verbose)
        abstract = json.loads(abstract)
        prompt.append(construct_prompt(abstract, summary))

    payload = {
        'created_at': datetime.now().isoformat(),
        'model_path': MODEL_PATH,
        'query': queries,
        'format_prompt': prompt
    }
    filename = get_description_filename(prompt_slug='prompt')
    out_path = get_description_path(filename, "prompt")
    with open(out_path, 'w') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    logger.info(f'Descriptions saved to: {out_path}')

    return prompt


def main():
    parser = argparse.ArgumentParser(description='Retrieve image based on the given query')
    parser.add_argument("--db_dir", default=LATEST_IMAGE_DB_PATH, help='Path to FAISS db directory')
    parser.add_argument("--benchmark", type=str, help='Path to benchmark json file')
    parser.add_argument('--strategy', required=True, choices=STRATEGY_CHOICES, help='Text embedding strategy')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    # Load model and embedder
    model, tokenizer = load_model()
    embedder = TextEmbedder(model, tokenizer, device='cuda')

    # Load query from benchmark
    if args.benchmark:
        query, answer, total = load_benchmark(args.benchmark, "/research/d7/fyp24/tpipatpajong2/graphRAG_multimodal/dataset/Stanford40Action_ImageLabelDescripion10template5")
    else:
        query = QUERY
        total = 1

    # Format the query
    prompt = process_query(model, tokenizer, query, args.verbose)

    # Load db   
    metadata, config = load_database_info(args.db_dir)
    db = FAISSEmbeddingDatabase(
        database_folder=args.db_dir,
        embedding_dimension=config['embedding_dimension'],
        index_type=config.get('index_type', 'exact'),
        create_new=False,
    )

    # Embed the query and search using ColBERT
    k = 5
    results = []
    embeddings, _ = embedder.embed(prompt, args.strategy)
    for i in range(total):
        if args.verbose:
            logger.info(f"Query {i})")
        scores, img_path_retrieved = db.search_colbert(embeddings[i].float().numpy(), k=k)
        
        result = ""
        if args.benchmark:
            ranking = None
        for j in range(k):
            result += f"{img_path_retrieved[j]} -- {scores[j]}\n"
            if args.verbose:
                logger.info(f"{img_path_retrieved[j]} -- {scores[j]}")

            if args.benchmark and img_path_retrieved[j] == answer[i]:
                ranking = j
        
        if args.benchmark:
            qa = {"query": query[i], "ranking": ranking, "result": result}
        else:
            qa = {"query": query[i], "result": result}
        results.append(qa)

    ensure_retrieval_result_dir()
    filename = get_description_filename(prompt_slug='search_re')
    out_path = get_description_path(filename, "result")
    payload = {
        'created_at': datetime.now().isoformat(),
        'database': args.db_dir,
        'model_path': MODEL_PATH,
        'result': results
    }
    with open(out_path, 'w') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    logger.info(f'Descriptions saved to: {out_path}')

    

if __name__ == "__main__":
    main()
    # model, tokenizer = load_model()
    # print(describe_single(model, tokenizer, ABSTRACT_PROMPT(range(8), QUERY), "abstract", True))