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
from json.decoder import JSONDecodeError
from typing import List, Dict, Any, Tuple
from transformers import AutoModel, AutoTokenizer
from text_embedding.text_embedder import TextEmbedder
from vector_db.faiss_storage import FAISSEmbeddingDatabase
from config import MODEL_PATH, DEVICE_MAP, TORCH_DTYPE, TEXT_DATABASE_BASE_DIR, ensure_prompt_description_dir, ensure_retrieval_result_dir, get_description_filename, get_description_path
from retrieval.prompt import ABSTRACT_PROMPT, SUMMARY_PROMPT, SYSTEM_PROMPT, SUMMARY, ENTITY, RELATION

from utils.logging_config import setup_logger

setup_logger()
logger = logging.getLogger(__name__)


# -----------------------------
# Edit these values
# -----------------------------

# Generation parameters
MAX_NEW_TOKENS: int = 1024
DO_SAMPLE: bool = True
TEMPERATURE: float = 0.1
TOP_P: float = 0.9
# VERBOSE: bool = True

PRESENT_DIR = "./retrieval/"
LATEST_IMAGE_DB_PATH: str = os.path.join(TEXT_DATABASE_BASE_DIR, os.listdir(TEXT_DATABASE_BASE_DIR)[-1])
NUM_TOPIC: int = 2

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


def describe_single(model, tokenizer, question: str, type: str = "", temp: float = TEMPERATURE,verbose: bool = False) -> str:
    start = time.time()
    gen_cfg = dict(
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=DO_SAMPLE,
        temperature=temp,
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


def load_benchmark(filename: str, ds_name: str) -> Tuple[List[str], List[str], int]:
    """Load existing benchmark"""
    with open(filename, 'r') as f:
        ds = json.load(f)

    questions = []
    answers = []
    for qa in ds["qa_pairs"]:
        questions.append(qa["question"])
        answers.append(qa["answer"])

    return questions, answers, ds["total_questions"]


def process_query(model, tokenizer, queries: List[str], verbose: bool = False) -> List[str]:
    ensure_prompt_description_dir()
    prompts = []
    format_prompt = []

    for i, query in enumerate(queries):
        model.system_message = SYSTEM_PROMPT
        loop = 0
        prompt = {
            'id': i,
            'query': query,
            'format_prompt': None
        }
        while True:
            try:
                summary = describe_single(model, tokenizer, SUMMARY_PROMPT(query), "summary", TEMPERATURE + loop * 0.1, verbose)
                summary = json.loads(summary)
                loop = 0
            except JSONDecodeError as e:
                logger.info(f"Summary - Error at index {i}: {e}, Trying again... ({2 - loop} times left)")
                logger.info(summary)
                loop += 1
                if loop == 3:
                    logger.info(f"Summary - Falling back to per topic at a time")
                    topic = "summary"
                    try:
                        records = []
                        record = describe_single(model, tokenizer, SUMMARY(query), topic, verbose)
                        records.append(json.loads(record))
                        topic = "entity"
                        record = describe_single(model, tokenizer, ENTITY(query), topic, verbose)
                        records.append(json.loads(record))
                        topic = "relation"
                        record = describe_single(model, tokenizer, RELATION(query), topic, verbose)
                        records.append(json.loads(record))
                        summary = {key: val for record in records for key, val in record.items()}
                        loop = 0
                    except JSONDecodeError as e:
                        logger.info(f"Summary - Error at index {i}: {e} with {topic}")
                        logger.info(record)
                        break
                else:
                    continue
            try:
                abstract = describe_single(model, tokenizer, ABSTRACT_PROMPT(range(8), query), "abstract", TEMPERATURE + loop * 0.1, verbose)
                abstract = json.loads(abstract)
                loop = 0
            except JSONDecodeError as e:
                logger.info(f"Abstract - Error at index {i}: {e}, Trying again... ({2 - loop} times left)")
                logger.info(abstract)
                loop += 1
                if loop == 3:
                    logger.info(f"Abstract - Falling back to {NUM_TOPIC} topics at a time")
                    try:
                        records = []
                        for j in range(0, 8, NUM_TOPIC):
                            record = describe_single(model, tokenizer, ABSTRACT_PROMPT(range(j, min(j + NUM_TOPIC, 8)), query), "abstract", verbose)
                            records.append(json.loads(record))
                        abstract = {key: val for record in records for key, val in record.items()}
                        loop = 0
                    except JSONDecodeError as e:
                        logger.info(f"Abstract - Error at index {i}: {e} with {NUM_TOPIC} topics, Try again...")
                        logger.info(record)
                        break
                else:
                    continue
            try:
                prompt['format_prompt'] = construct_prompt(abstract, summary)
                break
            except Exception as e:
                logger.info(f"Appending prompt - Error at index {i}: {e}, Trying again...")
                logger.info(summary)
                logger.info(abstract)
        
        prompts.append(prompt)
        format_prompt.append(prompt['format_prompt'])

    payload = {
        'created_at': datetime.now().isoformat(),
        'model_path': MODEL_PATH,
        'records': prompts
    }
    filename = get_description_filename(prompt_slug='prompt')
    out_path = get_description_path(filename, "prompt")
    with open(out_path, 'w') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    logger.info(f'Descriptions saved to: {out_path}')

    return format_prompt


def main(args):
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
    if not args.fprompt:
        prompt = process_query(model, tokenizer, query, args.verbose)
    else:
        with open(args.fprompt, "r") as f:
            data = json.load(f)
            prompt = [record['format_prompt'] for record in data['records']]

    if not prompt or None in prompt:
        raise ValueError("Prompt is not ready. There are \'None\' in prompt.")

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
    batch_size = 16
    embeddings = []
    for i in range(0, total, batch_size):
        batch_texts = prompt[i:i+batch_size]
        embedding, _ = embedder.embed(batch_texts, args.strategy)
        embeddings.extend(embedding)

    results = []
    if args.benchmark:
        rankings = [0] * k
    for i in range(total):
        qa = {
            "id": i,
            "query": query[i]
        }
        if args.verbose:
            logger.info(f"Query {i})")
        scores, img_path_retrieved = db.search_colbert(embeddings[i].float().numpy(), k=k)
        
        result = ""
        if args.benchmark:
            ranking = None
        for j in range(k):
            # TODO: Chage each full path img_path_retrieved[j] to only name
            img_name = os.path.basename(img_path_retrieved[j])
            img_name, ext = os.path.splitext(img_name)
            result += "%s -- %.4f\n" % (img_name, scores[j])
            if args.verbose:
                logger.info("%s -- %.4f\n" % (img_name, scores[j]))

            if args.benchmark and img_name == answer[i]:
                if args.verbose:
                    logger.info(f"The answer is at the Rank {j}")
                rankings[j] += 1
                ranking = j
        
        if args.benchmark:
            qa["ranking"] = ranking
            qa["right_ans"] = answer[i]
            qa["result"] = result
        else:
            qa["result"] = result
        results.append(qa)

    ensure_retrieval_result_dir()
    filename = get_description_filename(prompt_slug='search_re')
    out_path = get_description_path(filename, "result")
    if args.benchmark:
        payload = {
        'created_at': datetime.now().isoformat(),
        'database': args.db_dir,
        'model_path': MODEL_PATH,
        'ranking_stat': rankings,
        'result': results
        }
    else:
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
    parser = argparse.ArgumentParser(description='Retrieve image based on the given query')
    parser.add_argument("--db_dir", default=LATEST_IMAGE_DB_PATH, help='Path to FAISS db directory')
    parser.add_argument("--benchmark", type=str, help='Path to benchmark json file')
    parser.add_argument("--fprompt", type=str, help='Path to formated prompt')
    parser.add_argument('--strategy', required=True, choices=STRATEGY_CHOICES, help='Text embedding strategy')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()
    main(args)
    # model, tokenizer = load_model()
    # print(describe_single(model, tokenizer, ABSTRACT_PROMPT(range(8), QUERY), "abstract", True))