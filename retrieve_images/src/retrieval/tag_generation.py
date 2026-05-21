import os
import torch
import argparse
import json
import logging
import time
import numpy as np
from datetime import datetime
from transformers import AutoModel, AutoTokenizer, AutoProcessor
from text_embedding.text_embedder import TextEmbedder
from vector_db.faiss_storage import FAISSEmbeddingDatabase
from utils.logging_config import setup_logger
from utils.json_processing import fix_json
from retrieval.prompt import TAG_PROMPT, TAG_SYSTEM_PROMPT
from retrieval.img_retrieve import process_query
from text_embedding.text_embedder import TextEmbedder
from text_embedding.text_embedding import load_model
from config import FORMAT_PROMPT_DIR, MODEL_PATH, get_description_path, get_description_filename

import math
from PIL import Image, ImageOps

STRATEGY_CHOICES = [
    'last_layer_token_level',
    'last_layer_cls_token',
    'last_layer_mean_pooling',
    '-2_layer_token_level',
    '-2_layer_cls_token',
    '-2_layer_mean_pooling',
]

# Generation parameters
MAX_NEW_TOKENS: int = 1024
DO_SAMPLE: bool = True
TEMPERATURE: float = 0.1
TOP_P: float = 0.9
 
LATEST_PROMPT_PATH: str = os.path.join(FORMAT_PROMPT_DIR, os.listdir(FORMAT_PROMPT_DIR)[-1])

setup_logger()
logger = logging.getLogger(__name__)


# vibe coded function
def combine_and_save_images(image_paths, output_filepath, images_per_row=3, img_size=(300, 300)):
    valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp')

    print(f"Found {len(image_paths)} images. Processing...")

    images =[]
    for path in image_paths:
        try:
            img = Image.open(path)
            
            # 1. Convert image to RGB (Prevents errors if you have transparent PNGs)
            img = img.convert('RGB')
            
            # 2. SMART RESIZE: Crops from the center to perfectly fit the grid size 
            # without squishing or stretching the image!
            img = ImageOps.fit(img, img_size, Image.Resampling.LANCZOS)
            
            images.append(img)
        except Exception as e:
            print(f"Could not read {path}. Error: {e}")

    if not images:
        return

    # Calculate grid dimensions
    num_images = len(images)
    rows = math.ceil(num_images / images_per_row)
    grid_width = images_per_row * img_size[0]
    grid_height = rows * img_size[1]

    # Create a blank white canvas
    combined_image = Image.new('RGB', (grid_width, grid_height), color='white')

    # Paste images into the grid
    for index, img in enumerate(images):
        row = index // images_per_row
        col = index % images_per_row
        
        x_offset = col * img_size[0]
        y_offset = row * img_size[1]
        
        combined_image.paste(img, (x_offset, y_offset))

    # Display and Save
    combined_image.show()
    combined_image.save(output_filepath)
    print(f"Success! Combined image saved to: {output_filepath}")


# TODO: SAVE THIS IN LOCAL DIRECTORY FIRST (IF BENCHMARKING) -> take 10-15s per query
# TODO: HANDLE OUTPUT ERROR (LIKE REPEAT WORDS, WRONG FORMAT, ETC.)
# TODO: (Optional?) We can add image to the prompt to add more context for tag generations
def generate_tags(model, tokenizer, query: str, verbose: bool = False):
    start = time.time()
    if MODEL_PATH.split("/")[0] == "Qwen":
        logger.info("Qwen detected, Importing processor...")
        processor = AutoProcessor.from_pretrained(MODEL_PATH)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": TAG_PROMPT(query)}
                ]
            },
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": TAG_PROMPT}
                ]
            }
        ]
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        ).to(model.device)
        generated_ids = model.generate(**inputs, max_new_tokens=512)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        if verbose:
            elapsed = time.time() - start
            logger.info(f'Generated tags in {elapsed:.2f}s')
        return json.loads(fix_json(output_text[0]))
    else:
        gen_cfg = dict(
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=DO_SAMPLE,
            temperature=TEMPERATURE,
            top_p=TOP_P,
        )
        model.system_message = TAG_SYSTEM_PROMPT
        response = model.chat(tokenizer, None, TAG_PROMPT(query), gen_cfg)
        if verbose:
            elapsed = time.time() - start
            logger.info(f'Generated tags in {elapsed:.2f}s')
        return json.loads(fix_json(response))


def embed_tags(embedder, strategy: str, tags: dict, verbose: bool = False):
    start = time.time()
    tags_list = []
    for dimension in tags["dimensions"]:
        tags_list.extend(dimension["candidates"])
    embedding, _ = embedder.embed(tags_list, strategy)
    embedding = [x.float().numpy() for x in embedding]
    embedding = [x / np.linalg.norm(x) for x in embedding]
    if verbose:
        elapsed = time.time() - start
        logger.info(f'Embedded tags in {elapsed:.2f}s with shape ({len(embedding)}, {len(embedding[0])})')

    idx = 0
    dimensions = tags["dimensions"]
    for i in range(len(dimensions)):
        dimensions[i]["embeddings"] = np.array(embedding[idx:idx+len(dimensions[i]["candidates"])])
        idx += len(dimensions[i]["candidates"])
    
    return dimensions 


def get_relevant_tags(dimensions, retrieved_matrix, threshold: list[float, float, int]):
    """threshold: (similarity_threshold, coverages_threshold, hit_threshold)"""
    passed_tags = []
    # logger.info(retrieved_matrix)
    for i in range(len(dimensions)):
        similarity = retrieved_matrix @ dimensions[i]["embeddings"].T
        # logger.info(similarity)
        del dimensions[i]["embeddings"]
        dimensions[i]["similarity_matrix"] = similarity

        matched = (similarity >= threshold[0])
        covered = matched.sum(axis=0) / matched.shape[0]
        valid_tags = [dimensions[i]["candidates"][j] for j in np.where(covered > threshold[1])[0]]
        if len(valid_tags) >= threshold[2]:
            passed_tags.extend(valid_tags)

    return passed_tags, dimensions


def get_images_from_tag(dimensions, tag, retrieved_images_id, threshold=0.75):
    for dimension in dimensions:
        if tag in dimension["candidates"]:
            idx = dimension["candidates"].index(tag)
            scores = dimension["similarity_matrix"][:, idx]
            matched_indices = np.where(scores >= threshold)[0]
            sorted_indices = matched_indices[np.argsort(scores[matched_indices])[::-1]]
            return [retrieved_images_id[i] for i in sorted_indices]
    logger.info("Invalid tag")
    return None


def load_database_info(database_folder: str):
    """Load basic database information"""
    metadata_path = os.path.join(database_folder, 'metadata.json')
    config_path = os.path.join(database_folder, 'config.json')
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    return metadata, config


def main(args):
    
    # 0. Preparation
    model, tokenizer = load_model()
    embedder = TextEmbedder(model, tokenizer, 'cuda')
    payload = None
    if args.input == None or not os.path.exists(args.input):
        logger.info("Cannot find directory, or directory is None, use the latest file instead")
        with open(LATEST_PROMPT_PATH, 'r') as f:
            payload = json.load(f)
    else:
        with open(args.input, 'r') as f:
            logger.info(f"Loading generated text from {args.input}")
            payload = json.load(f)
    items = payload.get('records', [])
    # Load db  
    metadata, config = load_database_info(args.db_dir)
    db = FAISSEmbeddingDatabase(
        database_folder=args.db_dir,
        embedding_dimension=config['embedding_dimension'],
        index_type=config.get('index_type', 'exact'),
        create_new=False,
    )

    start_time = time.time()
    results = []
    for it in items:
        while True: # Handles error, keep retrying
            try:
                # 1. Generate + Embed tags
                tags = generate_tags(model, tokenizer, it['format_prompt'], True)
                embedded_tags = embed_tags(embedder, args.strategy, tags, True)

                # 2. Embed prompt
                embedding, _ = embedder.embed(it['format_prompt'], args.strategy)
                scores, indices = db.search_similar(embedding[0].float().numpy(), k=args.top_k)
                retrieved_matrix = np.array([db.get_embedding_by_id(x) for x in indices]) # Only addition to the original (?)

                # 3. Compute similarity and output relevant tags
                output, embedded_tags = get_relevant_tags(embedded_tags, retrieved_matrix, args.threshold)
                logger.info(f"QUERY: {it['id']}, TAGS: {output}")
                results.append({
                    "id": it['id'],
                    "prompt": it['format_prompt'],
                    "tags": output,
                })
                break
            except Exception as e:
                logger.info(f"Error occurred {e}\n Try Again...")
    
    # Save results
    elapse = time.time() - start_time
    out_path = get_description_path(get_description_filename(""), "tag")
    payloads = {
            'created_at': datetime.now().isoformat(),
            'database': args.db_dir,
            'model_path': MODEL_PATH,
            'time_elapsed': elapse,
            'average_time': elapse / len(results),
            'result': results
        }
    with open(out_path, 'w') as f:
        json.dump(payloads, f, indent=2, ensure_ascii=False)
    logger.info(f'Descriptions saved to: {out_path}')


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Generate tag script based on query")
    # parser.add_argument('--input', help="query file for benchmarking")
    # parser.add_argument('--strategy', required=True, choices=STRATEGY_CHOICES, help='Text embedding strategy')
    # parser.add_argument('--db_dir', required=True, help='Vector DB path')
    # parser.add_argument('--top_k', default=50, type=int, help='top k results to generate tags')
    # parser.add_argument('--threshold', default=[0.65, 0.2, 1], type=float, nargs=3, help='threshold value to filter tags')

    # args = parser.parse_args()
    # main(args)

    # Sketches (not used anymore)
    model, tokenizer = load_model()
    db_dir = "./description_embedding_databases/2025-11-13_21-33-13_-2_layer_mean_pooling_auto"
    metadata, config = load_database_info(db_dir)
    db = FAISSEmbeddingDatabase(
        database_folder=db_dir,
        embedding_dimension=config['embedding_dimension'],
        index_type=config.get('index_type', 'exact'),
        create_new=False,
    )
    # 1. Generate tags
    test_prompt1 = "An elderly man wearing a suit and medals is clapping his hands."
    test_prompt2 = "An elderly man"
    test_prompt3 = "Party"
    ex1 = generate_tags(model, tokenizer, test_prompt1, True)
    ex2 = generate_tags(model, tokenizer, test_prompt2, True)
    ex3 = generate_tags(model, tokenizer, test_prompt3, True)

    # 2. Embed tags into matrix
    embedding_method = "-2_layer_mean_pooling"
    embedder = TextEmbedder(model, tokenizer, device='cuda')
    ex1 = embed_tags(embedder, embedding_method, ex1, True)
    ex2 = embed_tags(embedder, embedding_method, ex2, True)
    ex3 = embed_tags(embedder, embedding_method, ex3, True)

    k = 50
    processed_prompt = process_query(model, tokenizer, [test_prompt1, test_prompt2, test_prompt3], True)
    embedding, _ = embedder.embed(processed_prompt, embedding_method) # 
    scores, indices1 = db.search_similar(embedding[0].float().numpy(), k=k) # throw away search_colbert because time overhead
    # Get embedding from indices
    retrieved_matrix1 = np.array([db.get_embedding_by_id(x) for x in indices1]) # Only addition to the original (?)
    # Do the same for other queries
    print([db.get_metadata_by_id(x)['image_path'] for x in indices1])
    combine_and_save_images([db.get_metadata_by_id(x)['image_path'] for x in indices1][:20], "./retrieved_images.png", 5, (400,400))
    scores, indices2 = db.search_similar(embedding[1].float().numpy(), k=k)
    retrieved_matrix2 = np.array([db.get_embedding_by_id(x) for x in indices2])
    scores, indices3 = db.search_similar(embedding[2].float().numpy(), k=k)
    retrieved_matrix3 = np.array([db.get_embedding_by_id(x) for x in indices3])

    # 4. Compute Similarity and output relevant tags
    threshold = [0.75, 0.2, 1] # threshold[0] may be adjusted by the user (?)
    output1, ex1 = get_relevant_tags(ex1, retrieved_matrix1, threshold)
    output2, ex2 = get_relevant_tags(ex2, retrieved_matrix2, threshold)
    output3, ex3 = get_relevant_tags(ex3, retrieved_matrix3, threshold)

    # Done
    logger.info(f"QUERY: {test_prompt1}\nTAGS: {output1}")
    logger.info(f"QUERY: {test_prompt2}\nTAGS: {output2}")
    logger.info(f"QUERY: {test_prompt3}\nTAGS: {output3}")

    # 5. Get additional image results
    while True:
        selected_tag = input("Select tag: ")
        if selected_tag == "":
            break
        additional_idx = get_images_from_tag(ex1, selected_tag, indices1)
        logger.info(f"ADDTIONAL RETRIEVED IMGES:\n{[db.get_metadata_by_id(x)['image_path'] for x in additional_idx]}")
        combine_and_save_images([db.get_metadata_by_id(x)['image_path'] for x in additional_idx], "./tag_retrieved.png", 4, (400,400))