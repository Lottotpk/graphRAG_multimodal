import os
import torch
import argparse
import json
import logging
import time
import numpy as np
from datetime import datetime
from transformers import AutoModel, AutoTokenizer
from text_embedding.text_embedder import TextEmbedder
from vector_db.faiss_storage import FAISSEmbeddingDatabase
from utils.logging_config import setup_logger
from utils.json_processing import fix_json
from retrieval.prompt import TAG_PROMPT, TAG_SYSTEM_PROMPT

# Generation parameters
MAX_NEW_TOKENS: int = 1024
DO_SAMPLE: bool = True
TEMPERATURE: float = 0.1
TOP_P: float = 0.9

setup_logger()
logger = logging.getLogger(__name__)


# TODO: SAVE THIS IN LOCAL DIRECTORY FIRST (IF BENCHMARKING) -> take 10-15s per query
# TODO: HANDLE OUTPUT ERROR (LIKE REPEAT WORDS, WRONG FORMAT, ETC.)
def generate_tags(model, tokenizer, query: str, verbose: bool = False):
    start = time.time()
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
    for i in range(len(dimensions)):
        similarity = retrieved_matrix @ dimensions[i]["embeddings"].T
        del dimensions[i]["embeddings"]
        dimensions[i]["similarity_matrix"] = similarity

        matched = (similarity >= threshold[0])
        covered = matched.sum(axis=0) / matched.shape[0]
        valid_tags = [dimensions[i]["candidates"][j] for j in np.where(covered > threshold[1])[0]]
        if len(valid_tags) >= threshold[2]:
            passed_tags.extend(valid_tags)

    return passed_tags, dimensions


def get_images_from_tag(dimensions, tag, retrieved_images_id, threshold=0.7):
    for dimension in dimensions:
        if tag in dimension["candidates"]:
            idx = dimension["candidates"].index(tag)
            scores = dimension["similarity_matrix"][:, idx]
            matched_indices = np.where(scores >= threshold)[0]
            sorted_indices = matched_indices[np.argsort(scores[matched_indices])[::-1]]
            return [retrieved_images_id[i] for i in sorted_indices]
    logger.info("Invalid tag")
    return None


# FOR TESTING ONLY
if __name__ == "__main__":
    from config import MODEL_PATH, DEVICE_MAP, TORCH_DTYPE
    from text_embedding.text_embedder import TextEmbedder
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

    # 1. Generate tags
    test_prompt1 = "An elderly man wearing a suit and medals is clapping his hands."
    test_prompt2 = "An elderly man, dressed formally in a dark suit with a patterned tie, is seen clapping his hands in what appears to be a public or ceremonial event. He is wearing several medals around his neck and on his jacket, suggesting he has been honored for his achievements or service. His expression appears engaged and respectful, while photographers and others can be seen in the background, capturing the moment."
    test_prompt3 = "- Objects: The man is clapping his hands.\n- Attributes: He is wearing a formal suit with medals.\n- Actions: He is expressing approval or celebration.\n- Scene: The man is standing in front of a crowd.\n- Style: The photograph has a candid and candid style."
    ex1 = generate_tags(model, tokenizer, test_prompt1, True)
    ex2 = generate_tags(model, tokenizer, test_prompt2, True)
    ex3 = generate_tags(model, tokenizer, test_prompt3, True)

    # 2. Embed tags into matrix
    embedding_method = "last_layer_mean_pooling"
    embedder = TextEmbedder(model, tokenizer, device='cuda')
    ex1 = embed_tags(embedder, embedding_method, ex1, True)
    ex2 = embed_tags(embedder, embedding_method, ex2, True)
    ex3 = embed_tags(embedder, embedding_method, ex3, True)

    # 3. Top-k result from DB (implemented, kind of)
    def load_database_info(database_folder: str):
        """Load basic database information"""
        metadata_path = os.path.join(database_folder, 'metadata.json')
        config_path = os.path.join(database_folder, 'config.json')
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        return metadata, config
    # Load db
    db_path = "./description_embedding_databases/2025-11-13_21-40-54_last_layer_mean_pooling_auto"   
    metadata, config = load_database_info(db_path)
    db = FAISSEmbeddingDatabase(
        database_folder=db_path,
        embedding_dimension=config['embedding_dimension'],
        index_type=config.get('index_type', 'exact'),
        create_new=False,
    )
    k = 50
    embedding, _ = embedder.embed([test_prompt1, test_prompt2, test_prompt3], embedding_method)
    scores, indices1 = db.search_similar(embedding[0].float().numpy(), k=k) # throw away search_colbert because time overhead
    # Get embedding from indices
    retrieved_matrix1 = np.array([db.get_embedding_by_id(x) for x in indices1]) # Only addition to the original (?)
    # Do the same for other queries
    scores, indices2 = db.search_similar(embedding[1].float().numpy(), k=k)
    retrieved_matrix2 = np.array([db.get_embedding_by_id(x) for x in indices2])
    scores, indices3 = db.search_similar(embedding[2].float().numpy(), k=k)
    retrieved_matrix3 = np.array([db.get_embedding_by_id(x) for x in indices3])

    # 4. Compute Similarity and output relevant tags
    threshold = [0.65, 0.2, 1] # threshold[0] may be adjusted by the user (?)
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