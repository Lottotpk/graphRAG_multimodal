# from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models
import logging
import argparse
import torch
import json

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

model = SentenceTransformer(
    "Qwen/Qwen3-Embedding-8B",
    model_kwargs={"attn_implementation": "flash_attention_2", "device_map": "auto", "torch_dtype": torch.float16},
    tokenizer_kwargs={"padding_side": "left"},
)

logging.info("Setting up QdrantClient")
persistent_path = "/research/d7/fyp24/tpipatpajong2/LaViC/db"
client = QdrantClient(path=persistent_path)
logging.info("QdrantClient created")

def create_vectordb(collection_name, ndim, vector, img_id):
    logging.info(f"checking collection: {collection_name}")
    if not client.collection_exists(collection_name=collection_name):
        logging.info(f"creating collection: {collection_name}")
        client.create_collection(
            collection_name,
            vectors_config=models.VectorParams(
                size=ndim,
                distance=models.Distance.COSINE,
                datatype=models.Datatype.FLOAT16,
                # multivector_config=models.MultiVectorConfig(
                #     comparator=models.MultiVectorComparator.MAX_SIM
                # ),
                on_disk=True,   
            ),
            optimizers_config=models.OptimizersConfigDiff(
                # max_segment_size=5_000_000,
                indexing_threshold=0,
            ),
            # hnsw_config=models.HnswConfigDiff(
            #     m=6,
            #     on_disk=False,
            # ),
        )
    logging.info("Start upserting")
    op_info = client.upsert(
        collection_name=collection_name,
        wait=True,
        points=models.Batch(
            ids=[i for i in range(len(vector))], 
            vectors=vector,
            payloads=img_id,
        ),
    )
    print(op_info)
    logging.info("Finished upserting")
    client.update_collection(
        collection_name=collection_name,
        optimizer_config=models.OptimizersConfigDiff(indexing_threshold=20000),
    )


def retrieval(query_vector: torch.Tensor, collection_name: str, top_k: int, report: bool = False):
    retrieved = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=top_k,
        with_payload=True,
        # score_threshold=0.8,
    )
    # logging.info(retrieved)
    if report:
        for item in retrieved.points:
            logging.info(f"From the query, {item.payload['path']} is retrieved")
    return retrieved.points


def embed_all_texts(args):
    # Read json file
    with open(args.train_data, "r", encoding="utf-8") as f:
        train_data = json.load(f)
    val_data = []
    with open(args.val_data, "r", encoding="utf-8") as f:
        for line in f:
            val_data.append(json.loads(line))
    
    # Embed train image description
    train_text_list = []
    train_img_list = []
    for _, value in train_data.items():
        description = value["image_descriptions_llava_cleaned"]
        for img, desc in description.items():
            train_text_list.append(desc)
            train_img_list.append({"filename": img})
    encoded_text = model.encode(train_text_list)
    create_vectordb("qwen", 4096, encoded_text, train_img_list)

    # Embed validation image description
    # val_text_list = []
    # val_img_list = []
    # for entry in val_data:
    #     val_text_list.append(entry["image_descriptions_llava_cleaned"])
    #     val_img_list.append({"filename": entry["image_name"]})
    # encoded_text = model.encode(val_text_list)
    # create_vectordb("val", 4096, encoded_text, val_img_list)


def main():
    parser = argparse.ArgumentParser(description="Distill Vision Embeddings with LoRA")
    parser.add_argument("--train_data", type=str, default="../data/item2meta_train.json", help="Path to training data")
    parser.add_argument("--val_data", type=str, default="../data/item2meta_valid.jsonl", help="Path to validation data")
    
    args = parser.parse_args()

    # Embed all of the description (both train and validation set)
    # embed_all_texts(args)

    # Append the retrieval to the jsonl file in all categories
    categories = ["all_beauty", "amazon_fashion", "amazon_home"]
    dataset_type = ["train", "valid", "test"]
    for category in categories:
        for set_type in dataset_type:
            file_path = f"../data/{category}/{set_type}.jsonl"
            output_path = f"../data/{category}/{set_type}2.jsonl"
            data = []
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    entry = json.loads(line)
                    retrieved = retrieval(model.encode(entry["context"]), "qwen", 10)
                    candidates = []
                    for i in range(len(retrieved)):
                        candidates.append(retrieved[i].payload["filename"].split("_")[0])
                    entry["candidates_qwen"] = candidates
                    data.append(entry)
            
            with open(output_path, "w") as f:
                for item in data:
                    f.write(json.dumps(item) + "\n")
            logging.info(f"File dumped into {output_path}")




if __name__ == "__main__":
    main()