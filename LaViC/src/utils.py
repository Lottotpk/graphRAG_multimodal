from qdrant_client import QdrantClient, models
from typing import Callable
import os
import torch
import logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

logging.info("Setting up QdrantClient")
persistent_path = "../db"
client = QdrantClient(path=persistent_path)
# client = QdrantClient(":memory:")
logging.info("QdrantClient Created")

def create_vectordb(video_dir: str, 
                    embed_func: Callable[[str], torch.Tensor],
                    collection_name: str,
                    ndim: int) -> None:
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
                max_segment_size=5_000_000,
            ),
            hnsw_config=models.HnswConfigDiff(
                m=6,
                on_disk=False,
            ),
        )
    logging.info("start embedding")
    count = 0
    points = []
    for filename in os.listdir(video_dir):
        count += 1
        file_path = os.path.join(video_dir, filename)
        points.append(models.PointStruct(id=count,
                                         vector=embed_func(file_path),
                                         payload={"path": file_path}))
        op_info = client.upsert(
            collection_name=collection_name,
            wait=True,
            points=points,
        )
        logging.info(f"Done {count} embeddings.")


def retrieval(query_vector: torch.Tensor, collection_name: str, top_k: int, report: bool = False):
    retrieved = client.query_points(
        collection_name=collection_name,
        query=query_vector.float().cpu().numpy(),
        limit=top_k,
        with_payload=True,
        # score_threshold=0.8,
    )
    # logging.info(retrieved)
    if report:
        for item in retrieved.points:
            logging.info(f"From the query, {item.payload['path']} is retrieved")
    return retrieved.points