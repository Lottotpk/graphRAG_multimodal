from qdrant_client import QdrantClient, models
from typing import Callable
import os
import torch

persistent_path = "/uac/y22/tpipatpajong2/qdrant_db"
client = QdrantClient(path=persistent_path)

def create_vectordb(video_dir: str, 
                    embed_func: Callable[[str], torch.Tensor],
                    collection_name: str,
                    ndim: int) -> None:
    count = 0
    points = []
    for filename in os.listdir(video_dir):
        count += 1
        file_path = os.path.join(video_dir, filename)
        points.append(models.PointStruct(id=count,
                                         vector=embed_func(file_path),
                                         payload={"path": file_path}))
        print(f"Done {count} video(s).")

    if not client.collection_exists(collection_name=collection_name):
        client.create_collection(
            collection_name,
            vectors_config=models.VectorParams(
                size=ndim,
                distance=models.Distance.COSINE,
                datatype=models.Datatype.FLOAT16,
                multivector_config=models.MultiVectorConfig(
                    comparator=models.MultiVectorComparator.MAX_SIM
                ),
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

    op_info = client.upsert(
        collection_name=collection_name,
        wait=True,
        points=points,
    )
    print(op_info)


def retrieval(query_vector: torch.Tensor, collection_name: str, top_k: int, report: bool = False):
    retrieved = client.query_points(
        collection_name=collection_name,
        query=query_vector.float().cpu().numpy(),
        limit=top_k,
        with_payload=True,
    ).points
    if report:
        for item in retrieved:
            print(f"From the query, {item.payload['path']} is retrieved")
    return retrieved