import os
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from vectordb import VectorDB
from qdrant_client import QdrantClient, models

device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def image_embedding(file_path: str) -> torch.Tensor:
    image = Image.open(file_path)

    if image.format == 'PNG' and image.mode != 'RGBA':
        image.convert('RGBA')

    inputs = clip_processor(images=image, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        outputs = clip_model.vision_model(**inputs)
    tokens = outputs.last_hidden_state.squeeze(0)
    visual_proj = clip_model.visual_projection(tokens)
    return torch.nn.functional.normalize(visual_proj, dim=-1)


def store_db(dataset_path: str = os.getenv("COLLECTION_PATH")) -> None:
    count = 0
    points = []
    for filename in os.listdir(dataset_path):
        count += 1
        file_path = os.path.join(dataset_path, filename)
        points.append(models.PointStruct(id=count,
                                         vector=image_embedding(file_path),
                                         payload={"path": file_path}))
        if count % 1000 == 0:
            print(f"Done embedding {count} images.")
    
    client = QdrantClient(path="qdrant_db")
    client.create_collection(
        "MRAG-image",
        vectors_config=models.VectorParams(
            size=512,
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
        collection_name="MRAG-image",
        wait=True,
        points=points,
    )
    print(op_info)



def text_embedding(text: str) -> torch.Tensor:
    inputs = clip_processor(text=text, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        outputs = clip_model.text_model(**inputs)
    tokens = outputs.last_hidden_state.squeeze(0)    
    return torch.nn.functional.normalize(tokens, dim=-1)


def MaxSim(Eq: torch.Tensor, Ed: torch.Tensor) -> int:
    scores = torch.matmul(Eq, Ed.T)
    max_row = scores.max(dim=1).values
    return max_row.sum()


def best_match(Eq: torch.Tensor, list_Ed: dict) -> tuple[torch.Tensor, int]:
    ret_vec = None
    ret_score = float("-inf")
    for path, Ed in list_Ed.items():
        tmp_score = MaxSim(Eq, Ed)
        if tmp_score > ret_score:
            ret_score = tmp_score
            ret_vec = Ed
    return ret_vec, ret_score


if __name__ == "__main__":
    # dataset_path = "./image_corpus"
    # store_db(dataset_path)
    vectordb = VectorDB()
    query = "apples"
    print(vectordb.get_topk_similar(vectordb.vec_data['./image_corpus\\Biological_0_gt_an-apple-cut-in-half-and-has-oxidise-1.jpg']))
