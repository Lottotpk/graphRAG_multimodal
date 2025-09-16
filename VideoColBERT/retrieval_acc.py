import torch
import imagehash
from tqdm.auto import tqdm
from datasets import load_dataset
from typing import Callable
from PIL import Image
from transformers import AutoProcessor, AutoModel, AutoTokenizer \
    # , Siglip2VisionModel
# import utils
import logging
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from vectordb import VectorDB

# import InternVideo
# import LVLM

logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
device = "cuda" if torch.cuda.is_available() else "cpu"

# CHANGE THIS AND EMBEDDING METHOD FOR DIFFERENT MODEL
model_name = "openai/clip-vit-large-patch14"
# model_name = "google/siglip2-base-patch16-naflex"
model = AutoModel.from_pretrained(model_name).to(device)
# model = Siglip2VisionModel.from_pretrained(model_name).to(device)
processor = AutoProcessor.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

def img_open(img_path: str):
    image = Image.open(img_path)
    if image.format == 'PNG':
        # and is not RGBA
        if image.mode != 'RGBA':
            image.convert("RGBA")
        else:
            image.convert("RGB")
    else:
        image.convert("RGB")
    return image

def SigLIP_encoder(image_path, opened: bool = False):
    if not opened:
        image = Image.open(image_path).convert("RGB")
    else:
        image = image_path

    with torch.no_grad():
        inputs = processor.image_processor(images=[image], return_tensors="pt").to(device)
        vision_outputs = model(**inputs)
        img_embeddings = vision_outputs.last_hidden_state[0]
        img_embeddings = img_embeddings / img_embeddings.norm(dim=-1, keepdim=True)
        return img_embeddings

def CLIP_encoder(image_path: str, text_embed: bool = False, text: str = None, opened: bool = False):
    if not opened:
        image = img_open(image_path)
    else:
        image = image_path

    with torch.no_grad():
        if not text_embed:
            inputs = processor(images=image, return_tensors="pt", padding=True).to(device)
            # vision_outputs = model.vision_model(**inputs)
            embeddings = model.get_image_features(**inputs)
            # img_embeddings = vision_outputs.pooler_output
            # img_embeddings = model.visual_projection(img_embeddings)
            # img_embeddings = img_embeddings / img_embeddings.norm(dim=-1, keepdim=True)
        else:
            inputs = processor(text=[text], images=image, return_tensors="pt", padding=True).to(device)
            outputs = model(**inputs)
            img_embeddings = outputs.image_embeds
            text_embeddings = outputs.text_embeds
            embeddings = torch.stack((img_embeddings, text_embeddings))
        return embeddings


def eval(ds, encoder: Callable[[str, bool, str, bool], torch.Tensor], collection_name: str, example: bool = False):
    total_retrieved = 0
    total_gt = 0
    correct = 0
    correct_retrieve = 0
    count = 0
    sum_recall = 0

    for item in tqdm(ds):
        count += 1
        # if count == 500:
        #     break
        for i, img in enumerate(item['gt_images']):
            item['gt_images'][i] = imagehash.average_hash(img)
        
        if not example:
            #item['retrieved_images'] = utils.retrieval(encoder(item['image'], True),
            #                                           collection_name=collection_name,
            #                                           top_k=6)
            #item['retrieved_images'].pop(0)
            db = VectorDB("./database/" + collection_name + "/")
            for i in db.vec_data:
                db.vec_data[i] = db.vec_data[i].to(device)
            similarity, paths = db.get_topk_similar(encoder(item['image'], False, None, True), 6) # True, item['question'], True), 6)
            paths.pop(0)
            for i, path in enumerate(paths):
                # print(path)
                # im1 = Image.open(path)
                # im1 = im1.save(f"image_{i}.jpg")
                item['retrieved_images'][i] = Image.open(path).convert("RGB")
                item['retrieved_images'][i] = imagehash.average_hash(item['retrieved_images'][i])
        else:
            # logging.info(item['retrieved_images'])    
            for i, img in enumerate(item['retrieved_images']):
                if not example:
                    img_path = img.payload['path']
                else:
                    img_path = img
                if isinstance(img_path, str):
                    item['retrieved_images'][i] = Image.open(img_path).convert("RGB")
                item['retrieved_images'][i] = imagehash.average_hash(item['retrieved_images'][i])
        
        same_images = set(item['gt_images']) & set(item['retrieved_images'])
        #same_images = set()
        #for hashi in item['gt_images']:
        #    for hashj in item['retrieved_images']:
        #        if hashi - hashj < 5:
        #            same_images.add(hashi)

        logging.info(f"ID {item['id']}: correct {len(same_images)} images.")
        # total_retrieved += len(item['retrieved_images'])
        total_retrieved += 1
        # total_gt += len(item['gt_images'])
        total_gt += 1
        correct_retrieve += len(same_images)
        if len(same_images) > 0:
            correct += 1
        sum_recall += len(same_images) / 5
    aver_recall = sum_recall / total_retrieved
    return total_retrieved, total_gt, correct_retrieve, correct, aver_recall


def main():
    ds = load_dataset("uclanlp/MRAG-Bench", split="test")
    # total_retrieved, total_gt, correct = eval(ds, LVLM.image_embedding, "LVLM")
    total_retrieved, total_gt, correct_retrieve, correct, recall5 = eval(ds, CLIP_encoder, "CLIP")
    # total_retrieved, total_gt, correct = eval(ds, SigLIP_encoder, "SigLIP", False)
    
    print("Final results:")
    print(f"Total number of questions: {len(ds)}")
    print(f"Total ground truth images: {total_gt}")
    print(f"Total images retrieved: {total_retrieved}")
    print(f"Percentage: {(total_retrieved*100/total_gt):.2f}%")
    print(f"Number of correct retrieval: {correct_retrieve}")
    print(f"Retrieval Accuracy (At least one correct): {(correct*100/total_retrieved):.2f}%")
    print(f"Percentage Recall@5: {recall5*100:.2f}%")

def one_eval(n: int, encoder: Callable[[str, bool, str, bool], torch.Tensor], collection_name: str, example: bool = False):
    ds = load_dataset("uclanlp/MRAG-Bench", split="test")
    item = ds[n]
    for i, img in enumerate(item['gt_images']):
        item['gt_images'][i] = imagehash.average_hash(img)
    if not example:
        #item['retrieved_images'] = utils.retrieval(encoder(item['image'], True),
        #                                            collection_name=collection_name,
        #                                            top_k=6)
        #item['retrieved_images'].pop(0)
        db = VectorDB("./database/" + collection_name + "/")
        for i in db.vec_data:
            db.vec_data[i] = db.vec_data[i].to(device)
        similarity, paths = db.get_topk_similar(encoder(item['image'], False, None, True), 6)
        print(similarity[1:])
        paths.pop(0)
        for i, path in enumerate(paths):
            print(path)
            im1 = Image.open(path)
            im1 = im1.save(f"image_{i}.jpg")
            item['retrieved_images'][i] = img_open(path)
            item['retrieved_images'][i] = imagehash.average_hash(item['retrieved_images'][i])
    else:
        for i, img in enumerate(item['retrieved_images']):
            img_path = img.payload['path']
            print(f"image {i} shows {img_path}")
            im1 = Image.open(img_path)
            im1 = im1.save(f"image_{i}.jpg")
            if isinstance(img_path, str):
                item['retrieved_images'][i] = Image.open(img_path).convert("RGB")
            item['retrieved_images'][i] = imagehash.average_hash(item['retrieved_images'][i])
    same_images = set(item['gt_images']) & set(item['retrieved_images'])
    logging.info(f"ID {item['id']}: correct {len(same_images)} images.")
    print("Final result:")
    print(f"Total correct retrieval: {len(same_images)}")

def create_db(path_data: str, collection_name: str = None, batch_size: int = 16):
    # Create my own DB
    images = []
    img_names = []
    count = 0
    for img in os.listdir(path_data):
        count += 1
        img_names.append(os.path.join(path_data, img))
        if count % 1000 == 0:
            logging.info(f"{count} done.")
        img = os.path.join(path_data, img)
        images.append(Image.open(img).convert("RGB"))
    
    db = VectorDB("./database/" + collection_name + "/")
    vec_data = []
    for i in tqdm(range(0, len(images), batch_size)):
        inputs = processor(images=images[i:i+batch_size], return_tensors="pt", padding=True).to(device)
        images_features = model.get_image_features(**inputs).detach().cpu()
        db.add_vector(i, images_features, img_names[i])
        vec_data.append(images_features)
        logging.info(images_features.shape) # verify that the shape is correct
    logging.info(torch.cat(vec_data).shape)
    db.save_to_json()

if __name__ == "__main__":
    # utils.create_vectordb("image_corpus", SigLIP_encoder, "SigLIP", 768)
    # main()
    # one_eval(1, CLIP_encoder, "CLIP_1d")
    # embedding = CLIP_encoder("image_0.jpg", True)
    # print(embedding)
    # print(embedding.shape)

    # CLIP colbert (?)
    # image = Image.open("/research/d7/fyp24/tpipatpajong2/graphRAG_multimodal/image_corpus/Biological_4_gt_a-bad-batch-of-moldy-peached-at-a-fa.jpg")
    # inputs = processor(images=image, return_tensors="pt", padding=True).to(device)
    # vision_outputs = model.vision_model(**inputs)
    # embedded = vision_outputs.last_hidden_state[0]
    # embedded = embedded / embedded.norm(dim=-1, keepdim=True)
    # print(embedded)
    # print(embedded.shape)

    # SigLIP
    # embedded = SigLIP_encoder("/research/d7/fyp24/tpipatpajong2/graphRAG_multimodal/image_corpus/Biological_4_gt_a-bad-batch-of-moldy-peached-at-a-fa.jpg")
    # print(embedded)
    # print(embedded.shape)

    # create_db("image_corpus", "CLIP", 1)
    # db = VectorDB("./database/CLIP")
    # test_vec = db.get_vector(0)
    # test_meta = db.get_metadata(0)
    # print(test_vec)
    # print(test_vec.shape)
    # print(test_meta)
    # one_eval(0, CLIP_encoder, "CLIP")

    ds = load_dataset("uclanlp/MRAG-Bench", split="test")
    item = ds[0]
    print(type(item['gt_images'][0]))
    # query_img = CLIP_encoder(item['gt_images'][0], False, None, True)
    # db = VectorDB("./database/CLIP")
    # for i in db.vec_data:
    #     db.vec_data[i] = db.vec_data[i].to(device)
    # similarities, paths = db.get_topk_similar(query_img, 5)
    # print(similarities)
    # for i, path in enumerate(paths):
    #     im1 = Image.open(path)
    #     im1 = im1.save(f"image_{i}.jpg")
    
    # gt_img = [CLIP_encoder(item['gt_images'][i], False, None, True) for i in range(5)]
    # gt_scores = [torch.matmul(query_img, gt_img[i].T) for i in range(5)]
    # rt_img = [CLIP_encoder(f"image_{i}.jpg", False, None, False) for i in range(5)]
    # rt_scores = [torch.matmul(query_img, rt_img[i].T) for i in range(5)]
    # print(f"The ground truth scores are {gt_scores},")
    # print(f"while our retrieval scores are {rt_scores}.")

    # Ed = [CLIP_encoder(f"image_{i}.jpg", False, None, False).detach().cpu() for i in range(5)]
    # scores = [torch.matmul(Ed[i], Ed[i].T) for i in range(5)]
    # print(f"The same image has similarity score at {scores}")
