import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
from PIL import Image
import requests
import copy
import torch
import sys
import warnings
import math
import time

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.dataloader import bench_data_loader 

sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))
from VideoColBERT.InternVideo import benchmark_chat, image_embedding
from VideoColBERT.utils import create_vectordb

def eval_model(args):
    # Output file
    ans_file = open(args.answers_file, "a")

    count = 0
    for item in bench_data_loader(args, image_placeholder="<image>"):
        count += 1
        # qs = item['question']
        # qs_img = item['image_files']

        # Add the image into the query
        outputs = benchmark_chat(item['question'], item['image_files'][0], item['scenario'] == 'Incomplete')
        print(f"AI Answer: {outputs}, Actual Answer: {item['gt_choice']}: {item['answer']}")

        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({
                                   "qs_id": item['id'],
                                   "prompt": item['prompt'],
                                   "output": outputs,
                                   "gt_answer": item['answer'],
                                   "shortuuid": ans_id,
                                   "model_id": 'gemini-2.0-flash',
                                   "gt_choice": item['gt_choice'],
                                   "scenario": item['scenario'],
                                   "aspect": item['aspect'],
                                   }) + "\n")
        ans_file.flush()
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--extra-prompt", type=str, default="")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--test_size", type=int, default=10000000)
    ############# added for mrag benchmark ####################
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--use_rag", type=lambda x: x.lower() == 'true', default=False, help="Use RAG")
    parser.add_argument("--use_retrieved_examples", type=lambda x: x.lower() == 'true', default=False, help="Use retrieved examples")
    parser.add_argument("--self_rag", type=lambda x: x.lower() == 'true', default=False, help="Use your own retrieval")

    args = parser.parse_args()

    # create_vectordb("image_corpus/", image_embedding, "InternVideo", 1024)
    eval_model(args)
