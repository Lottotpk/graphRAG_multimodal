from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image
from prompt import ABSTRACT_PROMPT, SYSTEM_PROMPT

import random
import torch
import os
import logging
import json
import argparse

model_name = "Qwen/Qwen2.5-VL-7B-Instruct"
processor = AutoProcessor.from_pretrained(model_name)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_name,
    dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="flash_attention_2",   
)
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

def generate_description(images: list[str], prompt: str, system: str):
    # Process the generation in batch (given that the process is not OOM).
    messages = []
    
    for image in images:
        tmp = [
            {"role": "system", "content": system},
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
                ],
            }
        ]
        messages.append(tmp)
    
    # Process input before inference
    texts = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) 
            for msg in messages]
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=texts,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=4096, temperature=1e-4)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text


def batch_processing(args):
    # Process image in batch
    images_dir_list = os.listdir(args.images_dir)
    images_dir_list = [os.path.join(args.images_dir, path) for path in images_dir_list]
 
    text_output = []

    if args.random_sample:
        TEST_RANDOM = random.sample(range(len(images_dir_list)), 30)
        logging.info(f"Random index list: {TEST_RANDOM}")
        images_dir_list = [images_dir_list[i] for i in TEST_RANDOM]
    
    for i in range(0, len(images_dir_list), args.batch_size):
        images_path = images_dir_list[i: i + args.batch_size]
        batch_output = []
        for j in range(0, 8, args.num_topic):
            batch_output.append(generate_description(images_path, ABSTRACT_PROMPT(range(j, min(j + args.num_topic, 8))), SYSTEM_PROMPT))
        for output in batch_output:
            logging.info(f"Entry #{i+1}:")
            logging.info(f"Image path: {images_path[0]}")
            logging.info(output[0])
            text_output.append(output[0])
    
    return text_output


def main():
    parser = argparse.ArgumentParser(description="Images abstract information extraction script")
    parser.add_argument("--images_dir", type=str)
    parser.add_argument("--output_file", type=str)
    parser.add_argument("--random_sample", type=bool, default=False) # do not add argument if you want to process everything
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_topic", type=int, default=4)
    args = parser.parse_args()

    output_list = batch_processing(args) # Always use 1 for 7B model due to context limits

    images_dir = os.listdir(args.images_dir)
    for i, output in enumerate(output_list):
        output = output.strip("```json")
        with open(args.output_file, "a") as f:
            entry = dict()
            entry["id"] = i
            entry["image_path"] = os.path.join(args.images_dir, images_dir[i])
            output = json.loads(output)

            for key, value in output.items():
                entry[key] = value
            json.dump(entry, f)
            f.write("\n")


if __name__ == "__main__":
    main()
    # clean_json()