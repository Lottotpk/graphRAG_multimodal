from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image

import random
import torch
import os
import logging
import json
import shutil

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

def generate_description(images: list[str], prompt: str):
    # Process the generation in batch (given that the process is not OOM).
    messages = []
    
    for image in images:
        tmp = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image,
                    },
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


def batch_processing(images_dir: str, batch_size: int):
    # Process image in batch
    images_dir_list = os.listdir(images_dir)
    images_dir_list = [os.path.join(images_dir, path) for path in images_dir_list]

    prompt = """
        You are an expert visual analyst. Analyse the provided images and break them into a few specific keywords (2-3 words in total) in these eight aspects with a short reason.
        If there is nothing that the images can be represented in this category, leave that section blank and do not make up answer.
        The examples are provided in each of the category. You may or may not use the keyword from these examples.
        1. Emotion / Mood: Describes the overall atmosphere or emotional tone of the image, for example
            - Energetic, Calm, Joyful, Dramatic, Melancholy, Serene, ...
            - Inspirational, Mysterious, Whimsical, Intense, Playful, Rebellious, ...
        2. Purpose / Context: Specifies the intended use case or application scenario, for example
            - Gym poster, Living-room wall art, Kids' room decoration
            - Corporate slide cover, Startup pitch deck, Meditation app hero
            - Fashion magazine cover, Wedding invitation, Product packaging
            - Music festival poster, NFT artwork, Book cover
        3. Style / Visual Attributes: Highlights the artistic or design style of the work, for example
            - Minimalist, Vintage, Futuristic, Pop-art, Emo, Grunge, Cyberpunk
            - Vaporwave, Surrealism, Bauhaus, Brutalist, Kawaii, Hand-drawn/Doodle
            - Advertising / Commercial ad, Corporate bland, Geometric, Dynamic motion
        4. Medium / Material: Indicates the creative medium or material representation, for example
            - Photography, Oil painting, Watercolor, Digital 3D render
            - Sketch, Collage, Graffiti, Infographic, Mixed media
        5. Color / Lightning Characteristics: Focuses on color atmosphere and lighting style, for example
            - Vivid, Bright, Pastel, Monochrome, Neon
            - Warm tones, Cold tones, High-contrast, Low-contrast
            - Dramatic lighting, Soft lighting, Golden hour, Chiaroscuro
        6. Cultural / Regional Elements: Incorporates cultural or regional aesthetics, for example
            - Japanese, Chinese, Nordic, Mediterranean, Middle Eastern
            - Traditional, Modernized, Indigenous, Folk-art
            - Festival-specific: Christmas, Diwali, Lunar New Year
        7. Target Audience / Perception: Tailors design to different demographic or user groups, for example
            - Kids, Teenagers, Adults, Seniors
            - Professional, Casual, Gamer, Traveler
            - Feminine, Masculine, Neutral
        8. Narrative / Symbolic Elements: Expresses abstract concepts or metaphorical meaning, for example
            - Symbolic: Power, Freedom, Love, Chaos, Harmony
            - Metaphorical: Rising sun for hope, A maze for confusion
            - Abstract shapes, Fractals, Mandalas
        Your answer must be in JSON format with key as string data type, and value as list of string (keywords) and reasons in pair, as the following JSON sample:
        ```
        {
            'Emotion / Mood' : [[ANSWER, REASON]],
            'Purpose / Context': [[ANSWER, REASON]],
            'Style / Visual Attributes': [[ANSWER, REASON]],
            'Medium / Material': [[ANSWER, REASON]],
            'Color / Lightning Characteristics': [[ANSWER, REASON]],
            'Cultural / Regional Elements': [[ANSWER, REASON]],
            'Target Audience / Perception': [[ANSWER, REASON]],
            'Narrative / Symbolic Elements': [[ANSWER, REASON]]
        }
        ```
    """
    text_output = []

    # TEST_RANDOM = random.sample(range(len(images_dir_list)), 30)
    # print(f"Random index list: {TEST_RANDOM}")
    for i in range(0, len(images_dir_list), batch_size):
        # if i not in TEST_RANDOM:
            # continue
        images_path = images_dir_list[i: i+batch_size]
        batch_output = generate_description(images_path, prompt)
        for output in batch_output:
            logging.info(f"Entry #{i+1}:")
            logging.info(f"Image path: {images_path[0]}")
            logging.info(output)
            text_output.append(output)
    
    return text_output

def fixing_answer(images_dir: str, batch_size: int, answer: any):
    # Process image in batch
    images_dir_list = os.listdir(images_dir)
    images_dir_list = [os.path.join(images_dir, path) for path in images_dir_list]

    prompt_1 = """
        You are an expert visual analyst. Analyse the provided images along with the keywords and a reason behind in json format. Fix or erase the unrealistic or exaggerated reason, and delete all the reasons.
        If there are tags that are used repeatedly, please revise wisely before reassigning those tags.
        For example, if the image is the flowers, 'Calm' and 'Photography' can be the tags, but 'gym poster' is not since flowers are not related to the gym. If objects or thoughts in the reason are not appeared in the picture, fix or remove the tag.
        If you agree or assign 'General' or 'None', leave it blank without makeup answers.
        Here is the previous answer of this image: 
    """

    prompt_2 = """
        Your answer must be in JSON format with key as string data type, and value as list of string (keywords), as the following JSON sample:
        ```
        {
            'Emotion / Mood' : [ANSWER HERE],
            'Purpose / Context': [ANSWER HERE],
            'Style / Visual Attributes': [ANSWER HERE],
            'Medium / Material': [ANSWER HERE],
            'Color / Lightning Characteristics': [ANSWER HERE],
            'Cultural / Regional Elements': [ANSWER HERE],
            'Target Audience / Perception': [ANSWER HERE],
            'Narrative / Symbolic Elements': [ANSWER HERE]
        }
        ```
    """
    text_output = []
    
    for i in range(0, len(images_dir_list), batch_size):
        # if i not in TEST_RANDOM:
            # continue
        images_path = images_dir_list[i: i+batch_size]
        batch_output = generate_description(images_path, prompt_1 + json.dumps(answer[i]) + prompt_2)
        for output in batch_output:
            logging.info(f"Entry #{i+1}:")
            logging.info(f"Image path: {images_path[0]}")
            logging.info(output)
            text_output.append(output)
    
    return text_output

def create_dir(images_dir: str, num_entries: int, dir_name: str):
    images_dir_list = os.listdir(images_dir)
    images_dir_list = [os.path.join(images_dir, path) for path in images_dir_list]
    TEST_RANDOM = random.sample(range(len(images_dir_list)), num_entries)
    os.mkdir(dir_name)
    path = "./" + dir_name
    for i in TEST_RANDOM:
        shutil.copy(images_dir_list[i], path)
    return path

def delete_dir(dir_name: str):
    shutil.rmtree(dir_name)


def main():
    images_dir = "./image_corpus"
    tmp_dir = "test_imgs"
    # if os.path.isdir(tmp_dir):
    #     delete_dir(tmp_dir)
    # tmp_dir = create_dir(images_dir, 30, tmp_dir)
    output_list = batch_processing(tmp_dir, 1) # Always use 1 for 7B model due to context limits
    json_1 = "./retrieve_images/json_output/reason_before_fixing2.jsonl"
    json_2 = "./retrieve_images/json_output/result_after_fixing2.jsonl"

    open(json_1, 'w').close()
    for output in output_list:
        output = output.strip("```json")
        with open(json_1, "a") as f:
            json.dump(json.loads(output), f)
            f.write("\n")
    
    output_list = fixing_answer(tmp_dir, 1, output_list)

    open(json_2, 'w').close()
    for output in output_list:
        output = output.strip("```json")
        with open(json_2, "a") as f:
            json.dump(json.loads(output), f)
            f.write("\n")


if __name__ == "__main__":
    main()