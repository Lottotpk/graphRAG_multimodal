# pip install transformers==4.46.2
import av
import torch
import torch.nn.functional as F
import numpy as np
from huggingface_hub import hf_hub_download
from transformers import LlavaNextVideoProcessor, LlavaNextVideoForConditionalGeneration, BitsAndBytesConfig, AutoModel
from PIL import Image
# import utils
from VideoColBERT import utils

import warnings
warnings.filterwarnings("ignore")

model_id = "llava-hf/LLaVA-NeXT-Video-7B-hf"

model_llava = LlavaNextVideoForConditionalGeneration.from_pretrained(
    model_id, 
    torch_dtype=torch.float16, 
    low_cpu_mem_usage=True,
    quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16),
    attn_implementation="flash_attention_2",
)

print("Loading NV-embed")
model_nv = AutoModel.from_pretrained('nvidia/NV-Embed-v2', trust_remote_code=True)
print("Finished loding NV-embed")

processor = LlavaNextVideoProcessor.from_pretrained(model_id)


def read_video_pyav(container, indices):
    '''
    Decode the video with PyAV decoder.
    Args:
        container (`av.container.input.InputContainer`): PyAV container.
        indices (`List[int]`): List of frame indices to decode.
    Returns:
        result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
    '''
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])


def text_embedding(text: str):
    embeddings = model_nv.encode([text], instruction="", max_length=32768)
    embeddings = F.normalize(embeddings, p=2, dim=1)
    return embeddings


def chat_llm_video(query: str, video_path: str):
    messages = [
        {

            "role": "user",
            "content": [
                {"type": "text", "text": query},
                {"type": "video"},
                ],
        },
    ]

    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    container = av.open(video_path)

    # sample uniformly 16 frames from the video, can sample more for longer videos
    total_frames = container.streams.video[0].frames
    indices = np.arange(0, total_frames, total_frames / 16).astype(int)
    clip = read_video_pyav(container, indices)
    inputs = processor(text=prompt, videos=clip, padding=True, return_tensors="pt").to(model_llava.device)

    input_len = inputs['input_ids'].shape[-1]
    output = model_llava.generate(**inputs, max_new_tokens=256, do_sample=False)
    prompt_output = processor.decode(output[0][input_len:], skip_special_tokens=True)
    print(f"input_len = {input_len}")
    # print(prompt_output)
    return prompt_output


def chat_llm_image(query: str, image_path: list[str], opened: bool = False):
    content = [{"type": "text", "text": query}]
    for _ in range(len(image_path)):
        content.append({"type": "image"})
    messages = [
        {

            "role": "user",
            "content": content,
        },
    ]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    image = []
    for ipath in image_path:
        if not opened:
            image.append(Image.open(ipath).convert("RGB").resize((224, 224)))
        else:
            image.append(ipath.resize((224, 224)))
    inputs = processor(text=prompt, images=image, padding=True, return_tensors="pt").to(model_llava.device)
    input_len = inputs['input_ids'].shape[-1]
    output = model_llava.generate(**inputs, max_new_tokens=256, do_sample=False)
    prompt_output = processor.decode(output[0][input_len:], skip_special_tokens=True)
    return prompt_output


def video_embedding(video_path: str):
    return text_embedding(chat_llm_video("Describe this video.", video_path))


def image_embedding(image_path: str, opened: bool = False):
    return text_embedding(chat_llm_image("Describe this image.", [image_path], opened))


def query(msg: str):
    text_vector = text_embedding(msg)
    retrieved = utils.retrieval(text_vector, "LVLM", 1, True)
    chat_llm_video(msg, retrieved[0].payload['path'])


def benchmark_chat(msg: str, image_query, incomplete):
    img_vector = image_embedding(image_query, True)
    retrieved = None
    if incomplete:
        retrieved = utils.retrieval(img_vector, "LVLM", 2, False)
    else:
        retrieved = utils.retrieval(img_vector, "LVLM", 3, False)
    retrieved_img = []
    for i in range(len(retrieved)):
        retrieved_img.append(retrieved[i].payload['path'])
    msg = msg.replace("<image>", "")
    return chat_llm_image(msg, retrieved_img)

# if __name__ == "__main__":
    # query("From the video consists of three people, give me the details on what are they doing?")
    # print(image_embedding("image_corpus/Biological_0_gt_77fa0cfd88f3bb00ed23789a476f0acd---d.jpg"))
    # embedded = image_embedding("image_corpus/Biological_1_gt_1.jpg")
    # print(embedded.shape)
    # chat_llm_video("Describe this video", "example_video/bigbang.mp4")
    # chat_llm_image("Describe these images separately.", ["image_corpus/Biological_0_gt_77fa0cfd88f3bb00ed23789a476f0acd---d.jpg", \
    #                                                         "image_corpus/Biological_0_gt_409E7C55-DDA5-442E-BEF368457F16CAA7.jpg", \
    #                                                         "image_corpus/Biological_1_gt_1.jpg", \
    #                                                         "image_corpus/Biological_1_input.png"])
    # create_vectordb("example_video/", video_embedding, "LVLM", 4096)