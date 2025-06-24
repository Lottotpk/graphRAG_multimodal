# pip install transformers==4.46.2
import av
import torch
import torch.nn.functional as F
import numpy as np
from huggingface_hub import hf_hub_download
from transformers import LlavaNextVideoProcessor, LlavaNextVideoForConditionalGeneration, BitsAndBytesConfig, AutoModel
from utils import create_vectordb

model_id = "llava-hf/LLaVA-NeXT-Video-7B-hf"

model_llava = LlavaNextVideoForConditionalGeneration.from_pretrained(
    model_id, 
    torch_dtype=torch.float16, 
    low_cpu_mem_usage=True,
    quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16),
    attn_implementation="flash_attention_2",
)

model_nv = AutoModel.from_pretrained('nvidia/NV-Embed-v2', trust_remote_code=True)

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


def video_embedding(video_path):
    # define a chat history and use `apply_chat_template` to get correctly formatted prompt
    # Each value in "content" has to be a list of dicts with types ("text", "image", "video") 
    messages = [
        {

            "role": "user",
            "content": [
                {"type": "text", "text": "What is this video about?"},
                {"type": "video"},
                ],
        },
    ]

    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    container = av.open(video_path)

    # sample uniformly 8 frames from the video, can sample more for longer videos
    total_frames = container.streams.video[0].frames
    indices = np.arange(0, total_frames, total_frames / 16).astype(int)
    clip = read_video_pyav(container, indices)
    inputs = processor(text=prompt, videos=clip, padding=True, return_tensors="pt").to(model_llava.device)

    input_len = inputs['input_ids'].shape[-1]
    output = model_llava.generate(**inputs, max_new_tokens=100, do_sample=False)
    prompt_output = processor.decode(output[0][input_len:], skip_special_tokens=True)
    # print(prompt_output)

    embeddings = model_nv.encode([prompt_output], instruction="", max_length=32768)
    embeddings = F.normalize(embeddings, p=2, dim=1)

    return embeddings


if __name__ == "__main__":
    create_vectordb("example_video/",
                    video_embedding,
                    "/uac/y22/tpipatpajong2/qdrant_db",
                    "LVLM",
                    4096)