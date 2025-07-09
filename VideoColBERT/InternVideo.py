# pip install transformers==4.40.1
import numpy as np
import torch
import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
# import utils
from VideoColBERT import utils

import warnings
warnings.filterwarnings("ignore")

# model setting
model_path = 'OpenGVLab/InternVideo2_5_Chat_8B'

print("test")
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModel.from_pretrained(model_path, trust_remote_code=True).half().cuda().to(torch.bfloat16)
model.language_model.config.output_hidden_states = True
print("Test")

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img), T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC), T.ToTensor(), T.Normalize(mean=MEAN, std=STD)])
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=6, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set((i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = ((i % (target_width // image_size)) * image_size, (i // (target_width // image_size)) * image_size, ((i % (target_width // image_size)) + 1) * image_size, ((i // (target_width // image_size)) + 1) * image_size)
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def load_image(image, input_size=448, max_num=6):
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


def get_index(bound, fps, max_frame, first_idx=0, num_segments=32):
    if bound:
        start, end = bound[0], bound[1]
    else:
        start, end = -100000, 100000
    start_idx = max(first_idx, round(start * fps))
    end_idx = min(round(end * fps), max_frame)
    seg_size = float(end_idx - start_idx) / num_segments
    frame_indices = np.array([int(start_idx + (seg_size / 2) + np.round(seg_size * idx)) for idx in range(num_segments)])
    return frame_indices

def get_num_frames_by_duration(duration):
        local_num_frames = 4        
        num_segments = int(duration // local_num_frames)
        if num_segments == 0:
            num_frames = local_num_frames
        else:
            num_frames = local_num_frames * num_segments
        
        num_frames = min(512, num_frames)
        num_frames = max(128, num_frames)

        return num_frames

def load_video(video_path, bound=None, input_size=448, max_num=1, num_segments=32, get_frame_by_duration = False):
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    max_frame = len(vr) - 1
    fps = float(vr.get_avg_fps())

    pixel_values_list, num_patches_list = [], []
    transform = build_transform(input_size=input_size)
    if get_frame_by_duration:
        duration = max_frame / fps
        num_segments = get_num_frames_by_duration(duration)
    frame_indices = get_index(bound, fps, max_frame, first_idx=0, num_segments=num_segments)
    for frame_index in frame_indices:
        img = Image.fromarray(vr[frame_index].asnumpy()).convert("RGB")
        img = dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(tile) for tile in img]
        pixel_values = torch.stack(pixel_values)
        num_patches_list.append(pixel_values.shape[0])
        pixel_values_list.append(pixel_values)
    pixel_values = torch.cat(pixel_values_list)
    return pixel_values, num_patches_list


def video_embedding(video_path):
    pixel_values, num_patches_list = load_video(video_path, num_segments=128)
    pixel_values = pixel_values.to(torch.bfloat16).to(model.device)
    with torch.no_grad():
        outputs = model.vision_model(pixel_values)
        return outputs.pooler_output


def image_embedding(image_path, loaded: bool = False):
    img = image_path
    if not loaded:
        img = Image.open(image_path).convert("RGB")
    pixel_values = load_image(img)
    pixel_values = pixel_values.to(torch.bfloat16).to(model.device)
    with torch.no_grad():
        outputs = model.vision_model(pixel_values)
        return outputs.pooler_output


# Not a good embedding, it's best to not use this
def text_embedding(text: str):
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        outputs =  model.language_model(**inputs)
        embed = outputs.hidden_states[-1][:, 0, :]
        text_proj = torch.nn.Linear(4096, 1024, bias=False).to(dtype=torch.bfloat16, device=model.device)
        return text_proj(embed)


def query(msg: str):
    # Text embedding + Retrieval
    text_vector = text_embedding(msg)
    retrieved = utils.retrieval(text_vector, "InternVideo", 1, True)

    # Chat
    generation_config = dict(
        do_sample=False,
        temperature=0.0,
        max_new_tokens=1024,
        top_p=0.1,
        num_beams=1
    )
    with torch.no_grad():
        pixel_values, num_patches_list = load_video(retrieved[0].payload['path'], num_segments=128, max_num=1, get_frame_by_duration=False)
        pixel_values = pixel_values.to(torch.bfloat16).to(model.device)
        video_prefix = "".join([f"Frame{i+1}: <image>\n" for i in range(len(num_patches_list))])
        
        # single-turn conversation
        question = video_prefix + msg
        output1, chat_history = model.chat(tokenizer, pixel_values, question, generation_config, num_patches_list=num_patches_list, history=None, return_history=True)
        print(output1)

        # multi-turn conversation
        # question2 = "How many people appear in the video?"
        # output2, chat_history = model.chat(tokenizer, pixel_values, question, generation_config, num_patches_list=num_patches_list, history=chat_history, return_history=True)

        # print(output2)


def benchmark_chat(msg: str, image_query, incomplete) -> str:
    # Text embedding + Retrieval
    img_vector = image_embedding(image_query, True)
    retrieved = None
    if incomplete:
        retrieved = utils.retrieval(img_vector, "InternVideo", 4, False)
    else:
        retrieved = utils.retrieval(img_vector, "InternVideo", 8, False)

    # Chat
    generation_config = dict(
        do_sample=False,
        temperature=0.0,
        max_new_tokens=1024,
        top_p=0.1,
        num_beams=1
    )
    with torch.no_grad():
        pixel_values, num_patches_list = [], []
        for i in range(len(retrieved)):
            img = Image.open(retrieved[i].payload['path']).convert("RGB")
            pv = load_image(img, max_num=1)
            pixel_values.append(pv)
            num_patches_list.append(pv.shape[0])
        pixel_values = torch.cat(pixel_values).to(torch.bfloat16).to(model.device)
        # print(msg)
        # print(pixel_values.shape)
        # print(num_patches_list)
        
        # single-turn conversation
        question = msg
        output1 = model.chat(tokenizer, pixel_values, question, generation_config, num_patches_list=num_patches_list, history=None, return_history=False)
        return output1


if __name__ == "__main__":
    # query("Do you know what these 3 people are doing?") # should retrive example_video/bigbang.mp4
    # video_path = "example_video/bigbang.mp4"
    # embedded = video_embedding(video_path)
    # image_path = "image_corpus/Biological_0_gt_77fa0cfd88f3bb00ed23789a476f0acd---d.jpg"
    # embedded = image_embedding(image_path)
    # embedded, num_patches_list = load_video("example_video/bigbang.mp4", num_segments=128, max_num=1, get_frame_by_duration=False)
    # print(num_patches_list)
    img = Image.open("image_corpus/Biological_1_gt_1.jpg").convert("RGB")
    embedded = load_image(img, max_num=1)
    print(embedded.shape)
    # utils.create_vectordb("example_video/", video_embedding, "InternVideo", 1024)