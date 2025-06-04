import os
import cv2
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def frame_sampling(video_name: str, freq: int = 60) -> int:
    cap = cv2.VideoCapture(f"./../example_video/{video_name}.mp4")
    if not os.path.exists("sampled_frames"):
        os.makedirs("sampled_frames")

    current_frame = -1
    while cap.isOpened():
        ret, frame = cap.read()
        current_frame += 1
        if ret == False:
            break
        if current_frame % freq != 0:
            continue
        cv2.imwrite(f"./../sampled_frames/{video_name}_{current_frame // 60}.jpg", frame)

    cap.release()
    cv2.destroyAllWindows()
    return current_frame + 1


def video_patch_embedding(filename: str, title: str, total_frames: int) -> torch.Tensor:
    embedded = []
    for i in range(total_frames):    
        image = Image.open(f"./../sampled_frames/{filename}_{i}.jpg")
        desc = f"Scene {i} from {title}"

        inputs = clip_processor(text=desc, images=image, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            outputs = clip_model(**inputs)
        tokens = outputs.vision_model_output.last_hidden_state.squeeze(0)
        visual_proj = clip_model.visual_projection(tokens)
        embedded.append(torch.nn.functional.normalize(visual_proj, dim=-1))
    return torch.cat(embedded, dim=0)


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


def best_match(Eq: torch.Tensor, list_Ed: list) -> tuple[torch.Tensor, int]:
    ret_vec = None
    ret_score = float("-inf")
    for Ed in list_Ed:
        tmp_score = MaxSim(Eq, Ed)
        if tmp_score > ret_score:
            ret_score = tmp_score
            ret_vec = Ed
    return ret_vec, ret_score


if __name__ == "__main__":
    document = []
    doc = video_patch_embedding("bigbang", "the big bang theory TV show", 3)
    document.append(doc)
    query_embedded = text_embedding("Describe the video about the scene in the big bang theory TV show")
    print(best_match(query_embedded, document))
    