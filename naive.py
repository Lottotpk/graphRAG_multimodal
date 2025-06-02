import os
import cv2
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"

def frame_sampling(video_name : str, freq : int = 60) -> int:
    cap = cv2.VideoCapture(f"./example_video/{video_name}.mp4")
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
        cv2.imwrite(f"./sampled_frames/{video_name}_{current_frame // 60}.jpg", frame)

    cap.release()
    cv2.destroyAllWindows()
    return current_frame + 1

def video_patch_embedding(filename : str, title : str, total_frames : int) -> torch.Tensor:
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    embedded = []
    for i in range(total_frames):    
        # desc = f"Scene {i+1} from {title}."
        image = Image.open(f"./sampled_frames/{filename}_{i}.jpg")

        # inputs = processor(text=desc, images=image, return_tensors="pt", padding=True)
        inputs = processor(images=image, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            outputs = model.vision_model(**inputs)
        tokens = outputs.last_hidden_state.squeeze(0)
        print(tokens)
        print(torch.nn.functional.normalize(tokens, dim=-1))
        embedded.append(torch.nn.functional.normalize(tokens, dim=-1))
    return torch.cat(embedded, dim=0)

if __name__ == "__main__":
    video_patch_embedding("bigbang", "the big bang theory TV show", 3)