"""
Clean the json output for usability in the future.
"""

import os
import json
import shutil

def combine_abstract(file_path: str):
    with open(file_path, 'r') as f:
        data = json.load(f)
    records = data["records"]

    tmp = {}
    for output in records:
        if output["image_path"] not in tmp:
            tmp[output["image_path"]] = {
                "image_path": output["image_path"],
                "description": output["description"],
                "error": output["error"],
            }
        else:
            tmp[output["image_path"]]["description"].update(output["description"])
    
    data["records"] = []
    for _, value in tmp.items():
        data["records"].append(value)
    
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)

def benchmark_to_ds(file_path: str):
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    ds_path = "/research/d7/fyp24/tpipatpajong2/graphRAG_multimodal/dataset/" + data["dataset_name"] + "/"
    os.makedirs(ds_path, exist_ok=True)

    itr = 1
    for qa in data["qa_pairs"]:
        for img_name in qa["image_ids"]:
            shutil.copy2(os.path.join("/research/d7/fyp24/tpipatpajong2/graphRAG_multimodal/dataset/stanford40", img_name + ".jpg"), ds_path)
        if itr % 100 == 0:
            print(f"Finished copying {itr} images")
        itr += 1

    return ds_path

if __name__ == "__main__":
    # combine_abstract("./generated_img_description/2025-10-07_13-49-42_desc.json")
    benchmark_to_ds("Stanford40Action_ImageLabelDescripion10template5.json")