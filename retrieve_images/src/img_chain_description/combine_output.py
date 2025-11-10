import os
import sys
import json
import argparse
import logging
from utils.logging_config import setup_logger
from datetime import datetime
from config import get_description_path, get_description_filename

setup_logger()
logger = logging.getLogger(__name__)
GENERATED_DIR = "./generated_all/"


def main(args):
    payload = {
        'created_at': datetime.now().isoformat(),
        'records': None
    }
    id = 0
    records = []
    json_files = os.listdir(GENERATED_DIR)[-(args.prev_size):]
    for file in json_files:
        with open(os.path.join(GENERATED_DIR, file), 'r') as f:
            dataset = json.load(f)
            for rec in dataset["records"]:
                rec['id'] = id
                id += 1
                records.append(rec)
    
    payload["records"] = records
    out_path = get_description_path(get_description_filename('all'), 'all')
    with open(out_path, 'w') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    logger.info(f'Combined descriptions saved to {out_path}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="combining output from generated description")
    parser.add_argument("--prev_size", type=int, required=True, help='number of latest files to combine')
    args = parser.parse_args()
    main(args)