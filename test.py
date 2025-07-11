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

sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))
from VideoColBERT.InternVideo import benchmark_chat, image_embedding
from VideoColBERT.utils import create_vectordb

create_vectordb("image_corpus/", image_embedding, "InternVideo", 1024)