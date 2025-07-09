#!/bin/bash
#SBATCH --job-name=mrag
#SBATCH --mail-user=tpipatpajong2@cse.cuhk.edu.hk
#SBATCH --mail-type=ALL
#SBATCH --output=/research/d7/fyp24/tpipatpajong2/graphRAG_multimodal/output_general.txt
#SBATCH --gres=gpu:1

export HF_HOME=/research/d7/fyp24/tpipatpajong2/.cache/huggingface
export PATH=$PATH:/usr/local/cuda-12.5/bin
export LD_LIBRARY_PATH=/usr/local/cuda-12.5/lib64
export PIP_CACHE_DIR=/research/d7/fyp24/tpipatpajong2/.cache/pip
source /research/d7/fyp24/tpipatpajong2/miniconda3/etc/profile.d/conda.sh
conda activate fyp
bash MRAG-Bench/eval/models/run_model.sh