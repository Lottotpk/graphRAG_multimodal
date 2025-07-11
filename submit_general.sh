#!/bin/bash
#SBATCH --job-name=mrag
#SBATCH --mail-user=tpipatpajong2@cse.cuhk.edu.hk
#SBATCH --mail-type=ALL
#SBATCH --output=/research/d7/fyp24/tpipatpajong2/graphRAG_multimodal/output_general.txt
#SBATCH --gres=gpu:1

bash MRAG-Bench/eval/models/run_model.sh