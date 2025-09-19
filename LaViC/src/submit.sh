#!/bin/bash
#SBATCH --job-name=lavic
#SBATCH --mail-user=tpipatpajong2@cse.cuhk.edu.hk
#SBATCH --mail-type=ALL
#SBATCH --output=/research/d7/fyp24/tpipatpajong2/graphRAG_multimodal/LaViC/output.txt
#SBATCH --gres=gpu:1
#SBATCH --reser=jcheng_gpu_301
#SBATCH --qos=gpu
#SBATCH --account=gpu
#SBATCH -c 4
#SBATCH -p gpu_24h

# pip install transformers==4.46.2
# python knowledge_distillation.py \
#   --model_name llava-hf/llava-v1.6-mistral-7b-hf \
#   --train_data ../data/item2meta_train.json \
#   --val_data ../data/item2meta_valid.jsonl \
#   --train_images_dir ../data/train_images \
#   --val_images_dir ../data/valid_images \
#   --output_dir ./out_distilled \
#   --num_workers 2 \
#   --lr 5e-5 --weight_decay 1e-5 --num_epochs 3 --batch_size 1

# python qwen_embed.py
# python crawl_images.py

# python prompt_tuning.py \
#   --model_dir ./out_distilled/vision_lora_adapter_best \
#   --candidate_type candidates_st \
#   --finetune_output_dir ./out_finetuned \
#   --max_length 2048 \
#   --batch_size 1 \
#   --lr 5e-5 --weight_decay 1e-5 \
#   --num_epochs 1 \
#   --item_meta_path ../data/item2meta_train.json \
#   --image_dir ../data/train_images \
#   --category all_beauty
# mv ./out_finetuned/test_results_candidates_st.jsonl ../data/all_beauty/test_results_candidates_st2.jsonl

# python prompt_tuning.py \
#   --model_dir ./out_distilled/vision_lora_adapter_best \
#   --candidate_type candidates_st \
#   --finetune_output_dir ./out_finetuned \
#   --max_length 2048 \
#   --batch_size 1 \
#   --lr 5e-5 --weight_decay 1e-5 \
#   --num_epochs 1 \
#   --item_meta_path ../data/item2meta_train.json \
#   --image_dir ../data/train_images \
#   --category amazon_fashion
# mv ./out_finetuned/test_results_candidates_st.jsonl ../data/amazon_fashion/test_results_candidates_st2.jsonl

# python prompt_tuning.py \
#   --model_dir ./out_distilled/vision_lora_adapter_best \
#   --candidate_type candidates_st \
#   --finetune_output_dir ./out_finetuned \
#   --max_length 2048 \
#   --batch_size 1 \
#   --lr 5e-5 --weight_decay 1e-5 \
#   --num_epochs 1 \
#   --item_meta_path ../data/item2meta_train.json \
#   --image_dir ../data/train_images \
#   --category amazon_home
# mv ./out_finetuned/test_results_candidates_st.jsonl ../data/amazon_home/test_results_candidates_st2.jsonl

# python prompt_tuning.py \
#   --model_dir ./out_distilled/vision_lora_adapter_best \
#   --candidate_type candidates_gpt_large \
#   --finetune_output_dir ./out_finetuned \
#   --max_length 2048 \
#   --batch_size 1 \
#   --lr 5e-5 --weight_decay 1e-5 \
#   --num_epochs 1 \
#   --item_meta_path ../data/item2meta_train.json \
#   --image_dir ../data/train_images \
#   --category all_beauty
# mv ./out_finetuned/test_results_candidates_gpt_large.jsonl ../data/all_beauty/test_results_candidates_gpt_large2.jsonl

# python prompt_tuning.py \
#   --model_dir ./out_distilled/vision_lora_adapter_best \
#   --candidate_type candidates_gpt_large \
#   --finetune_output_dir ./out_finetuned \
#   --max_length 2048 \
#   --batch_size 1 \
#   --lr 5e-5 --weight_decay 1e-5 \
#   --num_epochs 1 \
#   --item_meta_path ../data/item2meta_train.json \
#   --image_dir ../data/train_images \
#   --category amazon_fashion
# mv ./out_finetuned/test_results_candidates_gpt_large.jsonl ../data/amazon_fashion/test_results_candidates_gpt_large2.jsonl

# python prompt_tuning.py \
#   --model_dir ./out_distilled/vision_lora_adapter_best \
#   --candidate_type candidates_gpt_large \
#   --finetune_output_dir ./out_finetuned \
#   --max_length 2048 \
#   --batch_size 1 \
#   --lr 5e-5 --weight_decay 1e-5 \
#   --num_epochs 1 \
#   --item_meta_path ../data/item2meta_train.json \
#   --image_dir ../data/train_images \
#   --category amazon_home
# mv ./out_finetuned/test_results_candidates_gpt_large.jsonl ../data/amazon_home/test_results_candidates_gpt_large2.jsonl

python prompt_tuning.py \
  --model_dir ./out_distilled/vision_lora_adapter_best \
  --candidate_type candidates_qwen \
  --finetune_output_dir ./out_finetuned \
  --max_length 2048 \
  --batch_size 1 \
  --lr 5e-5 --weight_decay 1e-5 \
  --num_epochs 1 \
  --item_meta_path ../data/item2meta_train.json \
  --image_dir ../data/train_images \
  --category all_beauty
mv ./out_finetuned/test_results_candidates_qwen.jsonl ../data/all_beauty/test_results_candidates_qwen2.jsonl

python prompt_tuning.py \
  --model_dir ./out_distilled/vision_lora_adapter_best \
  --candidate_type candidates_qwen \
  --finetune_output_dir ./out_finetuned \
  --max_length 2048 \
  --batch_size 1 \
  --lr 5e-5 --weight_decay 1e-5 \
  --num_epochs 1 \
  --item_meta_path ../data/item2meta_train.json \
  --image_dir ../data/train_images \
  --category amazon_fashion
mv ./out_finetuned/test_results_candidates_qwen.jsonl ../data/amazon_fashion/test_results_candidates_qwen2.jsonl

python prompt_tuning.py \
  --model_dir ./out_distilled/vision_lora_adapter_best \
  --candidate_type candidates_qwen \
  --finetune_output_dir ./out_finetuned \
  --max_length 2048 \
  --batch_size 1 \
  --lr 5e-5 --weight_decay 1e-5 \
  --num_epochs 1 \
  --item_meta_path ../data/item2meta_train.json \
  --image_dir ../data/train_images \
  --category amazon_home
mv ./out_finetuned/test_results_candidates_qwen.jsonl ../data/amazon_home/test_results_candidates_qwen2.jsonl