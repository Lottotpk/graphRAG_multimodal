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
mv ./out_finetuned/test_results_candidates_qwen.jsonl ../data/all_beauty/test_results_candidates_qwen.jsonl