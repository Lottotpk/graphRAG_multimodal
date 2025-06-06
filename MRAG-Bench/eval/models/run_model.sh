# export HF_HOME="./cache"
# export HF_HUB_CACHE="./cache"
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES=5

python3 eval/models/gemini_eval.py \
    --answers-file "gemini_no_rag_results.jsonl" \
    --use_rag False \
    --use_retrieved_examples False 