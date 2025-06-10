# graphRAG_multimodal
The repository about implementing graph-based Retrieval Augmented Generation (graphRAG), which extends to multimodal model (texts, images, videos).

## Results (MRAG Benchmark)
We benchmark the model by using the `gemini-2.0-flash` as a base model for the RAG pipeline. Please note that the sources code correspond to the benchmark contain
- `embedding.py` contains all the embeddings.
- `vectordb.py` contains vector database class implementation.
- `MRAG-Bench/*.jsonl` contains the outputs from LLM.
- `MRAG-Bench/eval/score_without_ai.py` contains script to get the accuracy of LLM's answer.
- `MRAG-Bench/eval/models/gemini_eval.py` contains script for the RAG pipeline.

Here are some results:

### Without RAG
```bash
~/graphRAG_multimodal/MRAG-Bench> python eval/score_without_ai.py --filename gemini_no_rag_results.jsonl
```
```
--------------Results--------------
Scope: 70.588%
Obstruction: 72.222%
Temporal: 77.181%
Deformation: 59.804%
Biological: 52.941%
Angle: 72.671%
Partial: 33.74%
Incomplete: 43.137%
Others: 65.0%
Total accuracy: 60.532%
-----------------------------------
```

### With RAG (Without the given retrieval from the dataset)
```bash
~/graphRAG_multimodal/MRAG-Bench> python eval/score_without_ai.py --filename gemini_selfrag_results.jsonl
```
```
--------------Results--------------
Scope: 74.51%
Obstruction: 71.296%
Temporal: 75.839%
Deformation: 61.765%
Biological: 51.961%
Angle: 78.571%
Partial: 79.268%
Incomplete: 32.353%
Others: 69.167%
Total accuracy: 69.919%
-----------------------------------
```