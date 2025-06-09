import os

from plaidrepro.colbert.infra import Run, RunConfig, ColBERTConfig
from plaidrepro.colbert.data import Queries, Collection
from plaidrepro.colbert import Indexer, Searcher

if __name__ == "__main__":
    collection = os.getenv("COLLECTION_PATH")
    collection = Collection(path=collection)