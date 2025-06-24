# import cppyy
# import cppyy.ll
from usearch.index import Index, MetricKind, MetricSignature, CompiledMetric
from VideoColBERT.InternVideo import video_embedding
import numpy as np

if __name__ == "__main__":
    index = Index(
        ndim=1024,
        metric='cos',
        dtype='bf16',
        multi=True,
    )
    vector = video_embedding("example_video/bigbang.mp4").float().cpu().numpy()
    print(vector)
    print(vector.reshape(1, -1))
    print(vector.reshape(1, -1).shape)
    index.add([0], vector.reshape(1, -1))
