import numpy as np

from flatball.data.cache import FeatureCache
from flatball.data.dataset import DatasetSplit


def test_cache_roundtrip(tmp_path):
    split = DatasetSplit(
        name="train",
        features=np.arange(2 * 4 * 6, dtype=np.float32).reshape(2, 4, 6),
        labels=np.array([[0, 0, 1, 1], [0, 1, 0, 1]], dtype=np.uint8),
        meta=({"anomalous": True, "fraction": 0.1}, {"anomalous": False, "fraction": 0.0}),
    )
    cache = FeatureCache(tmp_path, "abcdef0123456789")
    assert cache.load("R5_r0.5_mu+0.1", 0, "train") is None
    cache.save("R5_r0.5_mu+0.1", 0, split)
    loaded = cache.load("R5_r0.5_mu+0.1", 0, "train")
    assert loaded is not None
    np.testing.assert_array_equal(loaded.features, split.features)
    np.testing.assert_array_equal(loaded.labels, split.labels)
    assert loaded.meta == split.meta
