import json
import os
import sys
import numpy as np

import fakeredis


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))


def test_cache_set_and_get_similar(monkeypatch):
    # patch redis.from_url used by RedisCache to return a fakeredis instance
    fake_redis = fakeredis.FakeRedis()

    monkeypatch.setattr('redis.from_url', lambda url: fake_redis)

    from cache.redis_cache import RedisCache

    cache = RedisCache()
    # create a normalized vector
    v = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    key = cache.set(v, 'answer1', {'id': 'doc-1'})
    assert key.startswith('cache:')

    found = cache.get_similar(v, threshold=0.9)
    assert found is not None
    assert found['answer'] == 'answer1'
    assert found['meta']['id'] == 'doc-1'
