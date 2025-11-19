import time
import json
from typing import Optional, Dict, Any

import numpy as np
import redis

from ..core.config import settings
from ..core.logger import get_logger


logger = get_logger('cache')


class RedisCache:
    """Simple Redis-backed cache storing embedding + answer + meta.

    Notes:
    - Uses HSET per key with fields: emb (json list), answer (str), meta (json dict)
    - get_similar performs a linear scan using SCAN_ITER to avoid blocking Redis.
    """

    def __init__(self):
        try:
            self.r = redis.from_url(settings.REDIS_URL)
        except Exception:
            logger.exception('Failed to connect to Redis at %s', settings.REDIS_URL)
            raise

    def _make_key(self, prefix: str = 'cache') -> str:
        return f'{prefix}:{int(time.time() * 1000)}'

    def get_similar(self, q_emb: np.ndarray, threshold: float = 0.82) -> Optional[Dict[str, Any]]:
        """Return best similar cached answer if similarity >= threshold.

        Embeddings are assumed to be normalized unit vectors; similarity is dot product.
        """
        try:
            best = None
            best_sim = -1.0
            for key in self.r.scan_iter(match='cache:*'):
                try:
                    data = self.r.hgetall(key)
                    emb_b = data.get(b'emb')
                    if not emb_b:
                        continue
                    emb_list = json.loads(emb_b.decode('utf-8'))
                    emb = np.array(emb_list, dtype=np.float32)
                    if q_emb.shape != emb.shape:
                        # shape mismatch -> skip
                        continue
                    sim = float(np.dot(q_emb, emb.T))
                except Exception:
                    # individual key read failure shouldn't break the scan
                    logger.debug('Failed to evaluate cache key %s', key, exc_info=True)
                    continue
                if sim > best_sim:
                    best_sim = sim
                    best = (key.decode('utf-8') if isinstance(key, bytes) else str(key), data)
            if best is None:
                return None
            if best_sim >= threshold:
                k, data = best
                answer_b = data.get(b'answer', b'')
                meta_b = data.get(b'meta', b'{}')
                try:
                    answer = answer_b.decode('utf-8') if isinstance(answer_b, (bytes, bytearray)) else str(answer_b)
                except Exception:
                    answer = str(answer_b)
                try:
                    meta = json.loads(meta_b.decode('utf-8')) if isinstance(meta_b, (bytes, bytearray)) else json.loads(str(meta_b))
                except Exception:
                    meta = {}
                return {'key': k, 'answer': answer, 'meta': meta, 'sim': best_sim}
            return None
        except Exception:
            logger.exception('Error scanning cache')
            return None

    def set(self, q_emb: np.ndarray, answer: str, meta: Dict[str, Any]) -> str:
        """Store embedding, answer, and meta in Redis and set TTL.

        Returns the created key.
        """
        key = self._make_key()
        try:
            emb_list = None
            try:
                emb_list = list(np.asarray(q_emb).astype(float).tolist())
            except Exception:
                emb_list = list(map(float, q_emb))
            mapping = {
                'answer': answer,
                'emb': json.dumps(emb_list),
                'meta': json.dumps(meta or {})
            }
            self.r.hset(key, mapping=mapping)
            try:
                self.r.expire(key, int(settings.CACHE_TTL_SECONDS))
            except Exception:
                # not critical
                logger.debug('Could not set expire on key %s', key, exc_info=True)
            return key
        except Exception:
            logger.exception('Failed to set cache key')
            raise