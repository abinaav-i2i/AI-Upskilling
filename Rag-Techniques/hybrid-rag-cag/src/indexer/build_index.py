"""Build FAISS index from a HF dataset or a local file. Run as module."""

import argparse
import json
import os
from typing import Optional

import faiss
import numpy as np
from datasets import load_dataset
from tqdm.auto import tqdm

from ..embeddings.encoder import Embedder
from ..utils.chunker import paragraph_chunk
from ..core.config import settings
from ..core.logger import get_logger


logger = get_logger('indexer')


def build_index(max_docs: int = 1000, out_index: Optional[str] = None, out_meta: Optional[str] = None):
    out_index = out_index or settings.FAISS_INDEX_PATH
    out_meta = out_meta or settings.FAISS_META_PATH
    os.makedirs(os.path.dirname(out_index) or '.', exist_ok=True)
    os.makedirs(os.path.dirname(out_meta) or '.', exist_ok=True)

    logger.info('Loading dataset...')
    ds = load_dataset('scientific_papers', 'arxiv', split='train')
    ds = ds.select(range(min(len(ds), max_docs)))
    embedder = Embedder()

    texts = []
    metas = []
    for doc in tqdm(ds, desc='docs'):
        paper_id = doc.get('id') or (doc.get('title') or '')[:64]
        title = doc.get('title') or ''
        # assemble text fields
        abstract = doc.get('abstract') or ''
        body = ''
        if 'sections' in doc and doc['sections']:
            body = '\n\n'.join([s.get('text', '') for s in doc['sections'] if s.get('text')])
        else:
            body = doc.get('article', '') or ''
        full_text = (abstract + '\n\n' + body).strip()
        if not full_text:
            continue
        chunks = paragraph_chunk(full_text)
        for i, ch in enumerate(chunks):
            texts.append(ch['text'])
            metas.append(
                {
                    'id': f"{paper_id}-{i}",
                    'paper_id': paper_id,
                    'title': title,
                    'start': ch['start'],
                    'end': ch['end'],
                }
            )

    if not texts:
        logger.warning('No text chunks found; aborting index build')
        return

    logger.info('Embedding %d chunks...', len(texts))
    batch = 64
    vectors = []
    for i in tqdm(range(0, len(texts), batch), desc='embed'):
        batch_texts = texts[i : i + batch]
        vecs = embedder.embed_texts(batch_texts)
        vectors.append(vecs)
    vectors = np.vstack(vectors).astype('float32')

    dim = vectors.shape[1]
    logger.info('Building FAISS index with dim=%d...', dim)
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)
    faiss.write_index(index, out_index)

    logger.info('Writing meta to %s...', out_meta)
    with open(out_meta, 'w', encoding='utf-8') as f:
        for m in metas:
            f.write(json.dumps(m, ensure_ascii=False) + '\n')
    logger.info('Index build complete.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-docs', type=int, default=1000)
    parser.add_argument('--out-index', type=str, default=None)
    parser.add_argument('--out-meta', type=str, default=None)
    args = parser.parse_args()
    build_index(args.max_docs, args.out_index, args.out_meta)