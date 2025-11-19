"""CLI entrypoint. Modes:
- --mode query --question "..." -> compute embedding and retrieve (no LLM call)
- --mode run -> interactive loop: check cache -> RAG -> call LLM -> optionally cache
"""
import argparse
import os
import json
import time
from .core.logger import get_logger
from .core.config import settings
from .retriever.query import Retriever
from .cache.redis_cache import RedisCache
from .llm.caller import call_llm


logger = get_logger('main')
SYSTEM_PROMPT = (
    "You are an assistant answering questions about scientific papers. "
    "Use retrieved passages and cite the title when possible. If unsure, say you don't know."
)


def mode_query(retriever: Retriever, question: str, k: int = 5):
    hits, q_emb = retriever.retrieve(question, k=k)
    print('\n--- Retrieval Results ---')
    for i, h in enumerate(hits):
        print(
            f"[{i+1}] id={h['meta'].get('id','')} score={h['score']:.4f} title={h['meta'].get('title','')}")
    print('--- End ---\n')
    return hits, q_emb


def mode_run(retriever: Retriever, cache: RedisCache):
    print('Interactive run mode. Type "exit" to quit.')
    while True:
        q = input('Question> ').strip()
        if not q:
            continue
        if q.lower() in ('exit', 'quit'):
            break

        # 1) embed and check cache
        hits, q_emb = retriever.retrieve(q, k=5)
        try:
            cached = cache.get_similar(q_emb)
        except Exception:
            logger.debug('Cache read failed; continuing', exc_info=True)
            cached = None
        if cached:
            logger.info('Cache hit sim=%.3f key=%s', cached.get('sim', 0.0), cached.get('key'))
            print('\n[CACHE]')
            print(cached['answer'])
            continue

        # 2) Build prompt using top hits
        evidence_texts = []
        for h in hits:
            evidence_texts.append(
                f"Title: {h['meta'].get('title','')}\nExcerpt: {h['meta'].get('excerpt','(no excerpt)')}\n"
            )
        prompt = '\n\n'.join(evidence_texts) + '\n\nQuestion: ' + q

        # 3) Call LLM
        try:
            answer = call_llm(SYSTEM_PROMPT, prompt, max_tokens=512)
        except Exception:
            logger.exception('LLM call failed')
            print('LLM backend error')
            continue

        print('\n[LLM ANSWER]')
        print(answer)

        # 4) cache heuristics (simple)
        try:
            if answer and len(answer) < 2000 and len(hits) > 0:
                meta = {
                    'retrieved_ids': [h['meta'].get('id') for h in hits],
                    'timestamp': int(time.time()),
                }
                cache.set(q_emb, answer, meta)
                logger.debug('Cached answer for question')
        except Exception:
            logger.debug('Failed to cache answer', exc_info=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['query', 'run'], required=True)
    parser.add_argument('--question', type=str, default='')
    parser.add_argument('--k', type=int, default=5)
    parser.add_argument('--max-docs', type=int, default=1000)
    args = parser.parse_args()

    retriever = Retriever()
    cache = RedisCache()
    if args.mode == 'query':
        if not args.question:
            raise SystemExit('Please provide --question for query mode')
        mode_query(retriever, args.question, k=args.k)
    else:
        mode_run(retriever, cache)


if __name__ == '__main__':
    main()