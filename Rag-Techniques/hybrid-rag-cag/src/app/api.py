from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

from ..retriever.query import Retriever
from ..cache.redis_cache import RedisCache
from ..core.logger import get_logger
from ..llm.caller import call_llm


logger = get_logger('api')
app = FastAPI(title='RAG+CAG API')


# instantiate lazily to allow import-time failures to surface clearly
def _get_retriever() -> Retriever:
    return Retriever()


def _get_cache() -> RedisCache:
    return RedisCache()


class QueryIn(BaseModel):
    question: str
    k: int = 5


class QueryOut(BaseModel):
    answer: str
    source: str
    retrieved: Optional[int] = None


@app.post('/query', response_model=QueryOut)
def query_endpoint(payload: QueryIn):
    q = (payload.question or '').strip()
    if not q:
        raise HTTPException(status_code=400, detail='question must not be empty')
    retriever = _get_retriever()
    cache = _get_cache()
    try:
        hits, q_emb = retriever.retrieve(q, k=payload.k)
    except Exception as exc:  # pragma: no cover - surface retriever issues
        logger.exception('Retriever failure')
        raise HTTPException(status_code=500, detail='retriever error')
    # check cache
    try:
        cached = cache.get_similar(q_emb)
    except Exception:
        logger.debug('Cache read failed, continuing without cache', exc_info=True)
        cached = None
    if cached:
        return QueryOut(answer=cached['answer'], source='cache', retrieved=len(hits))

    if not hits:
        # no documents retrieved -> safe fallback
        return QueryOut(answer='No relevant documents found.', source='rag', retrieved=0)

    # build prompt from top-k hits (titles + short excerpt if present)
    evidence_texts = []
    for h in hits:
        title = h['meta'].get('title', '')
        excerpt = h['meta'].get('excerpt') or ''
        evidence_texts.append(f"Title: {title}\nExcerpt: {excerpt}")
    prompt = '\n\n'.join(evidence_texts) + '\n\nQuestion: ' + q
    try:
        answer = call_llm('You are an assistant. Cite evidence when possible.', prompt)
    except Exception:
        logger.exception('LLM call failed')
        raise HTTPException(status_code=500, detail='llm backend error')
    return QueryOut(answer=answer, source='rag', retrieved=len(hits))