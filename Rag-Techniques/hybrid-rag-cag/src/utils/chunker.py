from typing import List
import hashlib


def hash_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def paragraph_chunk(text: str, chunk_size_words: int = 500, overlap_words: int = 80):
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    cur = []
    cur_words = 0
    start_par = 0
    for i, p in enumerate(paragraphs):
        w = len(p.split())
        cur.append(p)
        cur_words += w
        if cur_words >= chunk_size_words:
            chunk_text = "\n\n".join(cur)
            chunks.append({"text": chunk_text, "start": start_par, "end": i})
            if overlap_words > 0:
                kept = []
                kept_words = 0
                for para in reversed(cur):
                    pw = len(para.split())
                    if kept_words + pw > overlap_words:
                        break
                    kept.insert(0, para)
                    kept_words += pw
                cur = kept
                cur_words = kept_words
                start_par = i - len(kept) + 1
            else:
                cur = []
                cur_words = 0
                start_par = i + 1
    if cur:
        chunks.append(
            {"text": "\n\n".join(cur), "start": start_par, "end": len(paragraphs) - 1}
        )
    return chunks
