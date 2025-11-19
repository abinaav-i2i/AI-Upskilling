# src/summarizer.py
import os
from typing import Dict, Optional

# You may keep a transformer summarizer, but default to rule-based templating
USE_MODEL = os.environ.get("USE_TRANSFORMER_SUMMARIZER", "false").lower() in ("1", "true", "yes")
MODEL_NAME = os.environ.get("SUMMARIZER_MODEL", "t5-small")

if USE_MODEL:
    from transformers import pipeline
    # Force CPU
    summarizer = pipeline("summarization", model=MODEL_NAME, device=-1)

def template_summary_from_payload(payload: Dict) -> str:
    """
    Compose a human-readable one-liner from structured payload fields.
    This is deterministic, fast, and readable on small datasets like AmesHousing.
    """
    parts = []
    # Price
    price = payload.get("SalePrice") or payload.get("price")
    if price:
        try:
            price_str = f"${int(price):,}"
        except Exception:
            price_str = str(price)
        parts.append(f"Price: {price_str}")
    # Area
    area = payload.get("Gr Liv Area") or payload.get("gr_liv_area") or payload.get("GrLivArea")
    if area:
        parts.append(f"Area: {int(area)} sqft")
    # Bedrooms / bathrooms
    beds = payload.get("Bedroom AbvGr") or payload.get("bedrooms")
    baths = payload.get("Full Bath") or payload.get("FullBath") or payload.get("bathrooms")
    if beds:
        parts.append(f"{int(beds)} bed")
    if baths:
        parts.append(f"{int(baths)} bath")
    # Neighborhood
    neigh = payload.get("Neighborhood")
    if neigh:
        parts.append(f"Neighborhood: {neigh}")
    # Year built
    yb = payload.get("Year Built") or payload.get("YearBuilt")
    if yb:
        parts.append(f"Built: {int(yb)}")
    # Garage / pool
    gcars = payload.get("Garage Cars") or payload.get("GarageCars")
    if gcars:
        parts.append(f"Garage: {gcars} cars")
    pool = payload.get("Pool Area")
    if pool and int(pool) > 0:
        parts.append("Has pool")
    # Fallback: if no structured parts, use 'text' snippet
    if not parts:
        text = payload.get("text", "")
        return (text[:200] + "...") if len(text) > 200 else text
    return " · ".join(parts)

def summarize(payload: Dict, max_length: int = 60, min_length: int = 20) -> str:
    """
    If USE_MODEL is true, use transformer summarizer on payload['text'].
    Otherwise build a template summary from payload (preferred for speed & clarity).
    """
    if not payload:
        return ""
    if not USE_MODEL:
        return template_summary_from_payload(payload)
    # else model path
    text = payload.get("text", "")
    if not text:
        return template_summary_from_payload(payload)
    # prefer using max_new_tokens to avoid the transformers warning
    try:
        out = summarizer(text[:4000], max_new_tokens=max_length, do_sample=False)
        return out[0]["summary_text"]
    except Exception:
        # fallback to template
        return template_summary_from_payload(payload)
