import os
from pydantic import BaseSettings


class Settings(BaseSettings):
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    EMBED_MODEL: str = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
    FAISS_INDEX_PATH: str = os.getenv("FAISS_INDEX_PATH", "./data/faiss.index")
    FAISS_META_PATH: str = os.getenv("FAISS_META_PATH", "./data/faiss_meta.jsonl")
    CACHE_TTL_SECONDS: int = int(os.getenv("CACHE_TTL_SECONDS", 86400))
    LLM_BACKEND: str = os.getenv("LLM_BACKEND", "openai")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")


class Config:
    env_file = ".env"


settings = Settings()
