Structure and quick start:
1. Copy `.env.example` to `.env` and fill in OPENAI_API_KEY if you will use OpenAI.
2. Start redis locally or use docker-compose: `docker-compose up -d`
3. Install dependencies: `pip install -r requirements.txt`
4. Build the FAISS index (example): `python -m src.indexer.build_index --max-docs 1000`
5. Run queries in two modes:
- `python -m src.main --mode query --question "What are the main contributions of paper X?"`
- `python -m src.main --mode run` (interactive loop)