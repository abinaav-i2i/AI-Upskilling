# src/cli.py
"""
Single-command Typer CLI for retrieval + optional reranking.
Run with:
  python -m src.cli --query "your query" --k 5
  python -m src.cli --query "..." --k 5 --rerank --rerank-fetch 20
"""
import typer
from src.recommender import recommend_from_text, rerank_hits
from src.summarizer import summarize
from src.qdrant_utils import get_client

app = typer.Typer(add_completion=False, no_args_is_help=True)

@app.callback(invoke_without_command=True)
def main(
    query: str = typer.Option(..., "--query", "-q", help="Search query text"),
    k: int = typer.Option(5, "--k", "-k", help="Number of final top results to display"),
    rerank: bool = typer.Option(False, "--rerank", help="Enable Cross-Encoder reranking"),
    rerank_model: str = typer.Option("cross-encoder/ms-marco-MiniLM-L-6-v2", "--rerank-model", help="Cross-Encoder model name"),
    rerank_fetch: int = typer.Option(20, "--rerank-fetch", help="Fetch this many items from Qdrant before reranking"),
    device: str = typer.Option("cpu", "--device", help="'cpu' or 'cuda'"),
):
    """
    Retrieve and (optionally) rerank building/property listings.
    Example:
      python -m src.cli --query "2 bedroom apartment with balcony in Mumbai" --k 5 --rerank
    """
    client = get_client()
    fetch_k = rerank_fetch if rerank else k

    # 1) ANN search (fast)
    hits = recommend_from_text(query, top_k=fetch_k, client=client)
    if not hits:
        typer.echo("No hits found.")
        raise typer.Exit()

    # 2) Optional reranking
    final_results = []
    if rerank:
        typer.echo(f"Reranking {len(hits)} hits using {rerank_model} on {device}...")
        scored = rerank_hits(query, hits, rerank_model=rerank_model, top_k=k, device=device)
        # scored: list of {"hit": ScoredPoint, "rerank_score": float}
        final_results = scored
    else:
        # Wrap ScoredPoints into uniform dicts so printing is common
        final_results = [{"hit": h, "rerank_score": getattr(h, "score", 0.0)} for h in hits[:k]]

    # 3) Pretty print the results
    for idx, item in enumerate(final_results, start=1):
        h = item["hit"]
        score = item["rerank_score"]
        payload = h.payload or {}

        # Use summarizer (by default templated); summarize expects payload dict
        summary = summarize(payload)

        print(f"{idx}. (score={score:.4f}) — {summary}")

        # Show structured details (if present)
        details = []
        if payload.get("SalePrice"):
            try:
                details.append(f"Price: ${int(payload['SalePrice']):,}")
            except Exception:
                details.append(f"Price: {payload['SalePrice']}")
        if payload.get("Gr Liv Area"):
            details.append(f"Area: {int(payload['Gr Liv Area'])} sqft")
        if payload.get("Bedroom AbvGr"):
            details.append(f"{int(payload['Bedroom AbvGr'])} bed")
        if payload.get("Full Bath"):
            details.append(f"{int(payload['Full Bath'])} bath")
        if payload.get("Neighborhood"):
            details.append(f"Neighborhood: {payload['Neighborhood']}")

        if details:
            print("   " + " | ".join(details))

        print("-" * 80)

if __name__ == "__main__":
    app()
