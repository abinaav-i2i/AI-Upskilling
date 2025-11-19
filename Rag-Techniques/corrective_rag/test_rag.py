from rag import DocumentStore, Document, CorrectiveRAG


def test_high_similarity_avoids_llm():
    store = DocumentStore()
    store.add(Document("1", "Cats are small carnivorous mammals. They purr."))
    rag = CorrectiveRAG(store, direct_answer_threshold=0.5, llm_threshold=0.2)
    out = rag.generate("What sound do cats make?")
    assert out["used_llm"] is False
    assert "purr" in out["answer"].lower()


def test_low_similarity_triggers_no_llm_by_default():
    store = DocumentStore()
    store.add(Document("1", "Unrelated content about gardening and plants."))
    rag = CorrectiveRAG(store, direct_answer_threshold=0.9, llm_threshold=0.5)
    out = rag.generate("Explain quantum entanglement")
    assert out["used_llm"] is False
    assert "low confidence" in out["answer"].lower() or "no relevant" in out["answer"].lower()


def test_force_llm_causes_call():
    store = DocumentStore()
    rag = CorrectiveRAG(store, direct_answer_threshold=0.9, llm_threshold=0.9)
    out = rag.generate("Make a short summary of AI", force_llm=True)
    assert out["used_llm"] is True
