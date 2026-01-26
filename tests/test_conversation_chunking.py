# tests/test_conversation_chunking.py
from chinvex.chunking import chunk_conversation, approx_tokens
from math import ceil


def test_approx_tokens() -> None:
    text = "hello world"
    expected = ceil(len(text) / 4)
    assert approx_tokens(text) == expected


def test_chunk_conversation_respects_token_limit() -> None:
    # Create ConversationDoc with many long turns
    turns = []
    for i in range(50):
        turns.append({
            "turn_id": f"turn-{i}",
            "ts": "2026-01-26T10:00:00Z",
            "role": "user" if i % 2 == 0 else "assistant",
            "text": "x" * 500,  # ~125 tokens each
        })

    conversation = {
        "doc_type": "conversation",
        "source": "cx_appserver",
        "thread_id": "thread-123",
        "title": "Test Thread",
        "created_at": "2026-01-26T10:00:00Z",
        "updated_at": "2026-01-26T10:30:00Z",
        "turns": turns,
        "links": {}
    }

    chunks = chunk_conversation(conversation, max_tokens=1500)

    # Verify chunks exist
    assert len(chunks) > 0

    # Verify no chunk exceeds token limit
    for chunk in chunks:
        tokens = approx_tokens(chunk.text)
        assert tokens <= 1500, f"Chunk exceeded token limit: {tokens} > 1500"

    # Verify turn markers present
    for chunk in chunks:
        assert "[Turn" in chunk.text


def test_chunk_conversation_never_splits_single_turn() -> None:
    # Create one giant turn that exceeds limit
    turns = [{
        "turn_id": "turn-huge",
        "ts": "2026-01-26T10:00:00Z",
        "role": "assistant",
        "text": "x" * 10000,  # ~2500 tokens, exceeds 1500 limit
    }]

    conversation = {
        "doc_type": "conversation",
        "source": "cx_appserver",
        "thread_id": "thread-huge",
        "title": "Huge Turn",
        "created_at": "2026-01-26T10:00:00Z",
        "updated_at": "2026-01-26T10:00:00Z",
        "turns": turns,
        "links": {}
    }

    chunks = chunk_conversation(conversation, max_tokens=1500)

    # Should have exactly 1 chunk (never split a turn)
    assert len(chunks) == 1
    # It will exceed limit, but that's OK per spec
    assert approx_tokens(chunks[0].text) > 1500
