import json
from pathlib import Path

import pytest

from faq_service.application.ingestion.pipeline import chunk_text, load_documents
from faq_service.domain.errors import ServiceError
from faq_service.domain.validation import validate_vector


def test_actual_dataset_matches_audit():
    docs, rejected, _ = load_documents(Path("temp/data/train_our.jsonl"))
    assert len(docs) == 2741
    assert len(rejected) == 123
    assert sum(len(d.sources) for d in docs) == 2924


def test_dedup_retains_variants_and_valid_short_answers(tmp_path):
    path = tmp_path / "data.jsonl"
    pairs = [
        ("Вопрос?", "Нет."),
        ("Вопрос?", "Нет."),
        ("Вопрос?", "Да."),
        ("Что?", "!"),
        ("Что?", "card"),
    ]
    path.write_text(
        "\n".join(
            json.dumps(
                {
                    "messages": [
                        {"role": "user", "content": q},
                        {"role": "assistant", "content": a},
                    ]
                }
            )
            for q, a in pairs
        )
        + "\nnot json"
    )
    docs, rejected, _ = load_documents(path)
    assert len(docs) == 2
    assert len(docs[0].sources) == 2
    assert len(rejected) == 3
    assert docs[0].body == "Нет."


def test_chunks_cover_text_without_truncation():
    text = "Сохраните документ. " * 500
    chunks = list(chunk_text(text, 200, 20))
    assert all(len(c) <= 200 for c in chunks)
    assert len(chunks) > 1
    restored = chunks[0] + "".join(c[20:] for c in chunks[1:])
    assert restored == text
    assert list(chunk_text("x" * 500, 200, 20))[-1] == "x" * 140


@pytest.mark.parametrize("vector", [[0, 0, 0], [1, 2], [1, float("nan"), 0]])
def test_invalid_embeddings_rejected(vector):
    with pytest.raises(ServiceError):
        validate_vector(vector, 3)
