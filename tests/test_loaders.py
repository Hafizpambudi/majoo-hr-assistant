from pathlib import Path

from app.config import get_settings
from app.ingestion.loaders import FIRST_PARTY_PREFIX, load_documents


def test_first_party_documents_come_first():
    docs = load_documents(get_settings().docs_dir)
    prefixes = [d.doc_id.startswith(FIRST_PARTY_PREFIX) for d in docs]
    # Every first-party doc precedes every third-party doc.
    assert prefixes == sorted(prefixes, reverse=True)


def test_all_documents_are_loaded():
    docs = load_documents(get_settings().docs_dir)
    assert len(docs) >= 20
    assert all(d.text.strip() for d in docs)


def test_prefix_filter_selects_first_party_only():
    docs = load_documents(get_settings().docs_dir)
    first_party = [d for d in docs if d.doc_id.startswith(FIRST_PARTY_PREFIX)]
    assert len(first_party) >= 5
    assert all(d.path.exists() for d in first_party)


def test_missing_directory_raises():
    import pytest

    with pytest.raises(FileNotFoundError):
        load_documents(Path("does-not-exist"))
