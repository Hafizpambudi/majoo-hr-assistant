from app.ingestion.chunker import chunk_document

SAMPLE = """# Cuti

Intro paragraph about leave.

## Cuti Tahunan

Setiap karyawan memperoleh 12 hari cuti tahunan.

Setiap karyawan memperoleh 12 hari cuti tahunan.

## Carry Over

Maksimal 5 hari carry over.
"""


def test_splits_on_headings_and_keeps_label():
    chunks = chunk_document(SAMPLE, "doc", "Doc", chunk_size=500, chunk_overlap=50)
    headings = [c.heading for c in chunks]
    assert any("Cuti Tahunan" in h for h in headings)
    assert any("Carry Over" in h for h in headings)


def test_chunk_ids_are_unique():
    chunks = chunk_document(SAMPLE, "doc", "Doc", chunk_size=40, chunk_overlap=10)
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))


def test_respects_size_with_overlap():
    body = "kalimat " * 200
    chunks = chunk_document(body, "doc", "Doc", chunk_size=200, chunk_overlap=40)
    assert chunks
    assert max(len(c.text) for c in chunks) <= 220


def test_long_paragraph_is_split_with_overlap():
    text = "a" * 500
    chunks = chunk_document(text, "doc", "Doc", chunk_size=200, chunk_overlap=50)
    assert len(chunks) >= 3
    assert chunks[0].text[-50:] in chunks[1].text
