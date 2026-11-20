"""Metric logic tests.

These guard two bugs found during the first real eval run: a correct refusal
worded as "tidak ditemukan" scored as a false answer, and an answer citing
[2][3] scored as uncited because the check only looked for [1].
"""

from eval.run_eval import deterministic_report, has_citation, is_refusal


def test_exact_refusal_sentence_is_a_refusal():
    assert is_refusal(
        "Maaf, saya tidak tahu. Informasi itu tidak ada di dokumen HR yang saya akses."
    )


def test_tool_missing_record_counts_as_refusal():
    assert is_refusal(
        "Tidak ada data cuti untuk EMP-777 karena tidak ditemukan catatan karyawan tersebut."
    )


def test_real_answer_is_not_a_refusal():
    assert not is_refusal("Sisa cuti Anda adalah 9 hari [1].")


def test_any_citation_number_counts():
    assert has_citation("Plafon Rp25.000.000 [2][3]")
    assert has_citation("Maksimal 5 hari [1]")
    assert not has_citation("Sisa cuti 9 hari.")


def _row(**kwargs) -> dict:
    base = {
        "id": "Q1",
        "question": "q",
        "answer": "a",
        "reference": "r",
        "contexts": [],
        "retrieved": [],
        "expected_tools": [],
        "called_tools": [],
        "tools_ok": True,
        "expect_refusal": False,
        "refused": False,
        "refusal_ok": True,
        "cited": True,
        "tool_only": False,
        "latency_ms": 100.0,
        "run_id": "r",
    }
    base.update(kwargs)
    return base


def test_tool_only_answers_are_excluded_from_citation_rate():
    rows = [
        _row(id="Q1", cited=True),
        _row(id="Q2", cited=False, tool_only=True, called_tools=["get_leave_balance"]),
    ]
    report = deterministic_report(rows)
    assert report["citation_rate"] == 1.0
    assert report["documented_answers"] == 1


def test_uncited_document_answer_lowers_the_rate():
    rows = [_row(id="Q1", cited=True), _row(id="Q2", cited=False)]
    assert deterministic_report(rows)["citation_rate"] == 0.5


def test_partial_save_never_shrinks_an_existing_result_file(tmp_path, monkeypatch):
    # A run that dies on its first question must not erase a complete prior run.
    import json

    import eval.run_eval as run_eval

    monkeypatch.setattr(run_eval, "OUT_DIR", tmp_path)
    run_eval.save_partial([_row(id="Q1"), _row(id="Q2")])
    run_eval.save_partial([])
    saved = json.loads((tmp_path / "results.json").read_text(encoding="utf-8"))
    assert len(saved["rows"]) == 2


def test_partial_save_appends_growth(tmp_path, monkeypatch):
    import json

    import eval.run_eval as run_eval

    monkeypatch.setattr(run_eval, "OUT_DIR", tmp_path)
    run_eval.save_partial([_row(id="Q1")])
    run_eval.save_partial([_row(id="Q1"), _row(id="Q2"), _row(id="Q3")])
    saved = json.loads((tmp_path / "results.json").read_text(encoding="utf-8"))
    assert [r["id"] for r in saved["rows"]] == ["Q1", "Q2", "Q3"]


def test_false_answer_and_false_refusal_rates():
    rows = [
        _row(id="Q1", expect_refusal=True, refused=False),
        _row(id="Q2", expect_refusal=False, refused=True),
        _row(id="Q3"),
    ]
    report = deterministic_report(rows)
    assert report["false_answer_rate"] == round(1 / 3, 3)
    assert report["false_refusal_rate"] == round(1 / 3, 3)
