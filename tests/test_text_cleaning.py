from app.agent.nodes import REFUSAL, clean_text, parse_json


def test_citation_line_range_is_stripped():
    assert clean_text("Plafon Rp25.000.000[2\u2020L1-L2]") == "Plafon Rp25.000.000[2]"


def test_tool_transport_id_is_not_a_citation():
    assert clean_text("Sisa 13 hari[lc_cc431288-31e1].") == "Sisa 13 hari."


def test_cjk_brackets_become_ascii():
    assert clean_text("Sisa 9 hari\u30101\u3011") == "Sisa 9 hari[1]"


def test_narrow_spaces_and_non_breaking_hyphens():
    assert clean_text("EMP\u2011001\u202fadalah") == "EMP-001 adalah"


def test_plain_citation_is_untouched():
    assert clean_text("Maksimal 5 hari [1].") == "Maksimal 5 hari [1]."


def test_parse_json_handles_fenced_output():
    assert parse_json('```json\n{"grounded": true}\n```') == {"grounded": True}


def test_parse_json_handles_leading_prose():
    assert parse_json('Here you go: {"grounded": false} done') == {"grounded": False}


def test_refusal_sentence_is_stable():
    assert REFUSAL.startswith("Maaf, saya tidak tahu.")
