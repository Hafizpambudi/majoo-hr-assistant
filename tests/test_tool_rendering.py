"""Tool payload rendering.

Raw JSON sent to the writer made it echo field names at the user, e.g.
"13 hari (annual_remaining)". MCP content blocks also carry a transport `id`
that the model copied into answers as if it were a citation.
"""

import json

from app.agent.nodes import _extract_text, _render_tool_payload


def test_flat_dict_becomes_key_value_lines():
    raw = json.dumps({"annual_remaining": 9, "name": "Andi Pratama"})
    rendered = _render_tool_payload(raw)
    assert "annual_remaining: 9" in rendered
    assert "name: Andi Pratama" in rendered
    assert "{" not in rendered


def test_list_of_dicts_is_expanded():
    raw = json.dumps(
        {
            "direct_report_count": 2,
            "direct_reports": [
                {"employee_id": "EMP-002", "name": "Siti"},
                {"employee_id": "EMP-003", "name": "Budi"},
            ],
        }
    )
    rendered = _render_tool_payload(raw)
    assert "direct_reports:" in rendered
    assert "employee_id=EMP-002" in rendered
    assert "employee_id=EMP-003" in rendered


def test_non_json_payload_passes_through():
    assert _render_tool_payload("plain text") == "plain text"


def test_scalar_json_passes_through():
    assert _render_tool_payload(json.dumps([1, 2])) == "[1, 2]"


def test_mcp_content_blocks_are_unwrapped_and_id_dropped():
    payload = [
        {
            "type": "text",
            "text": json.dumps({"annual_remaining": 13}),
            "id": "lc_cc431288-31e1-4988-ae15-89b457f51594",
        }
    ]
    rendered = _render_tool_payload(payload)
    assert "annual_remaining: 13" in rendered
    assert "lc_" not in rendered


def test_extract_text_joins_multiple_blocks():
    blocks = [
        {"type": "text", "text": "first", "id": "lc_1"},
        {"type": "text", "text": "second", "id": "lc_2"},
    ]
    assert _extract_text(blocks) == "first\nsecond"


def test_extract_text_ignores_non_text_blocks():
    blocks = [{"type": "image", "data": "..."}, {"type": "text", "text": "kept"}]
    assert _extract_text(blocks) == "kept"
