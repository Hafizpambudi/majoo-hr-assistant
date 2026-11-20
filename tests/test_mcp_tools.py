from app.mcp_server.server import get_leave_balance, get_subordinat_list


def test_leave_balance_math():
    result = get_leave_balance("EMP-001")
    assert result["found"] is True
    expected = (
        result["annual_entitlement"]
        + result["carry_over_days"]
        - result["annual_used"]
        - result["annual_pending"]
    )
    assert result["annual_remaining"] == expected == 9


def test_leave_balance_is_case_insensitive():
    assert get_leave_balance("emp-002")["found"] is True


def test_leave_balance_unknown_employee():
    result = get_leave_balance("EMP-777")
    assert result["found"] is False
    assert result["error"] == "employee_not_found"


def test_subordinate_list():
    result = get_subordinat_list("EMP-001")
    ids = [r["employee_id"] for r in result["direct_reports"]]
    assert result["direct_report_count"] == 2
    assert ids == ["EMP-002", "EMP-003"]


def test_subordinate_list_leaf_node_is_empty():
    result = get_subordinat_list("EMP-003")
    assert result["found"] is True
    assert result["direct_reports"] == []


def test_subordinate_list_unknown_employee():
    assert get_subordinat_list("EMP-999")["found"] is False
