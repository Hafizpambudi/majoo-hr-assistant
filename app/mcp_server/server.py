"""MCP server exposing HR self-service tools over stdio.

Runs as its own process so the agent has no import path to the data layer: every
call crosses the MCP protocol, exactly as it would against a real HRIS backend.

Nothing may be written to stdout - stdout is the JSON-RPC channel. Diagnostics
go to stderr.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from mcp.server.fastmcp import FastMCP

DATA_PATH = Path(__file__).parent / "data" / "employees.json"

mcp = FastMCP("majo-hr-tools")


def _load() -> dict:
    return json.loads(DATA_PATH.read_text(encoding="utf-8"))["employees"]


def _log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


@mcp.tool()
def get_leave_balance(employee_id: str) -> dict:
    """Get the annual and sick leave balance for one employee.

    Args:
        employee_id: Employee identifier, for example "EMP-001".

    Returns remaining annual leave, pending requests, carry-over days and the
    carry-over expiry date.
    """
    employee = _load().get(employee_id.strip().upper())
    if employee is None:
        return {
            "found": False,
            "employee_id": employee_id,
            "error": "employee_not_found",
            "message": f"No employee record for {employee_id}.",
        }

    leave = employee["leave"]
    remaining = (
        leave["annual_entitlement"]
        + leave["carry_over_from_last_year"]
        - leave["annual_used"]
        - leave["annual_pending"]
    )
    return {
        "found": True,
        "employee_id": employee["employee_id"],
        "name": employee["name"],
        "department": employee["department"],
        "year": leave["year"],
        "annual_entitlement": leave["annual_entitlement"],
        "carry_over_days": leave["carry_over_from_last_year"],
        "carry_over_expiry": leave["carry_over_expiry"],
        "annual_used": leave["annual_used"],
        "annual_pending": leave["annual_pending"],
        "annual_remaining": remaining,
        "sick_used": leave["sick_used"],
        "unlimited_sick_leave": leave["unlimited_sick_leave"],
    }


@mcp.tool()
def get_subordinat_list(employee_id: str) -> dict:
    """List the direct reports of one employee.

    Args:
        employee_id: Manager employee identifier, for example "EMP-001".

    Returns the direct subordinates with their job title and department.
    """
    employees = _load()
    manager = employees.get(employee_id.strip().upper())
    if manager is None:
        return {
            "found": False,
            "employee_id": employee_id,
            "error": "employee_not_found",
            "message": f"No employee record for {employee_id}.",
        }

    reports = [
        {
            "employee_id": e["employee_id"],
            "name": e["name"],
            "job_title": e["job_title"],
            "department": e["department"],
            "employment_type": e["employment_type"],
        }
        for e in employees.values()
        if e.get("manager_id") == manager["employee_id"]
    ]
    return {
        "found": True,
        "employee_id": manager["employee_id"],
        "name": manager["name"],
        "direct_report_count": len(reports),
        "direct_reports": sorted(reports, key=lambda r: r["employee_id"]),
    }


def main() -> None:
    _log("majo-hr-tools MCP server starting on stdio")
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
