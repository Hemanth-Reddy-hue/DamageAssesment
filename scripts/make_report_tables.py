"""Script to regenerate every performance metric, gate status, and summary table in reports/technical_report.md."""

import sys
from pathlib import Path

# Add src and bench to sys.path
root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root / "src"))
sys.path.insert(0, str(root))

from bench.gates import EVALUATED_GATES

def generate_markdown_table() -> str:
    lines = [
        "| Gate | Metric Description | Threshold | Measured Result | Pass/Fail |",
        "|---|---|---|---|---|",
    ]
    for g_id, g_data in EVALUATED_GATES.items():
        status_badge = "[PASS]" if g_data["pass"] else "[FAIL]"
        lines.append(f"| **{g_id}** | {g_data['description']} | {g_data['threshold']} | {g_data['measured']} | {status_badge} |")

    return "\n".join(lines)

def main():
    table = generate_markdown_table()
    print("=== AreaMap Official Gates Table ===")
    print(table)

if __name__ == "__main__":
    main()
