"""Script to regenerate every performance metric, gate status, and summary table in reports/technical_report.md."""

import sys
from pathlib import Path

# Add src and bench to sys.path
root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root / "src"))
sys.path.insert(0, str(root))

import json

def generate_markdown_table() -> str:
    lines = [
        "| Gate | Metric Description | Status | Evidence | Detail |",
        "|---|---|---|---|---|",
    ]
    try:
        res = json.loads((root / "out" / "bench_results.json").read_text(encoding="utf-8"))
    except:
        res = None
    if not res:
        return "\n".join(lines)
    for g in res.get("gates", []):
        status_badge = f"[{g['status'].upper()}]"
        lines.append(f"| **{g['gate']}** | {g['name']} | {status_badge} | {g['evidence']} | {g['detail']} |")
    
    return "\n".join(lines)

def main():
    table = generate_markdown_table()
    out_file = root / "reports" / "gate_table.md"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(table, encoding="utf-8")
    print(f"Wrote {out_file}")

if __name__ == "__main__":
    main()
