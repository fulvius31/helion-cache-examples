"""Render the recorded results as SVGs; Python standard library only.

Usage: python plot_results.py
This redraws the published measurements. It does not run GPU experiments.
"""

from __future__ import annotations

from html import escape
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
COLORS = ["#526782", "#23834a", "#d87b24"]


def text(x: float, y: float, value: str, size: int = 15, **attrs: str) -> str:
    extra = " ".join(f'{k.replace("_", "-")}="{v}"' for k, v in attrs.items())
    return f'<text x="{x}" y="{y}" font-size="{size}" {extra}>{escape(value)}</text>'


def chart(title: str, subtitle: str, rows: list[dict], panels: list[tuple],
          notes: list[str], filename: str, colors: list[str] = COLORS) -> None:
    width, height = 1040, 390
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'role="img" aria-labelledby="title desc">',
        f'<title id="title">{escape(title)}</title>',
        f'<desc id="desc">{escape(subtitle + " " + " ".join(notes))}</desc>',
        '<rect width="1040" height="390" rx="12" fill="#f8fafc"/>',
        '<g font-family="system-ui, sans-serif" fill="#17293c">',
        text(28, 34, title, 23, font_weight="650"),
        text(28, 61, subtitle, 14, fill="#516174"),
    ]
    for panel, (heading, key, scale, maximum, unit) in enumerate(panels):
        x = 28 + panel * 520
        left, span = x + 148, 274
        parts.append(text(x, 104, heading, 17, font_weight="600"))
        for index, row in enumerate(rows):
            y = 135 + index * 49
            value = row[key] * scale
            bar = span * value / maximum
            parts.extend([
                text(x, y + 19, row["label"], 14),
                f'<rect x="{left}" y="{y}" width="{span}" height="27" rx="3" fill="#e5ebf1"/>',
                f'<rect x="{left}" y="{y}" width="{bar:.2f}" height="27" rx="3" fill="{colors[index]}"/>',
                text(left + bar + 7, y + 19, f'{value:.1f}', 14, font_weight="600"),
            ])
        axis_y = 135 + len(rows) * 49 + 7
        parts.extend([
            text(left, axis_y, "0", 12, fill="#64748b"),
            text(left + span, axis_y, f'{maximum:g} {unit}', 12,
                 fill="#64748b", text_anchor="end"),
        ])
    for index, note in enumerate(notes):
        parts.append(text(28, 337 + index * 21, note, 14, fill="#516174"))
    parts.append('</g></svg>')
    (ROOT / "plots" / filename).write_text("\n".join(parts) + "\n")


def main() -> None:
    data = json.loads((ROOT / "results" / "recorded_b200.json").read_text())
    (ROOT / "plots").mkdir(exist_ok=True)
    rows = [dict(row) for row in data["developer_iteration_1024"][1:]]
    rows[0]["label"], rows[1]["label"] = "Edited, warm quick", "Edited, cold quick"
    chart(
        "Keep tuning knowledge across source edits",
        "1024³ matmul · quick effort · same edited kernel · lower is better",
        rows, [("End-to-end run time", "wall_s", 1, 75, "s"),
               ("Measured kernel time", "kernel_ms", 1000, 45, "µs")],
        ["Warm-start transferred 1 prior winner: 10 initial configs versus 9 in the cold control.",
         "The warm result was ~3.9% lower latency, but took 3.4 s longer. No iteration-time speedup here."],
        "developer_iteration.svg", [COLORS[1], COLORS[2]],
    )
    chart(
        "Reuse a full-search winner during quick re-tuning",
        "4096³ matmul · full → warm quick → cold quick · lower is better",
        data["full_then_quick_4096"],
        [("End-to-end run time", "wall_s", 1, 300, "s"),
         ("Measured kernel time", "kernel_ms", 1000, 140, "µs")],
        ["Warm quick: 45.8 s versus 225.7 s full (4.9× less wall time), at comparable kernel latency.",
         "Different efforts and a fixed run order: this is not a 4.9× speedup attributable to warm-start alone."],
        "full_then_quick.svg",
    )
    print("Updated plots/developer_iteration.svg and plots/full_then_quick.svg")


if __name__ == "__main__":
    main()
