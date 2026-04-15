from __future__ import annotations

from pathlib import Path
from typing import Iterable

import csv


def render_ascii_table(title: str, rows: list[tuple[str, str]]) -> str:
    key_width = max((len(str(key)) for key, _ in rows), default=10)
    value_width = max((len(str(value)) for _, value in rows), default=10)
    line = "+" + "-" * (key_width + 2) + "+" + "-" * (value_width + 2) + "+"
    output = [title, line]
    for key, value in rows:
        output.append(f"| {str(key):<{key_width}} | {str(value):<{value_width}} |")
    output.append(line)
    return "\n".join(output)


def mini_chart(values: Iterable[float], width: int = 30) -> str:
    data = list(values)
    if not data:
        return ""
    minimum = min(data)
    maximum = max(data)
    if maximum == minimum:
        return "-" * min(len(data), width)
    blocks = " .:-=+*#%@"
    step = max(1, len(data) // width)
    sampled = data[::step][:width]
    chars = []
    for value in sampled:
        idx = int((value - minimum) / (maximum - minimum) * (len(blocks) - 1))
        chars.append(blocks[idx])
    return "".join(chars)


def export_csv(path: str, headers: list[str], rows: list[list[object]]) -> str:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        writer.writerows(rows)
    return str(target)


def export_text_pdf(path: str, title: str, lines: list[str]) -> str:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    text = [title] + lines
    content_stream = "BT /F1 10 Tf 40 780 Td " + " Tj T* ".join(f"({line[:110].replace('(', '[').replace(')', ']')})" for line in text) + " Tj ET"
    objects = [
        "1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj",
        "2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj",
        "3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >> endobj",
        f"4 0 obj << /Length {len(content_stream)} >> stream\n{content_stream}\nendstream endobj",
        "5 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Courier >> endobj",
    ]

    offsets = []
    pdf = ["%PDF-1.4"]
    for obj in objects:
        offsets.append(sum(len(part.encode('utf-8')) + 1 for part in pdf))
        pdf.append(obj)
    xref_offset = sum(len(part.encode('utf-8')) + 1 for part in pdf)
    pdf.append(f"xref\n0 {len(objects) + 1}\n0000000000 65535 f ")
    pdf.extend(f"{offset:010d} 00000 n " for offset in offsets)
    pdf.append(f"trailer << /Size {len(objects) + 1} /Root 1 0 R >>")
    pdf.append(f"startxref\n{xref_offset}\n%%EOF")
    target.write_text("\n".join(pdf), encoding="utf-8")
    return str(target)
