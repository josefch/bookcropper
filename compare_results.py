#!/usr/bin/env python3
"""Create visual overlays and a provider comparison report from AI JSONL output."""
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def latest_valid(path: Path) -> dict[str, dict]:
    result = {}
    if not path.exists():
        return result
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            item = json.loads(line)
            corners = item.get("corners")
            if item.get("source") and item.get("kind") and isinstance(corners, list) and len(corners) == 4:
                result[item["source"]] = item
        except json.JSONDecodeError:
            pass
    return result


def points(item: dict) -> list[tuple[float, float]]:
    return [(float(p["x"]), float(p["y"])) if isinstance(p, dict) else (float(p[0]), float(p[1])) for p in item["corners"]]


def mean_corner_error(a: dict, b: dict) -> float:
    pa, pb = points(a), points(b)
    return sum(math.hypot(x - u, y - v) for (x, y), (u, v) in zip(pa, pb)) / 4


def image_cost_tokens(path: Path) -> int:
    """Anthropic Sonnet high-resolution estimate: 28px patches, max 4784 tokens."""
    with Image.open(path) as image:
        w, h = image.size
    scale = min(1.0, 2576 / max(w, h))
    w, h = round(w * scale), round(h * scale)
    return min(math.ceil(w / 28) * math.ceil(h / 28), 4784)


def overlay(source: Path, openai: dict | None, anthropic: dict | None, out: Path) -> None:
    with Image.open(source).convert("RGB") as image:
        image.thumbnail((1400, 1400))
        canvas = image.copy()
    scale_x = canvas.width / Image.open(source).size[0]
    scale_y = canvas.height / Image.open(source).size[1]
    draw = ImageDraw.Draw(canvas)
    labels = [(openai, "OPENAI", (40, 220, 80)), (anthropic, "ANTHROPIC", (240, 70, 60))]
    y = 10
    for item, label, color in labels:
        if not item:
            continue
        xy = [(round(x * scale_x), round(y0 * scale_y)) for x, y0 in points(item)]
        draw.line(xy + [xy[0]], fill=color, width=max(3, canvas.width // 500), joint="curve")
        draw.text((10, y), f"{label} {item.get('kind')} confidence={item.get('confidence', 0):.2f}", fill=color)
        y += 24
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out, quality=90)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("source", type=Path)
    ap.add_argument("--results", type=Path, default=Path("ai_review_results"))
    ap.add_argument("--output", type=Path, default=Path("ai_review_results/comparison"))
    args = ap.parse_args()
    source = args.source
    providers = {
        "OpenAI": latest_valid(args.results / "openai.jsonl"),
        "Anthropic": latest_valid(args.results / "anthropic.jsonl"),
    }
    all_sources = sorted(set().union(*[set(v) for v in providers.values()]))
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for source_name in all_sources:
        path = Path(source_name)
        a, b = providers["OpenAI"].get(source_name), providers["Anthropic"].get(source_name)
        error = mean_corner_error(a, b) if a and b else None
        rows.append((path, a, b, error))
        if path.exists():
            overlay(path, a, b, args.output / f"{path.stem}_compare.jpg")

    both = [r for r in rows if r[1] and r[2]]
    agreements = [r for r in both if r[1]["kind"] == r[2]["kind"]]
    mean_error = sum(r[3] for r in both) / len(both) if both else 0
    errors = sorted(r[3] for r in both)
    median_error = statistics.median(errors) if errors else 0
    p90_error = errors[min(len(errors) - 1, math.ceil(len(errors) * 0.9) - 1)] if errors else 0
    claude_tokens = sum(image_cost_tokens(r[0]) for r in rows if r[0].exists())
    claude_tokens *= 3  # primary plus two context images per request
    report = [
        "# AI Book-Cover Review Comparison",
        "",
        f"Generated from {len(all_sources)} unique scans.",
        "",
        "| Provider | Valid unique results | Model | Estimated input cost |",
        "|---|---:|---|---:|",
        f"| OpenAI | {len(providers['OpenAI'])} | gpt-5.6-sol | about $1.3 estimated input/output |",
        f"| Anthropic | {len(providers['Anthropic'])} | claude-sonnet-4-6 | about ${claude_tokens * 3 / 1_000_000:.2f} image input |",
        "",
        "## Agreement",
        "",
        f"Both providers returned a result for {len(both)} scans. Their front/back/spine classification agreed on {len(agreements)} of those ({len(agreements) / len(both) * 100:.1f}%). Corner distance was mean {mean_error:.1f}px, median {median_error:.1f}px, and p90 {p90_error:.1f}px in the original scan coordinate system.",
        "",
        "Anthropic estimate uses its documented 28px visual-token patches, high-resolution 4,784-token cap, and $3/M input-token Sonnet rate. It includes the primary plus up to two context images per request and excludes text output. OpenAI estimate uses the observed smoke-test input size of roughly 6.8k tokens per three-image request and the documented $4/M input rate; the API dashboard is authoritative for exact billing.",
        "",
        "## Files",
        "",
        "- `openai.jsonl` and `anthropic.jsonl`: normalized model geometry",
        "- `comparison/*_compare.jpg`: original scan with OpenAI green and Anthropic red polygons",
        "",
        "| Scan | OpenAI | Anthropic | Mean corner distance |",
        "|---|---|---|---:|",
    ]
    for path, a, b, error in rows:
        report.append(f"| `{path.name}` | {a.get('kind', '-') if a else '-'} ({a.get('confidence', 0):.2f}) | {b.get('kind', '-') if b else '-'} ({b.get('confidence', 0):.2f}) | {error:.1f} |" if error is not None else f"| `{path.name}` | {'-' if not a else a.get('kind', '-')} | {'-' if not b else b.get('kind', '-')} | - |")
    (args.output / "comparison.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"Compared {len(all_sources)} scans; both providers: {len(both)}; mean corner distance: {mean_error:.1f}px")
    print(f"Report: {args.output / 'comparison.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
