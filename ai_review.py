#!/usr/bin/env python3
"""Review difficult book scans with a vision model.

The model returns geometry only. The original pixels remain local and are
cropped by cutout_all.py after the response has been reviewed.

Usage:
    export OPENAI_API_KEY=...
    python ai_review.py /Users/.../scanned

The output is JSONL so individual results can be retried or audited without
repeating successful requests.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def load_dotenv(path: Path = ROOT / ".env") -> None:
    """Load simple KEY=VALUE entries without overwriting shell variables."""
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip()
        if value[:1] in {"'", '"'} and value[-1:] == value[:1]:
            value = value[1:-1]
        os.environ.setdefault(key, value)


load_dotenv()
OPENAI_MODEL = os.environ.get("BOOKFIXER_OPENAI_MODEL", "gpt-5.6-sol")
ANTHROPIC_MODEL = os.environ.get("BOOKFIXER_ANTHROPIC_MODEL", "claude-sonnet-4-6")
EXTS = {".jpg", ".jpeg", ".png", ".webp"}
FAMILY_PREFIXES = (
    "Monografieën_over_filmkunst",
    "schwarzkogler_",
    "ishimoto_chicago",
)

SYSTEM_PROMPT = """You are a meticulous image-geometry reviewer for a book-scan cropper.
Your only job is to locate the physical boundary of the book or book part in
each scan. Do not identify the artwork, infer a rectangular design area, or
crop to printed content. The target is the outermost physical cover/spine
surface, including blank margins, but excluding scanner bed, cast shadow,
loose straps, ribbons, fingers, and unrelated objects.

The scans may show a front cover, back cover, or spine. They may be dark on a
dark background, low contrast, rotated, skewed, partly shadowed, or extremely
narrow. Use the scanner/background transition, parallel cover edges, binding
edge, and repeated geometry across context images. A shadow is not a book edge.
For a spine, return the four corners of the spine itself, not the whole book.
If an edge is genuinely invisible, estimate it from the other edges and the
book's perspective, but lower confidence.

Coordinates are pixel coordinates in the original primary image, with (0,0)
at the top-left. Return corners clockwise starting at the top-left corner of
the detected book part. Do not return a bounding box when the object is tilted.
Do not invent precision: confidence must reflect uncertainty.

Respond with exactly one compact JSON object and nothing else. Do not explain
your reasoning, do not use Markdown fences, and do not reconsider the answer.
"""

SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "image": {"type": "string"},
        "width": {"type": "integer"},
        "height": {"type": "integer"},
        "kind": {"type": "string", "enum": ["front", "back", "spine", "other"]},
        "corners": {
            "type": "array",
            "minItems": 4,
            "maxItems": 4,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {"x": {"type": "number"}, "y": {"type": "number"}},
                "required": ["x", "y"],
            },
        },
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "uncertainty": {"type": "string"},
    },
    "required": ["image", "width", "height", "kind", "corners", "confidence", "uncertainty"],
}


def image_size(path: Path) -> tuple[int, int]:
    """Read JPEG/PNG dimensions without importing the fragile CV environment."""
    with path.open("rb") as f:
        data = f.read(32)
    if data[:2] == b"\xff\xd8":
        with path.open("rb") as f:
            f.read(2)
            while True:
                marker = f.read(1)
                if not marker:
                    raise ValueError("invalid JPEG")
                if marker != b"\xff":
                    continue
                while marker == b"\xff":
                    marker = f.read(1)
                if marker in {bytes([0xC0 + i]) for i in range(16) if i not in {4, 8, 12}}:
                    length = int.from_bytes(f.read(2), "big")
                    f.read(1)
                    h = int.from_bytes(f.read(2), "big")
                    w = int.from_bytes(f.read(2), "big")
                    return w, h
                length = int.from_bytes(f.read(2), "big")
                f.seek(length - 2, 1)
    if data.startswith(b"\x89PNG"):
        return int.from_bytes(data[16:20], "big"), int.from_bytes(data[20:24], "big")
    raise ValueError(f"unsupported image format: {path}")


def family_key(path: Path) -> str:
    stem = path.stem
    match = re.match(r"^(.*)_\d+$", stem)
    return match.group(1) if match else stem


def selected_files(source: Path) -> list[Path]:
    return sorted(
        p for p in source.iterdir()
        if p.suffix.lower() in EXTS and any(p.stem.startswith(prefix) for prefix in FAMILY_PREFIXES)
    )


def context_for(primary: Path, files: list[Path]) -> list[Path]:
    peers = [p for p in files if p != primary and family_key(p) == family_key(primary)]
    return sorted(peers, key=lambda p: abs(len(p.stem) - len(primary.stem)))[:2]


def data_url(path: Path) -> str:
    media = {".png": "image/png", ".webp": "image/webp"}.get(path.suffix.lower(), "image/jpeg")
    return f"data:{media};base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def parse_result(text: str, primary: Path, context: list[Path], model: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0]
    try:
        decoder = json.JSONDecoder()
        result = None
        for offset, char in enumerate(text):
            if char != "{":
                continue
            try:
                candidate, _ = decoder.raw_decode(text[offset:])
            except json.JSONDecodeError:
                continue
            if isinstance(candidate, dict) and "corners" in candidate:
                result = candidate
                break
            if isinstance(candidate, dict) and all(name in candidate for name in ("top_left", "top_right", "bottom_right", "bottom_left")):
                result = {"corners": {name: candidate[name] for name in ("top_left", "top_right", "bottom_right", "bottom_left")}, **candidate}
                break
        if result is None:
            raise json.JSONDecodeError("no result object", text, 0)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Model returned invalid JSON: {text[:1000]!r}") from exc
    # Anthropic may follow the geometry instruction while choosing a slightly
    # different field shape; normalize it before writing the shared JSONL.
    corners = result.get("corners")
    if isinstance(corners, dict):
        result["corners"] = [corners[name] for name in ("top_left", "top_right", "bottom_right", "bottom_left")]
    result.setdefault("image", primary.name)
    result.setdefault("width", image_size(primary)[0])
    result.setdefault("height", image_size(primary)[1])
    result.setdefault("kind", "spine" if "spine" in result.get("notes", "").lower() else "front")
    result.setdefault("uncertainty", result.pop("notes", ""))
    result["source"] = str(primary)
    result["context"] = [p.name for p in context]
    result["model"] = model
    return result


def post_json(request: urllib.request.Request) -> dict:
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise ValueError(f"HTTP {exc.code}: {detail}") from exc


def review_openai(primary: Path, context: list[Path]) -> dict:
    width, height = image_size(primary)
    content = [
        {"type": "input_text", "text": f"PRIMARY IMAGE: {primary.name}\nThis is the only image whose coordinates you must return."},
        {"type": "input_image", "image_url": data_url(primary), "detail": "original"},
    ]
    for peer in context:
        content.extend([
            {"type": "input_text", "text": f"CONTEXT ONLY: {peer.name}. Do not return coordinates for this image."},
            {"type": "input_image", "image_url": data_url(peer), "detail": "high"},
        ])
    content.append({"type": "input_text", "text": f"Review {primary.name}. The primary image is {width} x {height} pixels."})
    payload = {
        "model": OPENAI_MODEL,
        "input": [
            {"role": "system", "content": [{"type": "input_text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": content},
        ],
        "reasoning": {"effort": "low"},
        "max_output_tokens": 1200,
        "text": {"format": {"type": "json_schema", "name": "book_edges", "strict": True, "schema": SCHEMA}},
    }
    request = urllib.request.Request(
        "https://api.openai.com/v1/responses",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}", "Content-Type": "application/json"},
        method="POST",
    )
    body = post_json(request)
    text = body.get("output_text")
    if not text:
        for item in body.get("output", []):
            for part in item.get("content", []):
                if part.get("type") == "output_text":
                    text = part.get("text")
                    break
    if not text:
        raise ValueError(f"API response has no output text: {body}")
    return parse_result(text, primary, context, OPENAI_MODEL)


def review_anthropic(primary: Path, context: list[Path]) -> dict:
    width, height = image_size(primary)
    content = []
    for label, path in [("PRIMARY IMAGE", primary)] + [("CONTEXT ONLY", p) for p in context]:
        media = {".png": "image/png", ".webp": "image/webp"}.get(path.suffix.lower(), "image/jpeg")
        content.extend([
            {"type": "text", "text": f"{label}: {path.name}"},
            {"type": "image", "source": {"type": "base64", "media_type": media, "data": base64.b64encode(path.read_bytes()).decode("ascii")}},
        ])
    content.append({"type": "text", "text": f"Review {primary.name}. It is {width} x {height} pixels. Return only the JSON object required by the instructions."})
    payload = {
        "model": ANTHROPIC_MODEL,
        "max_tokens": 1800,
        "system": SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": content}],
    }
    request = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=json.dumps(payload).encode("utf-8"),
        headers={"x-api-key": os.environ["ANTHROPIC_API_KEY"], "anthropic-version": "2023-06-01", "Content-Type": "application/json"},
        method="POST",
    )
    body = post_json(request)
    text = "".join(part.get("text", "") for part in body.get("content", []) if part.get("type") == "text")
    if not text:
        raise ValueError(f"API response has no output text: {body}")
    return parse_result(text, primary, context, ANTHROPIC_MODEL)


def main() -> int:
    ap = argparse.ArgumentParser(description="Ask a vision model for book-boundary corners on selected scan families.")
    ap.add_argument("source", type=Path)
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--limit", type=int, default=0, help="review only the first N images")
    ap.add_argument("--provider", choices=["auto", "openai", "anthropic"], default="auto")
    ap.add_argument("--dry-run", action="store_true", help="list selected scans without calling the API")
    args = ap.parse_args()
    if args.provider == "auto":
        provider = "openai" if os.environ.get("OPENAI_API_KEY") else "anthropic"
    else:
        provider = args.provider
    if not args.dry_run and not os.environ.get(f"{provider.upper()}_API_KEY"):
        ap.error(f"{provider.upper()}_API_KEY is required (or use --provider with the other configured key)")
    files = selected_files(args.source)
    if args.limit:
        files = files[:args.limit]
    out = args.output or args.source / "ai_review" / "reviews.jsonl"
    if args.dry_run:
        model = OPENAI_MODEL if provider == "openai" else ANTHROPIC_MODEL
        print(f"Selected {len(files)} scans with {provider}/{model}")
        for path in files:
            print(f"{path.name}  context={','.join(p.name for p in context_for(path, files)) or '-'}")
        return 0
    out.parent.mkdir(parents=True, exist_ok=True)
    model = OPENAI_MODEL if provider == "openai" else ANTHROPIC_MODEL
    print(f"Reviewing {len(files)} selected scans with {provider}/{model}; writing {out}")
    completed = set()
    if out.exists():
        for line in out.read_text(encoding="utf-8").splitlines():
            try:
                item = json.loads(line)
                if item.get("kind") and isinstance(item.get("corners"), list):
                    completed.add(item.get("source"))
            except json.JSONDecodeError:
                continue
    with out.open("a", encoding="utf-8") as stream:
        for index, primary in enumerate(files, 1):
            if str(primary) in completed:
                print(f"[{index}/{len(files)}] SKIP {primary.name}: already reviewed")
                continue
            try:
                context = context_for(primary, files)
                result = review_openai(primary, context) if provider == "openai" else review_anthropic(primary, context)
                stream.write(json.dumps(result, ensure_ascii=False) + "\n")
                stream.flush()
                print(f"[{index}/{len(files)}] {primary.name}: {result.get('kind', '?')} confidence={result.get('confidence', 0):.2f}")
            except (OSError, ValueError, urllib.error.HTTPError) as exc:
                print(f"[{index}/{len(files)}] FAIL {primary.name}: {exc}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
