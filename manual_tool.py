#!/usr/bin/env python3
"""Local manual book-scan correction station."""
from __future__ import annotations

import argparse
import io
import json
import mimetypes
import threading
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import numpy as np
from PIL import Image

EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def order_corners(points):
    return [points[i] for i in (0, 1, 2, 3)]


def colorchecker_correction(image: Image.Image) -> Image.Image:
    """Apply the scanner correction estimated from the 24-patch measurement."""
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    value = rgb.max(axis=2)
    minimum = rgb.min(axis=2)
    delta = value - minimum
    saturation = np.zeros_like(value)
    np.divide(delta, value, out=saturation, where=value > 0)
    hue = np.zeros_like(value)
    nonzero = delta > 0
    red, green, blue = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    red_max = (value == red) & nonzero
    green_max = (value == green) & nonzero
    blue_max = (value == blue) & nonzero
    hue[red_max] = ((green[red_max] - blue[red_max]) / delta[red_max]) % 6
    hue[green_max] = (blue[green_max] - red[green_max]) / delta[green_max] + 2
    hue[blue_max] = (red[blue_max] - green[blue_max]) / delta[blue_max] + 4
    hue *= 60
    value = np.clip(value * 1.095, 0, 1)
    adjustments = ((30, 60, 1.30), (60, 165, 1.30), (165, 210, 1.40),
                   (15, 45, 1.20), (285, 345, 1.25), (345, 360, 1.10))
    for start, end, factor in adjustments:
        mask = (hue >= start) & (hue < end)
        saturation[mask] *= factor
    hue[(hue >= 285) & (hue < 345)] -= 10
    hue[(hue >= 165) & (hue < 210)] -= 10
    saturation = np.clip(saturation, 0, 1)
    hue %= 360
    chroma = value * saturation
    sector = hue / 60
    second = chroma * (1 - np.abs(sector % 2 - 1))
    match = value - chroma
    hsv_rgb = np.zeros_like(rgb)
    for start, components in ((0, (0, 1, 2)), (1, (1, 0, 2)), (2, (2, 0, 1)),
                              (3, (2, 1, 0)), (4, (1, 2, 0)), (5, (0, 2, 1))):
        mask = (sector >= start) & (sector < start + 1)
        hsv_rgb[mask, components[0]] = chroma[mask]
        hsv_rgb[mask, components[1]] = second[mask]
    hsv_rgb += match[..., None]
    return Image.fromarray(np.uint8(np.clip(hsv_rgb * 255 + 0.5, 0, 255)), "RGB")


@lru_cache(maxsize=2)
def corrected_source(path: str) -> Image.Image:
    with Image.open(path) as image:
        return colorchecker_correction(image.convert("RGB"))


def save_crop(source: Path, output: Path, points: list[list[float]], rotation: float = 0, correction: bool = True) -> None:
    if correction:
        image = corrected_source(str(source))
    else:
        with Image.open(source) as source_image:
            image = source_image.convert("RGB")
    if rotation:
        image = image.rotate(rotation, expand=True)
        src = order_corners(points)
        left = max(0, min(image.width - 1, round(min(p[0] for p in src))))
        top = max(0, min(image.height - 1, round(min(p[1] for p in src))))
        right = max(left + 1, min(image.width, round(max(p[0] for p in src))))
        bottom = max(top + 1, min(image.height, round(max(p[1] for p in src))))
        crop = image.convert("RGB").crop((left, top, right, bottom))
        output.parent.mkdir(parents=True, exist_ok=True)
        crop.save(output, quality=95)


class Handler(BaseHTTPRequestHandler):
    source: Path
    output: Path
    lock = threading.Lock()

    def log_message(self, fmt, *args):
        pass

    def send_json(self, value, status=200):
        data = json.dumps(value, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self.serve_file(Path(__file__).with_name("manual_tool.html"), "text/html; charset=utf-8")
        elif parsed.path in ("/manual_tool.css", "/manual_tool.js"):
            asset = Path(__file__).with_name(parsed.path.lstrip("/"))
            self.serve_file(asset, "text/css" if asset.suffix == ".css" else "text/javascript")
        elif parsed.path == "/api/images":
            images = sorted(str(p.relative_to(self.source)) for p in self.source.rglob("*") if p.is_file() and p.suffix.lower() in EXTS)
            self.send_json({"images": images})
        elif parsed.path == "/api/thumbnail":
            rel = unquote(parse_qs(parsed.query).get("path", [""])[0])
            path = (self.source / rel).resolve()
            if not path.is_file() or self.source not in path.parents:
                self.send_error(404)
            else:
                with Image.open(path) as image:
                    image = image.convert("RGB")
                    image.thumbnail((180, 180), Image.Resampling.LANCZOS)
                    buffer = io.BytesIO()
                    image.save(buffer, format="JPEG", quality=82)
                    self.send_bytes(buffer.getvalue(), "image/jpeg")
        elif parsed.path == "/api/suggestion":
            rel = unquote(parse_qs(parsed.query).get("path", [""])[0])
            path = (self.source / rel).resolve()
            if not path.is_file() or self.source not in path.parents:
                self.send_error(404)
            else:
                try:
                    import math
                    from unified_detect import detect
                    _, box, note = detect(path)
                    if box is None:
                        self.send_json({"corners": None, "note": note})
                    else:
                        points = box.astype(float).tolist()
                        edges = [(points[0], points[1]), (points[1], points[2])]
                        edge = max(edges, key=lambda pair: abs(pair[1][0] - pair[0][0]))
                        dx = edge[1][0] - edge[0][0]
                        dy = edge[1][1] - edge[0][1]
                        angle = math.degrees(math.atan2(dy, dx))
                        if angle > 90:
                            angle -= 180
                        if angle <= -90:
                            angle += 180
                        self.send_json({"corners": points, "rotation": angle, "note": note})
                except Exception as exc:
                    self.send_json({"corners": None, "note": "suggestion failed: " + str(exc)}, 200)
        elif parsed.path == "/api/image":
            rel = unquote(parse_qs(parsed.query).get("path", [""])[0])
            query = parse_qs(parsed.query)
            rotation = float(query.get("rotate", ["0"])[0]) % 360
            correction = query.get("correct", ["0"])[0] == "1"
            path = (self.source / rel).resolve()
            if not path.is_file() or self.source not in path.parents:
                self.send_error(404)
            else:
                if rotation or correction:
                    with Image.open(path) as image:
                        buffer = io.BytesIO()
                        if correction:
                            image = corrected_source(str(path))
                        else:
                            image = image.convert("RGB")
                        if rotation:
                            image = image.rotate(rotation, expand=True)
                        image.save(buffer, format="JPEG", quality=95)
                        data = buffer.getvalue()
                    self.send_bytes(data, "image/jpeg")
                else:
                    self.serve_file(path, mimetypes.guess_type(path.name)[0] or "application/octet-stream")
        else:
            self.send_error(404)

    def serve_file(self, path, content_type):
        self.send_bytes(path.read_bytes(), content_type)

    def send_bytes(self, data, content_type):
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        if self.path != "/api/save":
            self.send_error(404)
            return
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))))
            rel = body["path"]
            points = body["corners"]
            rotation = float(body.get("rotation", 0)) % 360
            correction = bool(body.get("correction", True))
            if len(points) != 4 or any(len(p) != 2 for p in points):
                raise ValueError("four [x,y] corners required")
            source = (self.source / rel).resolve()
            if not source.is_file() or self.source not in source.parents:
                raise ValueError("invalid source path")
            stem = Path(rel).stem
            out = self.output / f"{stem}.jpg"
            sidecar = self.output / f"{stem}.json"
            with self.lock:
                save_crop(source, out, points, rotation, correction)
                sidecar.write_text(json.dumps({"source": rel, "rotation": rotation, "correction": correction, "corners": points}, indent=2) + "\n")
            self.send_json({"saved": str(out), "sidecar": str(sidecar)})
        except Exception as exc:
            self.send_json({"error": str(exc)}, 400)


def main():
    ap = argparse.ArgumentParser(description="Review and perspective-crop book scans locally.")
    ap.add_argument("source", type=Path)
    ap.add_argument("--output", type=Path, default=Path("manual_crops"))
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    if not args.source.is_dir():
        ap.error(f"not a directory: {args.source}")
    Handler.source = args.source.resolve()
    Handler.output = args.output.resolve()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Book scan station: http://127.0.0.1:{args.port}")
    print(f"Source: {Handler.source}")
    print(f"Output: {Handler.output}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
