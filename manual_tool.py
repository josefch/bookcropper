#!/usr/bin/env python3
"""Local manual book-scan correction station."""
from __future__ import annotations

import argparse
import io
import json
import mimetypes
import os
import re
import threading
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import numpy as np
from PIL import Image, ImageOps

EXTS = {".jpg", ".jpeg", ".png", ".webp"}
OUTPUT_SCALE = 0.5
DEFAULT_CONFIG_PATH = Path.home() / ".config" / "bookcropper" / "config.json"
DARK_EDGE_THRESHOLD = 40
DARK_EDGE_CLEAN_FRACTION = 0.985
DARK_EDGE_MAX_RATIO = 0.025
DARK_EDGE_MAX_PIXELS = 64
DARK_EDGE_STABLE_LINES = 3


class RequestError(ValueError):
    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


def list_images(source: Path) -> list[str]:
    return sorted(
        str(path.relative_to(source))
        for path in source.rglob("*")
        if path.is_file() and path.suffix.lower() in EXTS
    )


def load_config(path: Path) -> dict[str, str]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}
    return value if isinstance(value, dict) else {}


def save_config(path: Path, value: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def configured_directory(value: object, label: str, create: bool = False) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise RequestError(f"{label} is required")
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise RequestError(f"{label} must be an absolute path")
    if create:
        path.mkdir(parents=True, exist_ok=True)
    if not path.is_dir():
        raise RequestError(f"{label} is not a directory")
    return path.resolve()


def order_corners(points):
    return [points[i] for i in (0, 1, 2, 3)]


def oriented_image(path: Path) -> Image.Image:
    """Load pixels in the orientation shown by Finder and the browser."""
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


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
    for start, components in ((0, (0, 1, 2)), (1, (1, 0, 2)), (2, (1, 2, 0)),
                              (3, (2, 1, 0)), (4, (2, 0, 1)), (5, (0, 2, 1))):
        mask = (sector >= start) & (sector < start + 1)
        hsv_rgb[mask, components[0]] = chroma[mask]
        hsv_rgb[mask, components[1]] = second[mask]
    hsv_rgb += match[..., None]
    return Image.fromarray(np.uint8(np.clip(hsv_rgb * 255 + 0.5, 0, 255)), "RGB")


@lru_cache(maxsize=2)
def corrected_source(path: str, mtime_ns: int) -> Image.Image:
    with Image.open(path) as image:
        return colorchecker_correction(ImageOps.exif_transpose(image).convert("RGB"))


def _dark_edge_inset(grayscale: np.ndarray, axis: int, reverse: bool = False) -> int:
    light_fraction = (grayscale > DARK_EDGE_THRESHOLD).mean(axis=axis)
    if reverse:
        light_fraction = light_fraction[::-1]
    if not len(light_fraction) or light_fraction[0] >= DARK_EDGE_CLEAN_FRACTION:
        return 0
    maximum = min(
        DARK_EDGE_MAX_PIXELS,
        max(2, round(len(light_fraction) * DARK_EDGE_MAX_RATIO)),
    )
    for inset in range(1, maximum + 1):
        end = inset + DARK_EDGE_STABLE_LINES
        if end <= len(light_fraction) and np.all(
            light_fraction[inset:end] >= DARK_EDGE_CLEAN_FRACTION
        ):
            return inset
    return 0


def trim_dark_scanner_edges(image: Image.Image) -> Image.Image:
    """Remove only thin near-black margins that quickly transition into the book."""
    grayscale = np.asarray(image.convert("L"))
    top = _dark_edge_inset(grayscale, axis=1)
    bottom = _dark_edge_inset(grayscale, axis=1, reverse=True)
    left = _dark_edge_inset(grayscale, axis=0)
    right = _dark_edge_inset(grayscale, axis=0, reverse=True)
    if not any((left, top, right, bottom)):
        return image
    return image.crop((left, top, image.width - right, image.height - bottom))


def render_crop(source: Path, points: list[list[float]], rotation: float = 0,
                correction: bool = True, trim_dark_edges: bool = True) -> Image.Image:
    if correction:
        image = corrected_source(str(source), source.stat().st_mtime_ns)
    else:
        image = oriented_image(source)
    if rotation:
        image = image.rotate(rotation, expand=True)
    src = order_corners(points)
    left = max(0, min(image.width - 1, round(min(p[0] for p in src))))
    top = max(0, min(image.height - 1, round(min(p[1] for p in src))))
    right = max(left + 1, min(image.width, round(max(p[0] for p in src))))
    bottom = max(top + 1, min(image.height, round(max(p[1] for p in src))))
    crop = image.convert("RGB").crop((left, top, right, bottom))
    if trim_dark_edges:
        crop = trim_dark_scanner_edges(crop)
    size = (max(1, round(crop.width * OUTPUT_SCALE)), max(1, round(crop.height * OUTPUT_SCALE)))
    return crop.resize(size, Image.Resampling.LANCZOS)


def save_crop(source: Path, output: Path, points: list[list[float]], rotation: float = 0,
              correction: bool = True, trim_dark_edges: bool = True) -> None:
    crop = render_crop(source, points, rotation, correction, trim_dark_edges)
    output.parent.mkdir(parents=True, exist_ok=True)
    crop.save(output, quality=95, dpi=(150, 150))


def crop_jpeg(source: Path, points: list[list[float]], rotation: float = 0,
              correction: bool = True, trim_dark_edges: bool = True) -> bytes:
    crop = render_crop(source, points, rotation, correction, trim_dark_edges)
    buffer = io.BytesIO()
    crop.save(buffer, format="JPEG", quality=95, dpi=(150, 150))
    return buffer.getvalue()


def warm_detector_model() -> None:
    try:
        from ml_extend import warm_model
        warm_model()
    except Exception as exc:
        print(f"Detector warm-up failed: {exc}")


class Handler(BaseHTTPRequestHandler):
    source: Path
    output: Path
    final_store: Path | None
    config_path: Path
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
        elif parsed.path == "/api/settings":
            self.send_json({
                "sourceDirectory": str(self.source),
                "finalStoreDirectory": str(self.final_store) if self.final_store else "",
                "configPath": str(self.config_path),
            })
        elif parsed.path == "/api/images":
            self.send_json({"images": list_images(self.source)})
        elif parsed.path == "/api/thumbnail":
            rel = unquote(parse_qs(parsed.query).get("path", [""])[0])
            path = (self.source / rel).resolve()
            if not path.is_file() or self.source not in path.parents:
                self.send_error(404)
            else:
                with Image.open(path) as image:
                    image = ImageOps.exif_transpose(image).convert("RGB")
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
                        heuristic_score = re.search(r"^heur \(score=([0-9.]+)", note)
                        if heuristic_score and float(heuristic_score.group(1)) < 0.08:
                            self.send_json({"corners": None, "note": "Low-confidence suggestion; manual crop recommended"})
                            return
                        # The ensemble's detector coordinates are already in the
                        # displayed orientation; only the Pillow image needs EXIF
                        # normalization.
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
                            image = corrected_source(str(path), path.stat().st_mtime_ns)
                        else:
                            image = ImageOps.exif_transpose(image).convert("RGB")
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
        parsed = urlparse(self.path)
        if parsed.path not in ("/api/save", "/api/crop", "/api/finalize", "/api/settings"):
            self.send_error(404)
            return
        try:
            if parsed.path == "/api/settings":
                self.update_settings()
                return
            if parsed.path == "/api/finalize":
                self.finalize_upload(parsed)
                return
            body = json.loads(self.read_body())
            rel = body["path"]
            points = body["corners"]
            rotation = float(body.get("rotation", 0)) % 360
            correction = bool(body.get("correction", True))
            trim_dark_edges = bool(body.get("trimDarkEdges", True))
            if len(points) != 4 or any(len(p) != 2 for p in points):
                raise RequestError("four [x,y] corners required")
            source = self.source_path(rel)
            if parsed.path == "/api/crop":
                with self.lock:
                    data = crop_jpeg(source, points, rotation, correction, trim_dark_edges)
                self.send_bytes(data, "image/jpeg")
                return
            stem = Path(rel).stem
            out = self.output / f"{stem}.jpg"
            sidecar = self.output / f"{stem}.json"
            with self.lock:
                save_crop(source, out, points, rotation, correction, trim_dark_edges)
                sidecar.write_text(json.dumps({"source": rel, "rotation": rotation, "correction": correction, "trimDarkEdges": trim_dark_edges, "corners": points}, indent=2) + "\n")
            self.send_json({"saved": str(out), "sidecar": str(sidecar)})
        except RequestError as exc:
            self.send_json({"error": str(exc)}, exc.status)
        except Exception as exc:
            self.send_json({"error": str(exc)}, 400)

    def read_body(self, max_bytes: int = 30 * 1024 * 1024) -> bytes:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            raise RequestError("request body is required")
        if length > max_bytes:
            raise RequestError("request body is too large", 413)
        return self.rfile.read(length)

    def source_path(self, relative: object) -> Path:
        if not isinstance(relative, str) or not relative:
            raise RequestError("invalid source path")
        path = (self.source / relative).resolve()
        if not path.is_file() or self.source not in path.parents:
            raise RequestError("invalid source path")
        return path

    def update_settings(self) -> None:
        body = json.loads(self.read_body(64 * 1024))
        source = configured_directory(body.get("sourceDirectory"), "Source directory")
        final_value = body.get("finalStoreDirectory")
        final_store = configured_directory(final_value, "Final-store directory", create=True) if final_value else None
        if final_store and (final_store == source or source in final_store.parents):
            raise RequestError("Final-store directory must be outside the source directory")
        value = {
            "sourceDirectory": str(source),
            "finalStoreDirectory": str(final_store) if final_store else "",
        }
        with self.lock:
            save_config(self.config_path, value)
            type(self).source = source
            type(self).final_store = final_store
            images = list_images(source)
        self.send_json({**value, "configPath": str(self.config_path), "images": images})

    def finalize_upload(self, parsed) -> None:
        relative = unquote(parse_qs(parsed.query).get("path", [""])[0])
        data = self.read_body()
        with Image.open(io.BytesIO(data)) as image:
            dpi = image.info.get("dpi", (0, 0))
            if image.format != "JPEG" or not all(145 <= float(value) <= 155 for value in dpi[:2]):
                raise RequestError("final crop must be a 150 DPI JPEG")
            image.verify()
        with self.lock:
            source = self.source_path(relative)
            if not self.final_store:
                raise RequestError("Final-store directory is not configured", 409)
            destination = self.final_store / f"{Path(relative).stem}.jpg"
            if destination.exists():
                raise RequestError(f"Final crop already exists: {destination.name}", 409)
            temporary = destination.with_name(
                f".{destination.name}.{os.getpid()}.{threading.get_ident()}.tmp"
            )
            try:
                with temporary.open("wb") as file:
                    file.write(data)
                    file.flush()
                    os.fsync(file.fileno())
                temporary.replace(destination)
                try:
                    source.unlink()
                except Exception:
                    destination.unlink(missing_ok=True)
                    raise
            finally:
                temporary.unlink(missing_ok=True)
            corrected_source.cache_clear()
            images = list_images(self.source)
        self.send_json({
            "archived": str(destination),
            "removed": relative,
            "images": images,
        })


def main():
    ap = argparse.ArgumentParser(description="Review and perspective-crop book scans locally.")
    ap.add_argument("source", type=Path, nargs="?")
    ap.add_argument("--output", type=Path, default=Path("manual_crops"))
    ap.add_argument("--final-store", type=Path)
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    config_path = args.config.expanduser().resolve()
    config = load_config(config_path)
    source_value = args.source or config.get("sourceDirectory")
    final_store_value = args.final_store or config.get("finalStoreDirectory")
    try:
        Handler.source = configured_directory(str(source_value) if source_value else None, "Source directory")
        Handler.final_store = (
            configured_directory(str(final_store_value), "Final-store directory", create=True)
            if final_store_value else None
        )
    except RequestError as exc:
        ap.error(str(exc))
    if Handler.final_store and (
        Handler.final_store == Handler.source or Handler.source in Handler.final_store.parents
    ):
        ap.error("Final-store directory must be outside the source directory")
    Handler.output = args.output.resolve()
    Handler.config_path = config_path
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    threading.Thread(target=warm_detector_model, daemon=True).start()
    print(f"Book scan station: http://127.0.0.1:{args.port}")
    print(f"Source: {Handler.source}")
    print(f"Output: {Handler.output}")
    print(f"Final store: {Handler.final_store or 'not configured'}")
    print(f"Config: {Handler.config_path}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
