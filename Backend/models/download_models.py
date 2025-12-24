"""
YOLO Model Download Script

Usage (from Backend directory):

    python -m models.download_models           # download default models (yolov8n.pt)
    python -m models.download_models yolov8n   # download specific model
    python -m models.download_models yolov8n yolov8s yolov8m

This uses the Ultralytics YOLO API; calling YOLO("<model>.pt") will
auto-download the weights into the Ultralytics cache directory.
"""

from __future__ import annotations

import argparse
import sys
from typing import List

try:
    from ultralytics import YOLO
except ImportError as exc:  # pragma: no cover - runtime import guard
    print(
        "Error: ultralytics is not installed.\n"
        "Install it first (from Backend directory):\n"
        "    pip install -r requirements.txt\n"
    )
    raise SystemExit(1) from exc


DEFAULT_MODELS: List[str] = ["yolov8n"]  # nano model is enough for this project


def download_model(model_name: str) -> None:
    """
    Download a single YOLOv8 model by name.

    Args:
        model_name: e.g. 'yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x'
                    ('.pt' extension is optional)
    """
    if not model_name.endswith(".pt"):
        model_name = f"{model_name}.pt"

    print(f"Downloading model: {model_name} ...")
    try:
        # This call triggers Ultralytics to download the weights if not present
        YOLO(model_name)
        print(f"✓ Successfully downloaded/verified {model_name}")
    except Exception as exc:  # pragma: no cover - defensive
        print(f"✗ Failed to download {model_name}: {exc}")


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download YOLOv8 models for vehicle detection"
    )
    parser.add_argument(
        "models",
        nargs="*",
        help=(
            "Model names to download (e.g. yolov8n yolov8s). "
            "If omitted, downloads default models: "
            + ", ".join(DEFAULT_MODELS)
        ),
    )
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> None:
    if argv is None:
        argv = sys.argv[1:]

    args = parse_args(argv)
    models = args.models or DEFAULT_MODELS

    print("YOLOv8 model downloader")
    print("-----------------------")
    print(f"Models to download: {', '.join(models)}\n")

    for name in models:
        download_model(name)

    print("\nDone.")


if __name__ == "__main__":  # pragma: no cover
    main()


