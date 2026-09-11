from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps


class ImageInputError(ValueError):
    code = "INVALID_IMAGE"


class UnsupportedFormatError(ImageInputError):
    code = "UNSUPPORTED_FORMAT"


class ImageLimitError(ImageInputError):
    code = "IMAGE_LIMIT_EXCEEDED"


@dataclass(slots=True)
class DecodedImage:
    bgr: np.ndarray
    image_width: int
    image_height: int
    scale_x_to_original: float
    scale_y_to_original: float
    coordinate_space: str = "exif_oriented_pixels"
    format: str = "unknown"


SUPPORTED_FORMATS = {"JPEG", "PNG", "WEBP"}


def decode_image(
    path: str | Path,
    *,
    max_file_bytes: int,
    max_pixels: int,
    max_long_edge: int,
) -> DecodedImage:
    path = Path(path)
    if path.stat().st_size > max_file_bytes:
        raise ImageLimitError(f"file exceeds {max_file_bytes} bytes")

    with Image.open(path) as raw:
        fmt = (raw.format or "").upper()
        if fmt not in SUPPORTED_FORMATS:
            raise UnsupportedFormatError(fmt or "unknown")
        if fmt == "WEBP" and getattr(raw, "n_frames", 1) != 1:
            raise UnsupportedFormatError("animated WebP is not supported")
        w0, h0 = raw.size
        if w0 * h0 > max_pixels:
            raise ImageLimitError(f"decoded dimensions exceed {max_pixels} pixels")

        img = ImageOps.exif_transpose(raw)
        if img.mode in {"RGBA", "LA"} or (img.mode == "P" and "transparency" in img.info):
            rgba = img.convert("RGBA")
            bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            img = Image.alpha_composite(bg, rgba).convert("RGB")
        else:
            img = img.convert("RGB")

        image_width, image_height = img.size
        scale = min(1.0, float(max_long_edge) / max(image_width, image_height))
        if scale < 1.0:
            work_w = max(1, int(round(image_width * scale)))
            work_h = max(1, int(round(image_height * scale)))
            img = img.resize((work_w, work_h), Image.Resampling.LANCZOS)
        work = np.asarray(img, dtype=np.uint8)

    # RTMLib's current YOLOX/RTMPose path expects OpenCV-style BGR input.
    bgr = cv2.cvtColor(work, cv2.COLOR_RGB2BGR)
    work_h, work_w = bgr.shape[:2]
    return DecodedImage(
        bgr=bgr,
        image_width=image_width,
        image_height=image_height,
        scale_x_to_original=image_width / float(work_w),
        scale_y_to_original=image_height / float(work_h),
        format=fmt,
    )
