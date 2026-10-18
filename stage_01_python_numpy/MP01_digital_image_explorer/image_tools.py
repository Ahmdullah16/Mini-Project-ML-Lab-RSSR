"""Small, explicit helpers for the MP01 8-bit RGB display-image lesson."""
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps


def require_rgb8(image):
    """Require nonempty H×W×3 uint8 data; return an array without rescaling."""
    array = np.asarray(image)
    if array.dtype != np.uint8:
        raise ValueError('Expected uint8 values in [0, 255]; no automatic scaling.')
    if array.ndim != 3 or array.shape[2] != 3 or min(array.shape[:2]) < 1:
        raise ValueError('Expected a nonempty H×W×3 RGB array.')
    return array


def make_synthetic_image():
    """Return a deterministic 64×96 RGB image: gradients and known color blocks."""
    image = np.zeros((64, 96, 3), dtype=np.uint8)
    image[:, :, 0] = np.rint(np.linspace(0, 255, 96)).astype(np.uint8)
    image[:, :, 1] = np.rint(np.linspace(0, 255, 64))[:, None].astype(np.uint8)
    image[:, :, 2] = 64
    # Six 16×16 blocks: red, green, blue, white, black, neutral gray.
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255),
              (255, 255, 255), (0, 0, 0), (128, 128, 128)]
    for block, color in enumerate(colors):
        image[:16, block * 16:(block + 1) * 16] = color
    return image


def normalize_rgb(image):
    """Map RGB uint8 [0,255] to float32 [0,1]."""
    return require_rgb8(image).astype(np.float32) / 255.0


def to_grayscale(image):
    """Return float32 display-gray [0,255] using BT.601-style luma weights.

    This weights encoded display RGB, not linear-light physical luminance.
    It is not a multispectral reflectance or scientific radiance conversion.
    """
    rgb = require_rgb8(image).astype(np.float32)
    return rgb @ np.array([0.299, 0.587, 0.114], dtype=np.float32)


def histogram_u8(gray):
    """Count each level 0..255 in a nonempty 2D uint8 grayscale array."""
    gray = np.asarray(gray)
    if gray.dtype != np.uint8 or gray.ndim != 2 or gray.size == 0:
        raise ValueError('Expected a nonempty 2D uint8 grayscale array.')
    return np.bincount(gray.ravel(), minlength=256)


def brighten(image, amount):
    """Add an integer in [-255,255], saturating instead of wrapping uint8."""
    image = require_rgb8(image)
    if isinstance(amount, (bool, np.bool_)) or not isinstance(amount, (int, np.integer)):
        raise ValueError('Brightness amount must be an integer in [-255,255].')
    if not -255 <= amount <= 255:
        raise ValueError('Brightness amount must be in [-255,255].')
    return np.clip(image.astype(np.int16) + int(amount), 0, 255).astype(np.uint8)


def load_rgb_image(path):
    """Load the first frame of an 8-bit RGB/L/P display image; apply EXIF orientation.

    Reject scientific/high-bit-depth and alpha-bearing modes rather than silently
    discarding bands, precision, or transparency. Palette transparency is rejected.
    GeoTIFF metadata is not preserved: use a display export for this lesson.
    """
    with Image.open(Path(path)) as source:
        if source.mode not in ('RGB', 'L', 'P') or 'transparency' in source.info:
            raise ValueError('Use an opaque 8-bit RGB, grayscale, or palette display image; '
                             'export scientific or transparent data explicitly first.')
        oriented = ImageOps.exif_transpose(source)
        return np.array(oriented.convert('RGB'), dtype=np.uint8, copy=True)
