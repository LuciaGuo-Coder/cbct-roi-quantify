"""
Placeholder for task 1 ROI core algorithm.

Task 2 imports get_ring_roi from this module. Replace this file with the
official task 1 implementation when it is ready, keeping the function signature:

    get_ring_roi(mask_path, ct_path, expand_pixels) -> (mean_ct_value, ring_mask)
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


class RingROIError(ValueError):
    """Algorithm-level error that the API layer can map to response codes."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def get_ring_roi(mask_path: str | Path, ct_path: str | Path, expand_pixels: int) -> tuple[float, np.ndarray]:
    """
    Placeholder implementation of ring ROI quantification.

    The ring mask is defined as: dilated(mask) - original(mask).
    Returns the average CT value inside the ring and the ring mask as a uint8
    numpy array with values 0/255.
    """
    if expand_pixels < 1:
        raise RingROIError("INVALID_PARAM", "expand_pixels must be at least 1")

    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    ct = cv2.imread(str(ct_path), cv2.IMREAD_GRAYSCALE)

    if mask is None or ct is None:
        raise RingROIError("INVALID_IMAGE", "mask or CT image cannot be read")
    if mask.shape != ct.shape:
        raise RingROIError("INVALID_IMAGE", f"mask shape {mask.shape} does not match CT shape {ct.shape}")

    original = (mask > 0).astype(np.uint8)
    if cv2.countNonZero(original) == 0:
        raise RingROIError("EMPTY_MASK", "mask is empty")

    ys, xs = np.where(original > 0)
    height, width = original.shape
    if (
        xs.min() - expand_pixels < 0
        or ys.min() - expand_pixels < 0
        or xs.max() + expand_pixels >= width
        or ys.max() + expand_pixels >= height
    ):
        raise RingROIError("EXPAND_OUT_OF_BOUNDS", "expanded ROI exceeds image boundary")

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * expand_pixels + 1, 2 * expand_pixels + 1),
    )
    dilated = cv2.dilate(original, kernel, iterations=1)
    ring_mask = ((dilated > 0) & (original == 0)).astype(np.uint8) * 255

    if cv2.countNonZero(ring_mask) == 0:
        raise RingROIError("EMPTY_MASK", "ring ROI is empty")

    mean_ct_value = float(cv2.mean(ct, mask=ring_mask)[0])
    return mean_ct_value, ring_mask
