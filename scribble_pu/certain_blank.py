"""Verified Certain-Blank Anchor Mask Builder for ScribblePU.

Why this module exists:
  In Positive-Unlabeled (PU) ink detection, a small certain-blank anchor set
  (e.g. 1-2 tiles per batch at weight 1/12 to 1/6) anchors the negative floor
  y_lo and prevents background drift. However, scroll volumes that look mostly
  blank often still contain text regions (for example, PHerc0172 segments w062,
  w078, w079, w080 contain 21-30% labeled ink). Treating full tiles from such
  segments as pure negatives actively suppresses real ink.

  `build_certain_blank_mask` extracts ONLY verified blank papyrus pixels:
      certain_blank = supervised & (~ink) & (EDT(~ink) >= margin_px)
  taking the maximum over depth slices when 3D label stacks are supplied.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi


def build_certain_blank_mask(
    ink_labels: np.ndarray,
    supervision_mask: np.ndarray,
    margin_px: int = 16,
) -> np.ndarray:
    """Return a 2D boolean mask of verified blank papyrus pixels.

    Args:
        ink_labels: 2D (H, W) or 3D (Z, H, W) ink label array (> 0 = ink).
        supervision_mask: 2D (H, W) or 3D (Z, H, W) supervision mask (> 0 = reviewed).
        margin_px: Minimum Euclidean distance (in pixels) from any labeled ink pixel.

    Returns:
        2D boolean mask (H, W) that is True only on reviewed non-ink pixels at
        least `margin_px` away from every labeled ink pixel.
    """
    ink = np.asarray(ink_labels)
    sup = np.asarray(supervision_mask)
    if ink.ndim == 3:
        ink2d = ink.max(axis=0) > 0
    elif ink.ndim == 2:
        ink2d = ink > 0
    else:
        raise ValueError(f"Expected 2D or 3D ink_labels, got shape {ink.shape}")

    if sup.ndim == 3:
        sup2d = sup.max(axis=0) > 0
    elif sup.ndim == 2:
        sup2d = sup > 0
    else:
        raise ValueError(f"Expected 2D or 3D supervision_mask, got shape {sup.shape}")

    if ink2d.shape != sup2d.shape:
        raise ValueError(f"Shape mismatch: ink {ink2d.shape} vs supervision {sup2d.shape}")

    if ink2d.any():
        dist = ndi.distance_transform_edt(~ink2d)
        far_from_ink = dist >= float(margin_px)
    else:
        far_from_ink = np.ones_like(ink2d, dtype=bool)

    return sup2d & (~ink2d) & far_from_ink
