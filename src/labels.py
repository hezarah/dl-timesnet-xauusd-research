from __future__ import annotations
import numpy as np

def yraw_to_3class(y_raw: np.ndarray, threshold: float) -> np.ndarray:
    """Convert future returns to 3 classes:
    Down -> 0, Flat -> 1, Up -> 2
    """
    y = np.zeros_like(y_raw, dtype=np.int64) + 1
    y[y_raw > threshold] = 2
    y[y_raw < -threshold] = 0
    return y
