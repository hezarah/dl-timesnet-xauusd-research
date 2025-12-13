from __future__ import annotations
import numpy as np

def make_windows(series: np.ndarray, window_length: int, horizon: int, stride: int):
    """Create sliding windows X and corresponding future targets y_raw (future return).
    series: 1D array of close prices.
    Returns: X shape (N, window_length), y_raw shape (N,)
    """
    n = len(series)
    xs = []
    ys = []
    for start in range(0, n - window_length - horizon + 1, stride):
        end = start + window_length
        fut = end + horizon - 1
        xs.append(series[start:end])
        ys.append((series[fut] - series[end-1]) / (series[end-1] + 1e-12))
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)
