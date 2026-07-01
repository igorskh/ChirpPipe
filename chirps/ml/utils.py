import logging
import os
import shutil
import tempfile
import urllib.request

import numpy as np

from typing import Tuple, List


def download_file(url: str, dst: str) -> bool:
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, dir=os.path.dirname(dst)) as tmp:
            tmp_path = tmp.name
            with urllib.request.urlopen(url) as r:
                shutil.copyfileobj(r, tmp)
        os.replace(tmp_path, dst)
        return True
    except Exception as e:
        if tmp is not None:
            try:
                os.remove(tmp.name)
            except Exception:
                pass
        logging.error(f"Error downloading {url} -> {dst}: {e}")
        return False


def chunk_audio(y: np.ndarray, chunk_length: float, overlap: float = 0.0, sr: int = 32000) -> Tuple[np.ndarray, List[Tuple[float, float]]]:
    """
    Split audio into chunks with optional temporal overlap.

    Args:
        y: 1D numpy array (mono audio).
        chunk_length: Length of each chunk in seconds (>0).
        overlap: Overlap between consecutive chunks in seconds (0 <= overlap < chunk_length).
        sr: Sample rate.

    Returns:
        chunks: Float32 array of shape [N, chunk_samples]
        spans: List of (start_sec, end_sec) for each chunk (end_sec truncated to original audio length).
    """
    chunk_len = int(round(chunk_length * sr))
    if chunk_len <= 0:
        raise ValueError("chunk_length must be > 0")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")
    if overlap >= chunk_length:
        raise ValueError("overlap must be < chunk_length")

    step = chunk_len - int(round(overlap * sr))
    if step <= 0:
        raise ValueError(
            "Invalid step size (adjust overlap/chunk_length).")

    n = len(y)
    if n == 0:
        return np.zeros((0, chunk_len), dtype=np.float32), []

    starts = np.arange(0, n, step)
    chunks = []
    spans = []
    for s in starts:
        e = min(s + chunk_len, n)
        seg = y[s:e]
        if len(seg) < chunk_len:
            pad = np.zeros(chunk_len - len(seg), dtype=seg.dtype)
            seg = np.concatenate([seg, pad], axis=0)
        chunks.append(seg.astype(np.float32, copy=False))
        spans.append((s / sr, min(e, n) / sr))
        if e >= n:
            break
    return np.stack(chunks, axis=0), spans
