"""Clarke Error Grid Analysis — zone classification.

The classification logic is taken verbatim from the project specification.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def classify_point(ref: float, pred: float) -> str:
    """Return the Clarke EGA zone ('A'–'E') for a single (reference, predicted) pair."""

    if (ref <= 70 and pred <= 70) or (pred <= 1.2 * ref and pred >= 0.8 * ref):
        return "A"
    elif (ref >= 180 and pred <= 70) or (ref <= 70 and pred >= 180):
        return "E"
    elif ((ref >= 70 and ref <= 290) and pred >= ref + 110) or (
        (ref >= 130 and ref <= 180) and (pred <= (7 / 5) * ref - 182)
    ):
        return "C"
    elif (
        (ref >= 240 and (70 <= pred <= 180))
        or (ref <= 175 / 3 and (70 <= pred <= 180))
        or ((175 / 3 <= ref <= 70) and pred >= (6 / 5) * ref)
    ):
        return "D"
    else:
        return "B"


def classify_array(
    refs: Sequence[float],
    preds: Sequence[float],
) -> tuple[list[str], dict[str, int]]:
    """Classify an array of (reference, predicted) pairs.

    Returns
    -------
    labels : list[str]
        Per-point zone label.
    counts : dict[str, int]
        Count of points in each zone A–E.
    """

    refs_arr = np.asarray(refs, dtype=float)
    preds_arr = np.asarray(preds, dtype=float)
    assert len(refs_arr) == len(preds_arr), "Mismatched lengths"

    labels: list[str] = []
    counts: dict[str, int] = {"A": 0, "B": 0, "C": 0, "D": 0, "E": 0}

    for r, p in zip(refs_arr, preds_arr):
        zone = classify_point(float(r), float(p))
        labels.append(zone)
        counts[zone] += 1

    return labels, counts
