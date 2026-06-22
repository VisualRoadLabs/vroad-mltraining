"""vroad_mlt.metric — F1 oficial de CULane (port fiel de LaneTR/CLRNet, == C++).

La "vara de medir" del proyecto. Cada carril se interpola con spline y se rasteriza como
una línea de `width=30` px sobre una máscara a la resolución de origen; se calcula la IoU de
píxeles entre cada predicción y cada GT, se emparejan 1-a-1 con el algoritmo húngaro y un par
cuenta como TP si IoU > umbral (0.5). Validado idéntico al evaluador oficial en C++.

El núcleo (`interp`/`draw_lane`/`cross_iou`/`culane_metric`/`evaluate`/`f1_from_counts`) es un
PORT FIEL de `lanetr/metrics/culane.py`. Encima se añade:
- el puente desde el FORMATO COMÚN `.lines.json` (con umbral de `Scores` + tope de 4 carriles),
- el desglose en las 9 categorías de CULane,
- la calibración offline del umbral (barriendo `Scores` sobre predicciones ya escritas).

Lógica pura (numpy + scipy + cv2), sin GCP. Las coordenadas llegan en píxeles a la resolución
de ORIGEN (CULane 1640×590; usuario 1280×720; etc.); GT y predicción comparten resolución.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Sequence

import cv2  # type: ignore[import-untyped]
import numpy as np
from scipy.interpolate import splev, splprep  # type: ignore[import-untyped]
from scipy.optimize import linear_sum_assignment  # type: ignore[import-untyped]

if TYPE_CHECKING:
    from vroad_mlt.lines_format import LinesFile

__all__ = [
    "IMG_SHAPE_CULANE",
    "LANE_WIDTH",
    "IOU_THRESHOLD",
    "MAX_LANES",
    "CULANE_CATEGORIES",
    "Result",
    "interp",
    "draw_lane",
    "cross_iou",
    "f1_from_counts",
    "culane_metric",
    "evaluate",
    "evaluate_by_category",
    "lanes_from_lines",
    "predicted_lanes",
    "score",
    "calibrate_threshold",
]

IMG_SHAPE_CULANE = (590, 1640)  # (H, W) original de CULane
LANE_WIDTH = 30
IOU_THRESHOLD = 0.5
MAX_LANES = 4

# Las 9 categorías oficiales de CULane (las claves estables del desglose).
CULANE_CATEGORIES = (
    "normal", "crowd", "night", "noline", "shadow", "arrow", "dazzle", "curve", "cross",
)

_LaneArray = np.ndarray  # (N, 2) float


# --------------------------------------------------------------- núcleo (port fiel)


def interp(points: _LaneArray, n: int = 5) -> _LaneArray:
    """Densifica un carril con spline (como el C++ oficial y CLRNet)."""
    points = np.asarray(points, dtype=np.float64)
    if len(points) < 2:
        return points
    x, y = points[:, 0], points[:, 1]
    k = min(3, len(points) - 1)
    tck, u = splprep([x, y], s=0, k=k)
    u2 = np.linspace(0.0, 1.0, num=(len(u) - 1) * n + 1)
    return np.asarray(splev(u2, tck)).T  # (M, 2)


def draw_lane(lane: _LaneArray, img_shape=IMG_SHAPE_CULANE, width: int = LANE_WIDTH) -> np.ndarray:
    """Rasteriza un carril como línea de `width` px sobre una máscara uint8."""
    img = np.zeros(img_shape, dtype=np.uint8)
    lane = np.asarray(lane, dtype=np.int32)
    for p1, p2 in zip(lane[:-1], lane[1:]):
        cv2.line(img, tuple(p1), tuple(p2), color=255, thickness=width)
    return img


def cross_iou(xs, ys, width: int = LANE_WIDTH, img_shape=IMG_SHAPE_CULANE) -> np.ndarray:
    """Matriz de IoU (px) entre cada carril de `xs` y cada carril de `ys`."""
    xm = [draw_lane(l, img_shape, width) > 0 for l in xs]
    ym = [draw_lane(l, img_shape, width) > 0 for l in ys]
    ious = np.zeros((len(xm), len(ym)))
    for i, x in enumerate(xm):
        for j, y in enumerate(ym):
            union = (x | y).sum()
            ious[i, j] = float((x & y).sum()) / union if union > 0 else 0.0
    return ious


def f1_from_counts(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def culane_metric(
    pred, anno, width: int = LANE_WIDTH, iou_threshold: float = IOU_THRESHOLD, img_shape=IMG_SHAPE_CULANE
) -> tuple[int, int, int]:
    """TP/FP/FN de UNA imagen. `pred`/`anno`: listas de carriles (N,2) en coords de origen."""
    if len(pred) == 0:
        return (0, 0, len(anno))
    if len(anno) == 0:
        return (0, len(pred), 0)

    interp_pred = [interp(p, n=5) for p in pred]
    interp_anno = [interp(a, n=5) for a in anno]
    ious = cross_iou(interp_pred, interp_anno, width, img_shape)
    row, col = linear_sum_assignment(1 - ious)
    tp = int((ious[row, col] > iou_threshold).sum())
    return (tp, len(pred) - tp, len(anno) - tp)


# ------------------------------------------------------------------ resultados


@dataclass(frozen=True)
class Result:
    """TP/FP/FN + Precision/Recall/F1 de un conjunto de imágenes."""

    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1: float

    @classmethod
    def from_counts(cls, tp: int, fp: int, fn: int) -> "Result":
        p, r, f = f1_from_counts(tp, fp, fn)
        return cls(tp=tp, fp=fp, fn=fn, precision=p, recall=r, f1=f)

    def as_dict(self) -> dict:
        return {
            "TP": self.tp, "FP": self.fp, "FN": self.fn,
            "precision": self.precision, "recall": self.recall, "f1": self.f1,
        }


def evaluate(
    preds: Sequence[Sequence[_LaneArray]],
    annos: Sequence[Sequence[_LaneArray]],
    *,
    width: int = LANE_WIDTH,
    iou_threshold: float = IOU_THRESHOLD,
    img_shape=IMG_SHAPE_CULANE,
) -> Result:
    """Agrega TP/FP/FN sobre un conjunto de imágenes -> `Result` global."""
    tp = fp = fn = 0
    for pred, anno in zip(preds, annos):
        a, b, c = culane_metric(pred, anno, width, iou_threshold, img_shape)
        tp += a; fp += b; fn += c
    return Result.from_counts(tp, fp, fn)


def evaluate_by_category(
    preds: Sequence[Sequence[_LaneArray]],
    annos: Sequence[Sequence[_LaneArray]],
    categories: Sequence[str],
    *,
    width: int = LANE_WIDTH,
    iou_threshold: float = IOU_THRESHOLD,
    img_shape=IMG_SHAPE_CULANE,
) -> tuple[Result, dict[str, Result]]:
    """Cuenta por imagen UNA vez y agrega global + por categoría (de `categories[i]`)."""
    if len(categories) != len(preds):
        raise ValueError("categories debe tener una etiqueta por imagen")
    counts = [culane_metric(p, a, width, iou_threshold, img_shape) for p, a in zip(preds, annos)]
    g = [sum(c[k] for c in counts) for k in range(3)]
    overall = Result.from_counts(*g)
    by_cat: dict[str, Result] = {}
    for cat in dict.fromkeys(categories):  # orden de aparición, sin duplicados
        idx = [i for i, c in enumerate(categories) if c == cat]
        s = [sum(counts[i][k] for i in idx) for k in range(3)]
        by_cat[cat] = Result.from_counts(*s)
    return overall, by_cat


# ----------------------------------------------------- puente al formato común


def lanes_from_lines(lf: "LinesFile") -> list[_LaneArray]:
    """`LinesFile` (formato común) -> lista de carriles (N,2) float."""
    return [np.array([[p.x, p.y] for p in lane], dtype=np.float64) for lane in lf.lines]


def predicted_lanes(lf: "LinesFile", *, threshold: float = 0.0, max_lanes: int = MAX_LANES) -> list[_LaneArray]:
    """Carriles de una predicción tras filtrar por `Scores>=threshold` y quedarse el top-`max_lanes`.

    NMS-free: ordena por confianza y aplica el tope. Si no hay `Scores` (es un GT), devuelve
    todos los carriles sin filtrar.
    """
    arrays = lanes_from_lines(lf)
    if lf.scores is None:
        return arrays
    kept = [(s, a) for s, a in zip(lf.scores, arrays) if s >= threshold]
    kept.sort(key=lambda sa: -sa[0])
    return [a for _, a in kept[:max_lanes]]


def score(
    pred_files: Sequence["LinesFile"],
    gt_files: Sequence["LinesFile"],
    categories: Optional[Sequence[str]] = None,
    *,
    threshold: float = 0.0,
    max_lanes: int = MAX_LANES,
    iou_threshold: float = IOU_THRESHOLD,
    img_shape=IMG_SHAPE_CULANE,
    benchmark: str = "culane@v1",
) -> dict:
    """Puntúa predicciones vs GT (formato común) -> el contenido de `results.json`.

    Devuelve f1_global + (si hay `categories`) f1 por categoría + el desglose completo.
    """
    preds = [predicted_lanes(p, threshold=threshold, max_lanes=max_lanes) for p in pred_files]
    annos = [lanes_from_lines(g) for g in gt_files]

    if categories is not None:
        overall, by_cat = evaluate_by_category(
            preds, annos, categories, iou_threshold=iou_threshold, img_shape=img_shape
        )
    else:
        overall = evaluate(preds, annos, iou_threshold=iou_threshold, img_shape=img_shape)
        by_cat = {}

    return {
        "benchmark": benchmark,
        "f1_global": overall.f1,
        "precision": overall.precision,
        "recall": overall.recall,
        "threshold": threshold,
        "metrics_by_category": {c: r.f1 for c, r in by_cat.items()},
        "by_category_detail": {c: r.as_dict() for c, r in by_cat.items()},
    }


def calibrate_threshold(
    pred_files: Sequence["LinesFile"],
    gt_files: Sequence["LinesFile"],
    thresholds: Sequence[float],
    *,
    max_lanes: int = MAX_LANES,
    iou_threshold: float = IOU_THRESHOLD,
    img_shape=IMG_SHAPE_CULANE,
) -> tuple[float, dict[float, Result]]:
    """Barre umbrales de confianza sobre predicciones YA escritas -> (mejor_umbral, scores).

    Offline: no reejecuta inferencia, solo refiltra `Scores`. Elige el de mayor F1.
    """
    annos = [lanes_from_lines(g) for g in gt_files]
    scores: dict[float, Result] = {}
    for t in thresholds:
        preds = [predicted_lanes(p, threshold=float(t), max_lanes=max_lanes) for p in pred_files]
        scores[float(t)] = evaluate(preds, annos, iou_threshold=iou_threshold, img_shape=img_shape)
    best = max(scores, key=lambda t: scores[t].f1)
    return best, scores
