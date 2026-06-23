"""Tests de la métrica F1 de CULane (vroad_mlt.metric).

Usa carriles SINTÉTICOS (líneas verticales) — lógica pura, sin datos de CULane ni GCP.
Porta el espíritu de los tests de LaneTR (perfecto/vacío/espurio/desplazamiento/umbral).
"""

from __future__ import annotations

import numpy as np
import pytest

from vroad_mlt import metric as M
from vroad_mlt.lines_format import parse as parse_lines


def _vlane(x, y0=100, y1=580, step=20) -> np.ndarray:
    """Una línea vertical en x, de y0 a y1."""
    return np.array([[float(x), float(y)] for y in range(y0, y1, step)], dtype=np.float64)


def _shift(lane, dx) -> np.ndarray:
    out = lane.copy()
    out[:, 0] += dx
    return out


def _lf(lanes, scores=None):
    """Construye un LinesFile (formato común) desde carriles np (coords -> int)."""
    d = {
        "timestamp": 1,
        "Lines": [[{"x": int(round(x)), "y": int(round(y))} for x, y in lane] for lane in lanes],
    }
    if scores is not None:
        d["Scores"] = scores
    return parse_lines(d)


# ----------------------------------------------------------------- núcleo

def test_perfect_prediction_f1_1():
    annos = [[_vlane(300), _vlane(900)], [_vlane(500)]]
    res = M.evaluate(annos, annos)
    assert res.fp == 0 and res.fn == 0
    assert res.tp == 3
    assert res.f1 == pytest.approx(1.0)


def test_empty_prediction_all_fn():
    annos = [[_vlane(300), _vlane(900)]]
    res = M.evaluate([[]], annos)
    assert (res.tp, res.fp, res.fn) == (0, 0, 2)


def test_spurious_prediction_all_fp():
    preds = [[_vlane(300), _vlane(900)]]
    res = M.evaluate(preds, [[]])
    assert (res.tp, res.fp, res.fn) == (0, 2, 0)


def test_small_shift_still_tp():
    annos = [[_vlane(300), _vlane(900)]]
    preds = [[_shift(l, 5) for l in annos[0]]]
    res = M.evaluate(preds, annos)
    assert res.tp == 2
    assert res.f1 == pytest.approx(1.0)


def test_large_shift_breaks_match():
    annos = [[_vlane(300), _vlane(900)]]
    preds = [[_shift(l, 60) for l in annos[0]]]
    res = M.evaluate(preds, annos)
    assert res.tp < 2 and res.fp > 0 and res.fn > 0


def test_stricter_threshold_counts_fewer_tp():
    anno = [_vlane(300)]
    pred = [_shift(anno[0], 5)]  # IoU ~0.7
    assert M.culane_metric(pred, anno, iou_threshold=0.5)[0] == 1
    assert M.culane_metric(pred, anno, iou_threshold=0.95)[0] == 0


def test_interp_densifies_and_short_passthrough():
    pts = _vlane(300, 100, 200, 20)
    assert len(M.interp(pts, n=5)) > len(pts)
    one = np.array([[1.0, 2.0]])
    assert np.array_equal(M.interp(one), one)


def test_result_from_counts():
    r = M.Result.from_counts(3, 1, 1)
    assert r.precision == pytest.approx(0.75)
    assert r.recall == pytest.approx(0.75)
    assert r.f1 == pytest.approx(0.75)


# ------------------------------------------------------- puente formato común

def test_lanes_from_lines():
    lf = parse_lines({"timestamp": 1, "Lines": [[{"x": 1, "y": 2}, {"x": 3, "y": 4}]]})
    arrs = M.lanes_from_lines(lf)
    assert len(arrs) == 1
    assert arrs[0].tolist() == [[1.0, 2.0], [3.0, 4.0]]


def test_predicted_lanes_threshold_and_topk():
    lf = parse_lines(
        {
            "timestamp": 1,
            "Lines": [
                [{"x": 0, "y": 0}, {"x": 0, "y": 10}],
                [{"x": 5, "y": 0}, {"x": 5, "y": 10}],
                [{"x": 9, "y": 0}, {"x": 9, "y": 10}],
            ],
            "Scores": [0.9, 0.3, 0.8],
        }
    )
    kept = M.predicted_lanes(lf, threshold=0.5, max_lanes=4)
    assert len(kept) == 2  # se descarta el 0.3
    assert kept[0][0, 0] == 0 and kept[1][0, 0] == 9  # ordenado por confianza desc
    one = M.predicted_lanes(lf, threshold=0.0, max_lanes=1)
    assert len(one) == 1 and one[0][0, 0] == 0  # tope de 1


def test_predicted_lanes_gt_returns_all():
    lf = parse_lines({"timestamp": 1, "Lines": [[{"x": 0, "y": 0}, {"x": 0, "y": 10}]]})
    assert len(M.predicted_lanes(lf, threshold=0.99)) == 1  # GT sin Scores: no filtra


# ------------------------------------------------------------- score / 9 cats

def test_score_global_and_categories():
    gt0 = _lf([_vlane(300), _vlane(900)])
    gt1 = _lf([_vlane(500)])
    pred0 = _lf([_vlane(300), _vlane(900)], scores=[0.9, 0.8])
    pred1 = _lf([_vlane(500)], scores=[0.95])
    out = M.score([pred0, pred1], [gt0, gt1], categories=["normal", "curve"], threshold=0.5)
    assert out["f1_global"] == pytest.approx(1.0)
    assert out["metrics_by_category"]["normal"] == pytest.approx(1.0)
    assert out["metrics_by_category"]["curve"] == pytest.approx(1.0)
    assert out["threshold"] == 0.5
    assert out["by_category_detail"]["normal"]["TP"] == 2


# -------------------------------------------------------- calibración umbral

def test_calibrate_threshold_drops_low_score_false_positive():
    gt = _lf([_vlane(300)])
    pred = _lf([_vlane(300), _vlane(1200)], scores=[0.9, 0.2])  # 2º es espurio, baja confianza
    best, scores = M.calibrate_threshold([pred], [gt], thresholds=[0.1, 0.5])
    assert scores[0.5].f1 > scores[0.1].f1  # subir el umbral elimina el FP
    assert best == 0.5
