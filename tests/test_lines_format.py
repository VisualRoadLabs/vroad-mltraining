"""Tests del formato común `.lines.json` (vroad_mlt.lines_format)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vroad_mlt import lines_format as lf
from vroad_mlt.lines_format import LinesFile, LinesFormatError, Point

FIXTURES = Path(__file__).parent / "fixtures"


# --------------------------------------------------------------------- válidos

def test_parse_gt_culane_example():
    obj = json.loads((FIXTURES / "culane_gt.lines.json").read_text(encoding="utf-8"))
    f = lf.parse(obj)
    assert isinstance(f, LinesFile)
    assert f.timestamp == 1781646771
    assert f.num_lanes == 4
    assert f.scores is None
    assert f.is_prediction is False
    # primer punto del primer carril sale del marco por la izquierda (x < 0)
    assert f.lines[0][0] == Point(-11, 550)


def test_parse_prediction_example_has_aligned_scores():
    obj = json.loads((FIXTURES / "prediction.lines.json").read_text(encoding="utf-8"))
    f = lf.parse(obj)
    assert f.is_prediction is True
    assert f.scores == [0.98, 0.95, 0.91, 0.62]
    assert len(f.scores) == f.num_lanes


def test_empty_lines_is_valid():
    f = lf.parse({"timestamp": 1, "Lines": []})
    assert f.num_lanes == 0
    assert f.is_prediction is False


def test_empty_lines_with_empty_scores_is_valid():
    f = lf.parse({"timestamp": 1, "Lines": [], "Scores": []})
    assert f.is_prediction is True
    assert f.scores == []


def test_more_than_max_lanes_is_not_rejected():
    # MAX_LANES es tope de consumo, no regla de formato.
    lane = [{"x": 0, "y": 10}, {"x": 5, "y": 0}]
    f = lf.parse({"timestamp": 1, "Lines": [lane] * (lines_max := lf.MAX_LANES + 1)})
    assert f.num_lanes == lines_max


def test_int_scores_are_coerced_to_float():
    f = lf.parse({"timestamp": 1, "Lines": [[{"x": 0, "y": 1}, {"x": 1, "y": 0}]], "Scores": [1]})
    assert f.scores == [1.0]
    assert isinstance(f.scores[0], float)


# ------------------------------------------------------------------ round-trip

def test_roundtrip_to_dict_and_back():
    obj = json.loads((FIXTURES / "prediction.lines.json").read_text(encoding="utf-8"))
    f = lf.parse(obj)
    assert lf.parse(lf.to_dict(f)) == f


def test_to_dict_omits_scores_for_gt():
    obj = json.loads((FIXTURES / "culane_gt.lines.json").read_text(encoding="utf-8"))
    d = lf.to_dict(lf.parse(obj))
    assert "Scores" not in d
    assert list(d.keys()) == ["timestamp", "Lines"]


def test_to_dict_includes_scores_in_order_for_prediction():
    obj = json.loads((FIXTURES / "prediction.lines.json").read_text(encoding="utf-8"))
    d = lf.to_dict(lf.parse(obj))
    assert list(d.keys()) == ["timestamp", "Lines", "Scores"]


def test_dumps_loads_roundtrip():
    obj = json.loads((FIXTURES / "prediction.lines.json").read_text(encoding="utf-8"))
    f = lf.parse(obj)
    assert lf.loads(lf.dumps(f)) == f


def test_dump_load_file_roundtrip(tmp_path):
    obj = json.loads((FIXTURES / "culane_gt.lines.json").read_text(encoding="utf-8"))
    f = lf.parse(obj)
    p = tmp_path / "out.lines.json"
    lf.dump(f, p, indent=2)
    assert lf.load(p) == f


# ----------------------------------------------------------------- inválidos

@pytest.mark.parametrize(
    "bad, msg_part",
    [
        ({"Lines": []}, "timestamp"),
        ({"timestamp": 1.5, "Lines": []}, "entero"),
        ({"timestamp": True, "Lines": []}, "entero"),
        ({"timestamp": "1", "Lines": []}, "entero"),
        ({"timestamp": -1, "Lines": []}, "negativo"),
        ({"timestamp": 1}, "Lines"),
        ({"timestamp": 1, "Lines": {}}, "lista de carriles"),
        ({"timestamp": 1, "Lines": ["x"]}, "lista de puntos"),
        ({"timestamp": 1, "Lines": [[{"x": 0, "y": 0}]]}, "al menos"),
        ({"timestamp": 1, "Lines": [[{"x": 0}, {"x": 1, "y": 1}]]}, "falta"),
        ({"timestamp": 1, "Lines": [[{"x": 0, "y": 0, "z": 9}, {"x": 1, "y": 1}]]}, "no permitida"),
        ({"timestamp": 1, "Lines": [[{"x": 0.5, "y": 0}, {"x": 1, "y": 1}]]}, "x' debe ser un entero"),
        ({"timestamp": 1, "Lines": [[{"x": True, "y": 0}, {"x": 1, "y": 1}]]}, "x' debe ser un entero"),
        ({"timestamp": 1, "Lines": [], "extra": 1}, "no permitidas"),
    ],
)
def test_invalid_inputs_raise(bad, msg_part):
    with pytest.raises(LinesFormatError) as exc:
        lf.parse(bad)
    assert msg_part in str(exc.value)


def test_scores_length_mismatch_raises():
    one_lane = [[{"x": 0, "y": 1}, {"x": 1, "y": 0}]]
    with pytest.raises(LinesFormatError, match="misma longitud"):
        lf.parse({"timestamp": 1, "Lines": one_lane, "Scores": [0.5, 0.6]})


@pytest.mark.parametrize("bad_score", [1.5, -0.1])
def test_scores_out_of_range_raise(bad_score):
    one_lane = [[{"x": 0, "y": 1}, {"x": 1, "y": 0}]]
    with pytest.raises(LinesFormatError, match=r"\[0, 1\]"):
        lf.parse({"timestamp": 1, "Lines": one_lane, "Scores": [bad_score]})


def test_score_not_number_raises():
    one_lane = [[{"x": 0, "y": 1}, {"x": 1, "y": 0}]]
    with pytest.raises(LinesFormatError, match="número"):
        lf.parse({"timestamp": 1, "Lines": one_lane, "Scores": ["x"]})


def test_root_must_be_object():
    with pytest.raises(LinesFormatError, match="objeto JSON"):
        lf.parse([1, 2, 3])  # type: ignore[arg-type]


# --------------------------------------------------------------------- helpers

def test_summary_shape():
    obj = json.loads((FIXTURES / "prediction.lines.json").read_text(encoding="utf-8"))
    s = lf.summary(lf.parse(obj))
    assert s["kind"] == "prediction"
    assert s["num_lanes"] == 4
    assert s["x_range"] == [12, 1639]
    assert s["points_per_lane"] == [2, 2, 2, 2]


def test_validate_returns_none_on_valid():
    assert lf.validate({"timestamp": 1, "Lines": []}) is None
