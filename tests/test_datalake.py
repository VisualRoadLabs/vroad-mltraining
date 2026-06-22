"""Tests de la lógica pura de consultas al Data Lake (vroad_mlt.datalake)."""

from __future__ import annotations

import pytest

from vroad_mlt.datalake import queries as dl
from vroad_mlt.datalake.queries import DatalakeError, build_images_query, build_where, lines_uri_for_image

DLP = "vr-prj-prod-data-v1"


# ------------------------------------------------------- lines_uri_for_image

@pytest.mark.parametrize(
    "image_uri, expected",
    [
        ("gs://bkt-prod-public-usc1/culane/images/000001.jpg",
         "gs://bkt-prod-public-usc1/culane/lines/000001.lines.json"),
        ("gs://bkt-prod-user-usc1/usr_a/sess_1/images/42.jpeg",
         "gs://bkt-prod-user-usc1/usr_a/sess_1/lines/42.lines.json"),
        ("gs://b/d/images/x.PNG", "gs://b/d/lines/x.lines.json"),
    ],
)
def test_lines_uri_for_image(image_uri, expected):
    assert lines_uri_for_image(image_uri) == expected


def test_lines_uri_requires_images_segment():
    with pytest.raises(DatalakeError, match="/images/"):
        lines_uri_for_image("gs://b/d/pics/x.jpg")


def test_lines_uri_requires_known_extension():
    with pytest.raises(DatalakeError, match="extensión"):
        lines_uri_for_image("gs://b/images/x.bmp")


# ----------------------------------------------------------------- build_where

def test_build_where_single_and_multi():
    clauses, params = build_where({"road_geometry": "curve"})
    assert clauses == ["c.road_geometry = @f_road_geometry"]
    assert params == {"f_road_geometry": "curve"}

    clauses, params = build_where({"weather": "rain", "timeofday": "night"})
    assert "c.weather = @f_weather" in clauses and "c.timeofday = @f_timeofday" in clauses
    assert params == {"f_weather": "rain", "f_timeofday": "night"}


def test_build_where_rejects_unknown_column():
    with pytest.raises(DatalakeError, match="no permitida"):
        build_where({"colour": "red"})


# ------------------------------------------------------- build_images_query

def test_images_query_public_no_filters():
    sql, params = build_images_query(DLP, "public", "culane", "train")
    assert "SELECT i.image_id, i.gcs_uri, i.width, i.height" in sql
    assert f"FROM `{DLP}.ds_raw_metadata.tbl_images` i" in sql
    assert "WHERE i.source = @source AND i.dataset = @dataset AND i.split = @split" in sql
    assert "JOIN" not in sql  # sin filtros ni usuario -> sin joins
    assert params == {"source": "public", "dataset": "culane", "split": "train"}


def test_images_query_public_with_filters_joins_classifications():
    sql, params = build_images_query(DLP, "public", "culane", "train", {"road_geometry": "curve"})
    assert f"JOIN `{DLP}.ds_classification.tbl_classifications` c ON c.image_id = i.image_id" in sql
    assert "c.road_geometry = @f_road_geometry" in sql
    assert params["f_road_geometry"] == "curve"
    assert "tbl_label_review_status" not in sql  # público: sin review


def test_images_query_user_left_joins_review_keeps_unreviewed_and_reviewed():
    sql, params = build_images_query(DLP, "user", "user", "train")
    # LEFT JOIN para poder conservar las que NO están en revisión (bien anotadas).
    assert f"LEFT JOIN `{DLP}.ds_label_review.tbl_label_review_status` r ON r.image_id = i.image_id" in sql
    assert "(r.image_id IS NULL OR r.status = @review_status)" in sql
    assert params["review_status"] == "reviewed"


def test_images_query_user_with_filters_has_both_joins():
    sql, _ = build_images_query(DLP, "user", "user", "train", {"weather": "rain"})
    assert "tbl_classifications` c" in sql
    assert "tbl_label_review_status` r" in sql


# ----------------------------------------------- build_distinct_values_query

def test_distinct_values_query():
    sql = dl.build_distinct_values_query(DLP, "road_geometry")
    assert sql == (
        "SELECT DISTINCT road_geometry AS value "
        f"FROM `{DLP}.ds_classification.tbl_classifications` "
        "WHERE road_geometry IS NOT NULL ORDER BY road_geometry"
    )


def test_distinct_values_query_rejects_unknown_column():
    with pytest.raises(DatalakeError):
        dl.build_distinct_values_query(DLP, "speed")


# ----------------------------------------------------------------- dl_fqn

def test_dl_fqn():
    assert dl.dl_fqn(DLP, dl.DL_IMAGES) == f"{DLP}.ds_raw_metadata.tbl_images"
