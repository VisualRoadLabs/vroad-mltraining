"""vroad_mlt.datalake.queries — consultas (read-only) al Data Lake y rutas.

Lógica COMÚN a `dataset-build` (materializa shards) y al `dashboard` (rellena los
desplegables de filtros): construye las consultas a BigQuery del Data Lake
(`vr-prj-prod-data-v1`) y deriva la ruta del `.lines.json` desde la de la imagen.

Tablas del Data Lake (ver DATALAKE.md):
- `ds_raw_metadata.tbl_images`        (image_id, source, dataset, gcs_uri, width, height, split, ...)
- `ds_classification.tbl_classifications` (image_id, weather, scene, timeofday, road_geometry, ...)
- `ds_label_review.tbl_label_review_status` (image_id, lines_gcs_uri, status, ...)

Diseño:
- Lógica PURA: solo construye SQL parametrizado (sin inyección) y rutas; la ejecución
  la hace el `bq` del llamante (read-only, con los roles del Data Lake).
- Los VALORES de los filtros NO se hardcodean: se descubren con `SELECT DISTINCT`
  (`build_distinct_values_query`), por si cambian. (Las 9 categorías de EVALUACIÓN
  de CULane son otra cosa: fijas, de `categories.json`, no de aquí.)
- Las columnas filtrables sí son un allowlist (van al SQL como identificador, no como
  parámetro), para no permitir inyección por nombre de columna.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Mapping, Optional

if TYPE_CHECKING:
    from vroad_mlt.bq import BigQuery

__all__ = [
    "DatalakeError",
    "FILTERABLE_COLUMNS",
    "REVIEWED_STATUS",
    "DL_IMAGES",
    "DL_CLASSIFICATIONS",
    "DL_REVIEW",
    "dl_fqn",
    "label_uri_for_image",
    "build_where",
    "build_images_query",
    "build_distinct_values_query",
    "distinct_filter_values",
]

# Columnas de `tbl_classifications` por las que se puede filtrar (confirmar con DATALAKE.md).
FILTERABLE_COLUMNS = ("weather", "scene", "timeofday", "road_geometry")
# Estado "bueno" de una etiqueta de usuario (los que ya pasaron revisión).
REVIEWED_STATUS = "reviewed"

DL_IMAGES = ("ds_raw_metadata", "tbl_images")
DL_CLASSIFICATIONS = ("ds_classification", "tbl_classifications")
DL_REVIEW = ("ds_label_review", "tbl_label_review_status")

_IMAGE_EXTS = (".jpg", ".jpeg", ".png")


class DatalakeError(ValueError):
    """Uso incorrecto de las consultas al Data Lake."""


def dl_fqn(project: str, dataset_table: tuple[str, str]) -> str:
    """`project.dataset.table` de una tabla del Data Lake."""
    dataset, table = dataset_table
    return f"{project}.{dataset}.{table}"


def label_uri_for_image(image_uri: str) -> str:
    """Deriva la ruta del `.lines.json` (carpeta `label/`) desde la de la imagen.

    Convención del Data Lake: `.../images/<...>.jpg` -> `.../label/<...>.lines.json`.
    """
    if "/images/" not in image_uri:
        raise DatalakeError(f"URI de imagen sin '/images/': {image_uri!r}")
    base = image_uri.replace("/images/", "/label/", 1)
    low = base.lower()
    for ext in _IMAGE_EXTS:
        if low.endswith(ext):
            return base[: -len(ext)] + ".lines.json"
    raise DatalakeError(f"URI sin extensión de imagen conocida {_IMAGE_EXTS}: {image_uri!r}")


def build_where(filters: Mapping[str, Any], *, alias: str = "c") -> tuple[list[str], dict[str, Any]]:
    """Cláusulas WHERE + params para unos filtros de clasificación (columna -> valor)."""
    clauses: list[str] = []
    params: dict[str, Any] = {}
    for col, val in filters.items():
        if col not in FILTERABLE_COLUMNS:
            raise DatalakeError(f"columna de filtro no permitida: {col!r} (válidas: {FILTERABLE_COLUMNS})")
        pname = f"f_{col}"
        clauses.append(f"{alias}.{col} = @{pname}")
        params[pname] = val
    return clauses, params


def build_images_query(
    dl_project: str,
    source: str,
    dataset: str,
    split: str,
    filters: Optional[Mapping[str, Any]] = None,
    *,
    limit: Optional[int] = None,
) -> tuple[str, dict[str, Any]]:
    """SELECT de las imágenes de un (source, dataset, split) con filtros opcionales.

    - Si hay `filters`: JOIN con `tbl_classifications`.
    - Si `source == 'user'`: LEFT JOIN con `tbl_label_review_status`; se CONSERVAN las
      imágenes que NO están en revisión (bien anotadas) o cuyo `status='reviewed'`.
      Se descartan solo las que están en revisión y aún no son `reviewed`.
    - `limit`: si se pasa, añade `LIMIT n` (para pruebas; evita descargar todo).
    Devuelve (sql, params). El `.lines.json` se deriva luego con `lines_uri_for_image`.
    """
    filters = dict(filters or {})
    images = dl_fqn(dl_project, DL_IMAGES)
    joins: list[str] = []
    where = ["i.source = @source", "i.dataset = @dataset", "i.split = @split"]
    params: dict[str, Any] = {"source": source, "dataset": dataset, "split": split}

    if filters:
        clf = dl_fqn(dl_project, DL_CLASSIFICATIONS)
        joins.append(f"JOIN `{clf}` c ON c.image_id = i.image_id")
        clauses, fparams = build_where(filters)
        where += clauses
        params.update(fparams)

    if source == "user":
        # LEFT JOIN: conservar las NO presentes en revisión (bien anotadas) o las reviewed.
        rev = dl_fqn(dl_project, DL_REVIEW)
        joins.append(f"LEFT JOIN `{rev}` r ON r.image_id = i.image_id")
        where.append("(r.image_id IS NULL OR r.status = @review_status)")
        params["review_status"] = REVIEWED_STATUS

    sql = (
        f"SELECT i.image_id, i.gcs_uri, i.width, i.height FROM `{images}` i"
        + ("".join(f" {j}" for j in joins))
        + " WHERE " + " AND ".join(where)
        + " ORDER BY i.image_id"
    )
    if limit is not None:
        sql += f" LIMIT {int(limit)}"
    return sql, params


def build_distinct_values_query(dl_project: str, column: str) -> str:
    """`SELECT DISTINCT` de los valores de una columna filtrable (descubrimiento dinámico)."""
    if column not in FILTERABLE_COLUMNS:
        raise DatalakeError(f"columna no permitida: {column!r} (válidas: {FILTERABLE_COLUMNS})")
    clf = dl_fqn(dl_project, DL_CLASSIFICATIONS)
    return f"SELECT DISTINCT {column} AS value FROM `{clf}` WHERE {column} IS NOT NULL ORDER BY {column}"


def distinct_filter_values(bq: "BigQuery", dl_project: str, column: str) -> list[Any]:
    """Ejecuta el DISTINCT y devuelve la lista de valores (para los desplegables)."""
    rows = bq.query(build_distinct_values_query(dl_project, column))
    return [r["value"] for r in rows]
