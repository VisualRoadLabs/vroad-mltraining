"""vroad_mlt.bq — helpers de BigQuery (SELECT + MERGE upsert) compartidos.

Los jobs escriben con MERGE (query job), no con load jobs: así solo necesitan
`bigquery.jobUser` + R/W de la tabla destino (sin `tables.create` a nivel dataset).
Este módulo construye el MERGE PARAMETRIZADO (sin inyección) y lo ejecuta, y ofrece
un `query()` que devuelve filas como dicts.

Partes PURAS (testeables sin GCP): construir el SQL del MERGE, inferir el tipo BQ
de un valor y planear los parámetros. La ejecución contra BigQuery se verifica a
mano (CLI `python -m vroad_mlt.bq ...` o el comando `bq`).

Diseño:
- `project` SIEMPRE explícito (`Settings.project_training`).
- Upsert de 1..N filas en UN job: el `USING` es un `UNION ALL` de SELECTs con
  parámetros escalares (`@p{i}_{col}`). Solo se actualizan las columnas presentes
  (upsert parcial), por lo que el workflow puede tocar solo `status`/timestamps.
- Requiere el extra `gcp` (`pip install -e ".[gcp]"`).
"""

from __future__ import annotations

import datetime as _dt
import json
from typing import TYPE_CHECKING, Any, Mapping, Optional, Sequence

from google.cloud import bigquery  # type: ignore[import-untyped]

if TYPE_CHECKING:
    from vroad_mlt.config import Settings

__all__ = [
    "BqError",
    "BigQuery",
    "param_name",
    "bq_type_for",
    "columns_of",
    "build_merge_sql",
    "plan_merge_params",
]


class BqError(ValueError):
    """Error de uso de BigQuery (filas/columnas/tipos mal formados)."""


# ----------------------------------------------------------------- puro (testable)


def param_name(i: int, col: str) -> str:
    """Nombre de parámetro para la fila `i`, columna `col` (identificador BQ válido)."""
    return f"p{i}_{col}"


def bq_type_for(value: Any) -> str:
    """Infiere el tipo BigQuery de un valor Python (escalares + JSON + tiempo)."""
    if isinstance(value, bool):            # antes que int (bool es subclase de int)
        return "BOOL"
    if isinstance(value, int):
        return "INT64"
    if isinstance(value, float):
        return "FLOAT64"
    if isinstance(value, str):
        return "STRING"
    if isinstance(value, _dt.datetime):
        return "TIMESTAMP"
    if isinstance(value, _dt.date):
        return "DATE"
    if isinstance(value, (dict, list)):
        return "JSON"
    raise BqError(f"tipo no soportado para BigQuery: {type(value).__name__}")


def columns_of(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """Columnas (en el orden de la 1ª fila); exige que todas las filas tengan el mismo set."""
    if not rows:
        raise BqError("no hay filas")
    columns = list(rows[0].keys())
    expected = set(columns)
    for r in rows:
        if set(r.keys()) != expected:
            raise BqError("todas las filas deben tener exactamente las mismas columnas")
    return columns


def build_merge_sql(
    table_fqn: str, columns: Sequence[str], key_columns: Sequence[str], n_rows: int
) -> str:
    """Construye el MERGE parametrizado (upsert) para `n_rows` filas.

    Solo actualiza columnas NO clave (upsert parcial); si solo hay claves, no
    genera `WHEN MATCHED` (un match sería un no-op).
    """
    columns = list(columns)
    key_columns = list(key_columns)
    if not columns:
        raise BqError("sin columnas")
    if not key_columns:
        raise BqError("sin key_columns")
    if n_rows < 1:
        raise BqError("n_rows debe ser >= 1")
    missing = [k for k in key_columns if k not in columns]
    if missing:
        raise BqError(f"key_columns no presentes en columns: {missing}")
    non_key = [c for c in columns if c not in key_columns]

    selects = [
        "SELECT " + ", ".join(f"@{param_name(i, c)} AS {c}" for c in columns)
        for i in range(n_rows)
    ]
    source = " UNION ALL ".join(selects)
    on = " AND ".join(f"T.{k} = S.{k}" for k in key_columns)
    insert_cols = ", ".join(columns)
    insert_vals = ", ".join(f"S.{c}" for c in columns)

    sql = f"MERGE `{table_fqn}` T USING ({source}) S ON {on}"
    if non_key:
        sql += " WHEN MATCHED THEN UPDATE SET " + ", ".join(f"{c} = S.{c}" for c in non_key)
    sql += f" WHEN NOT MATCHED THEN INSERT ({insert_cols}) VALUES ({insert_vals})"
    return sql


def plan_merge_params(
    rows: Sequence[Mapping[str, Any]],
    columns: Sequence[str],
    types: Optional[Mapping[str, str]] = None,
) -> list[tuple[str, str, Any]]:
    """Planea los parámetros (name, bq_type, value) del MERGE. Puro/testeable.

    `types` fija el tipo BQ de columnas que lo necesiten (JSON, TIMESTAMP, o
    columnas que puedan ser None). Las JSON con valor dict/list se serializan.
    """
    types = types or {}
    out: list[tuple[str, str, Any]] = []
    for i, row in enumerate(rows):
        for col in columns:
            value = row[col]
            bqt = types.get(col) or (bq_type_for(value) if value is not None else None)
            if bqt is None:
                raise BqError(f"columna {col!r} es None sin tipo; pásalo en types=")
            if bqt == "JSON" and value is not None and not isinstance(value, str):
                value = json.dumps(value, ensure_ascii=False)
            out.append((param_name(i, col), bqt, value))
    return out


# ----------------------------------------------------------------- cliente (GCP)


class BigQuery:
    """Ejecuta SELECT y MERGE upsert contra BigQuery."""

    def __init__(self, client: "bigquery.Client") -> None:
        self._client = client
        self._project = client.project

    @classmethod
    def from_project(cls, project: str, *, credentials: Any = None) -> "BigQuery":
        return cls(bigquery.Client(project=project, credentials=credentials))

    @classmethod
    def from_settings(cls, settings: "Settings", *, credentials: Any = None) -> "BigQuery":
        return cls.from_project(settings.project_training, credentials=credentials)

    @property
    def client(self) -> "bigquery.Client":
        return self._client

    def table_fqn(self, dataset: str, table: str) -> str:
        return f"{self._project}.{dataset}.{table}"

    def query(self, sql: str, params: Optional[Mapping[str, Any]] = None) -> list[dict]:
        """Ejecuta SQL y devuelve las filas como dicts. `params`: name -> valor."""
        job_config = None
        if params:
            qp = [bigquery.ScalarQueryParameter(n, bq_type_for(v), v) for n, v in params.items()]
            job_config = bigquery.QueryJobConfig(query_parameters=qp)
        job = self._client.query(sql, job_config=job_config)
        return [dict(row) for row in job.result()]

    def merge_upsert(
        self,
        dataset: str,
        table: str,
        rows: Sequence[Mapping[str, Any]],
        key_columns: Sequence[str],
        *,
        types: Optional[Mapping[str, str]] = None,
    ) -> int:
        """Upsert de 1..N filas por `key_columns`. Devuelve filas afectadas."""
        if not rows:
            return 0
        columns = columns_of(rows)
        sql = build_merge_sql(self.table_fqn(dataset, table), columns, key_columns, len(rows))
        planned = plan_merge_params(rows, columns, types)
        params = [bigquery.ScalarQueryParameter(n, t, v) for n, t, v in planned]
        job = self._client.query(sql, job_config=bigquery.QueryJobConfig(query_parameters=params))
        job.result()
        return job.num_dml_affected_rows or 0


# --------------------------------------------------------------------------- CLI


def _main(argv: Optional[list[str]] = None) -> int:
    """Comprobación manual contra BigQuery real (necesita ADC y red).

        python -m vroad_mlt.bq query "SELECT 1 AS ok"
        python -m vroad_mlt.bq selftest    # tabla temporal: upsert+update+select+drop

    `selftest` crea `_selftest_<uuid>` en el dataset de experimentos, prueba el
    MERGE (insert y luego update de la misma clave), lee y BORRA la tabla.
    """
    import sys
    import uuid

    from vroad_mlt.config import get_settings

    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        print(_main.__doc__, file=sys.stderr)
        return 2
    op, rest = args[0], args[1:]

    try:
        cfg = get_settings()
        bq = BigQuery.from_settings(cfg)
        if op == "query":
            for row in bq.query(rest[0]):
                print(row)
        elif op == "selftest":
            ds = cfg.bq_dataset_experiments
            tbl = f"_selftest_{uuid.uuid4().hex[:8]}"
            fqn = bq.table_fqn(ds, tbl)
            bq.query(f"CREATE TABLE `{fqn}` (k STRING, v INT64, note STRING)")
            try:
                bq.merge_upsert(ds, tbl, [{"k": "a", "v": 1, "note": "insert"}], ["k"])
                bq.merge_upsert(ds, tbl, [{"k": "a", "v": 2, "note": "update"}], ["k"])
                rows = bq.query(f"SELECT k, v, note FROM `{fqn}` ORDER BY k")
                ok = rows == [{"k": "a", "v": 2, "note": "update"}]
                print(f"filas tras upsert+update: {rows}")
                print("OK selftest (upsert correcto)" if ok else "FALLO: resultado inesperado")
            finally:
                bq.query(f"DROP TABLE `{fqn}`")
        else:
            print(f"operación desconocida: {op}", file=sys.stderr)
            return 2
    except Exception as e:  # noqa: BLE001 - herramienta manual: error legible
        print(f"ERROR: {type(e).__name__}: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
