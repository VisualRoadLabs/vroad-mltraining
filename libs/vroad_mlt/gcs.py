"""vroad_mlt.gcs — helpers de Google Cloud Storage (lo común de leer/escribir objetos).

Una sola capa fina sobre `google-cloud-storage` para que dataset-build, trainer y
dashboard no repitan código de I/O: leer/escribir bytes/texto/JSON, subir/bajar
ficheros, listar, comprobar existencia, borrar y URLs firmadas. Todo opera sobre
URIs `gs://...`.

Diseño:
- El `project` SIEMPRE es explícito (el default de gcloud aquí es el Data Lake de
  PROD; nunca se usa). `Gcs.from_settings(cfg)` lo coge de `Settings.project_training`.
- Sin estado global: se crea un `Gcs` con su cliente y se reutiliza.
- Requiere el extra `gcp` (`pip install -e ".[gcp]"`).
"""

from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator, Optional, Union

from google.cloud import storage  # type: ignore[import-untyped]

from vroad_mlt.naming import gs_uri, split_gs_uri

if TYPE_CHECKING:
    from vroad_mlt.config import Settings

__all__ = ["Gcs"]

_PathLike = Union[str, Path]


class Gcs:
    """Operaciones de GCS sobre URIs `gs://`."""

    def __init__(self, client: "storage.Client") -> None:
        self._client = client

    # ------------------------------------------------------------- fábricas
    @classmethod
    def from_project(cls, project: str, *, credentials: Any = None) -> "Gcs":
        return cls(storage.Client(project=project, credentials=credentials))

    @classmethod
    def from_settings(cls, settings: "Settings", *, credentials: Any = None) -> "Gcs":
        return cls.from_project(settings.project_training, credentials=credentials)

    @property
    def client(self) -> "storage.Client":
        return self._client

    # ------------------------------------------------------------- interno
    def _blob(self, uri: str) -> "storage.Blob":
        bucket, key = split_gs_uri(uri)
        if not key:
            raise ValueError(f"URI sin objeto (solo bucket): {uri!r}")
        return self._client.bucket(bucket).blob(key)

    # ------------------------------------------------------------- escritura
    def write_bytes(self, uri: str, data: bytes, *, content_type: Optional[str] = None) -> None:
        self._blob(uri).upload_from_string(data, content_type=content_type)

    def write_text(self, uri: str, text: str, *, content_type: str = "text/plain; charset=utf-8") -> None:
        self._blob(uri).upload_from_string(text.encode("utf-8"), content_type=content_type)

    def write_json(self, uri: str, obj: Any, *, indent: Optional[int] = None) -> None:
        body = json.dumps(obj, ensure_ascii=False, indent=indent)
        self.write_text(uri, body, content_type="application/json; charset=utf-8")

    def upload_file(self, uri: str, local_path: _PathLike) -> None:
        self._blob(uri).upload_from_filename(str(local_path))

    # ------------------------------------------------------------- lectura
    def read_bytes(self, uri: str) -> bytes:
        return self._blob(uri).download_as_bytes()

    def read_text(self, uri: str) -> str:
        return self.read_bytes(uri).decode("utf-8")

    def read_json(self, uri: str) -> Any:
        return json.loads(self.read_text(uri))

    def download_file(self, uri: str, local_path: _PathLike) -> None:
        p = Path(local_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        self._blob(uri).download_to_filename(str(p))

    # ------------------------------------------------------------- consulta
    def exists(self, uri: str) -> bool:
        return self._blob(uri).exists()

    def list_uris(self, prefix_uri: str, *, max_results: Optional[int] = None) -> list[str]:
        bucket, key_prefix = split_gs_uri(prefix_uri)
        blobs = self._client.list_blobs(bucket, prefix=key_prefix, max_results=max_results)
        return [gs_uri(bucket, b.name) for b in blobs]

    def iter_blobs(self, prefix_uri: str) -> Iterator["storage.Blob"]:
        bucket, key_prefix = split_gs_uri(prefix_uri)
        return iter(self._client.list_blobs(bucket, prefix=key_prefix))

    # ------------------------------------------------------------- borrado
    def delete(self, uri: str, *, missing_ok: bool = True) -> bool:
        """Borra un objeto. Devuelve True si lo borró, False si no existía."""
        from google.api_core import exceptions as gax  # type: ignore[import-untyped]

        try:
            self._blob(uri).delete()
            return True
        except gax.NotFound:
            if missing_ok:
                return False
            raise

    def delete_prefix(self, prefix_uri: str) -> int:
        """Borra todos los objetos bajo un prefijo. Devuelve cuántos borró."""
        count = 0
        for blob in self.iter_blobs(prefix_uri):
            blob.delete()
            count += 1
        return count

    # ------------------------------------------------------------- firmadas
    def signed_url(self, uri: str, *, expires: int = 3600, method: str = "GET") -> str:
        """URL firmada v4 (p. ej. para que el dashboard sirva una imagen de viz).

        Necesita credenciales capaces de FIRMAR (clave de SA o impersonación con
        `iam.serviceAccountTokenCreator`); con ADC de usuario puro no funciona.
        """
        return self._blob(uri).generate_signed_url(
            version="v4", expiration=timedelta(seconds=expires), method=method
        )


# --------------------------------------------------------------------------- CLI


def _main(argv: Optional[list[str]] = None) -> int:
    """Comprobación manual contra GCS real (necesita ADC y red).

        python -m vroad_mlt.gcs selftest          # write+read+delete en staging/_test/
        python -m vroad_mlt.gcs ls    gs://bucket/prefijo
        python -m vroad_mlt.gcs cat   gs://bucket/obj
        python -m vroad_mlt.gcs exists gs://bucket/obj
        python -m vroad_mlt.gcs put   gs://bucket/obj  fichero_local
        python -m vroad_mlt.gcs get   gs://bucket/obj  fichero_local
        python -m vroad_mlt.gcs rm    gs://bucket/obj
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
        gcs = Gcs.from_settings(cfg)
        if op == "selftest":
            base = gs_uri(cfg.bucket_vertex_staging, "_test", uuid.uuid4().hex)
            uri = f"{base}/hello.txt"
            gcs.write_text(uri, "hola gcs")
            got = gcs.read_text(uri)
            assert got == "hola gcs", got
            assert gcs.exists(uri)
            n = gcs.delete_prefix(base)
            print(f"OK selftest -> escribió/leyó/borró ({n}) en {base}")
        elif op == "ls":
            for u in gcs.list_uris(rest[0]):
                print(u)
        elif op == "cat":
            print(gcs.read_text(rest[0]))
        elif op == "exists":
            print(gcs.exists(rest[0]))
        elif op == "put":
            gcs.upload_file(rest[0], rest[1]); print(f"subido -> {rest[0]}")
        elif op == "get":
            gcs.download_file(rest[0], rest[1]); print(f"bajado -> {rest[1]}")
        elif op == "rm":
            print("borrado" if gcs.delete(rest[0]) else "no existía")
        else:
            print(f"operación desconocida: {op}", file=sys.stderr); return 2
    except Exception as e:  # noqa: BLE001 - manual tool: queremos el error legible
        print(f"ERROR: {type(e).__name__}: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
