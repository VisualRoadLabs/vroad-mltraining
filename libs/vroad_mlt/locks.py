"""vroad_mlt.locks — candado de GPU en GCS (compare-and-swap atómico).

Blinda el presupuesto SIN servicios nuevos: antes de crear un Vertex job, el
workflow reclama un lock creando un objeto en GCS con `ifGenerationMatch=0` (crea
solo si NO existe -> compare-and-swap atómico). Si el objeto ya existe, ese slot
está ocupado y se prueba el siguiente. Slots:
- A100: 1  -> `locks/a100.lock`
- L4:   N  -> `locks/l4/slot-{0..N-1}.lock`  (N = MAX_L4)

El objeto-lock guarda `{run_id, who, claimed_at}`. Se libera borrándolo (en el
paso final del workflow, también si falla). El lock es la verdad, no el botón.

Partes PURAS (testeables sin GCP): normalizar el acelerador, las claves de los
slots y el contenido del lock. El CAS real se verifica a mano (CLI / `gsutil`).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Optional

from google.cloud import storage  # type: ignore[import-untyped]

from vroad_mlt.naming import gs_uri, split_gs_uri

if TYPE_CHECKING:
    from vroad_mlt.config import Settings

__all__ = [
    "LocksError",
    "A100",
    "L4",
    "ClaimedLock",
    "normalize_accelerator",
    "slot_keys",
    "lock_payload",
    "Locks",
]

A100 = "NVIDIA_TESLA_A100"
L4 = "NVIDIA_L4"


class LocksError(ValueError):
    """Uso incorrecto de los locks (acelerador desconocido, dueño que no coincide...)."""


def normalize_accelerator(accel: str) -> str:
    """Acepta `NVIDIA_TESLA_A100`/`NVIDIA_L4` o `a100`/`l4` -> `'a100'` | `'l4'`."""
    a = accel.strip().lower()
    if a in ("a100", "nvidia_tesla_a100"):
        return "a100"
    if a in ("l4", "nvidia_l4"):
        return "l4"
    raise LocksError(f"acelerador desconocido: {accel!r} (usa A100 o L4)")


def slot_keys(accelerator: str, max_a100: int = 1, max_l4: int = 3, *, root: str = "locks") -> list[str]:
    """Claves candidatas (en orden) de los slots de un acelerador.

    Con `root='locks'` coincide exactamente con la convención de `naming`.
    """
    kind = normalize_accelerator(accelerator)
    if kind == "a100":
        return [f"{root}/a100.lock"]
    return [f"{root}/l4/slot-{i}.lock" for i in range(int(max_l4))]


def lock_payload(run_id: str, who: str, claimed_at: str) -> dict[str, str]:
    """Contenido del objeto-lock."""
    return {"run_id": run_id, "who": who, "claimed_at": claimed_at}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True)
class ClaimedLock:
    """Un lock reclamado con éxito."""

    uri: str
    key: str
    run_id: str
    who: str
    claimed_at: str


class Locks:
    """Reclama/libera/lista slots de GPU en el bucket de staging."""

    def __init__(
        self,
        client: "storage.Client",
        bucket: str,
        *,
        max_a100: int = 1,
        max_l4: int = 3,
        root: str = "locks",
    ) -> None:
        self._client = client
        self._bucket_name = bucket
        self._bucket = client.bucket(bucket)
        self.max_a100 = max_a100
        self.max_l4 = max_l4
        self.root = root

    @classmethod
    def from_settings(cls, settings: "Settings", *, credentials: Any = None, root: str = "locks") -> "Locks":
        client = storage.Client(project=settings.project_training, credentials=credentials)
        return cls(
            client,
            settings.bucket_vertex_staging,
            max_a100=settings.max_a100,
            max_l4=settings.max_l4,
            root=root,
        )

    def _keys(self, accelerator: str) -> list[str]:
        return slot_keys(accelerator, self.max_a100, self.max_l4, root=self.root)

    def _as_key(self, key_or_uri: str) -> str:
        if key_or_uri.startswith("gs://"):
            _, key = split_gs_uri(key_or_uri)
            return key
        return key_or_uri

    # ------------------------------------------------------------- reclamar
    def claim(
        self, accelerator: str, run_id: str, who: str, *, claimed_at: Optional[str] = None
    ) -> Optional[ClaimedLock]:
        """Reclama el primer slot libre (CAS atómico). `None` si están todos ocupados."""
        from google.api_core import exceptions as gax  # type: ignore[import-untyped]

        payload = lock_payload(run_id, who, claimed_at or _now_iso())
        body = json.dumps(payload)
        for key in self._keys(accelerator):
            blob = self._bucket.blob(key)
            try:
                blob.upload_from_string(body, content_type="application/json", if_generation_match=0)
                return ClaimedLock(uri=gs_uri(self._bucket_name, key), key=key, **payload)
            except gax.PreconditionFailed:
                continue  # ese slot ya está ocupado, probamos el siguiente
        return None

    # ------------------------------------------------------------- liberar
    def release(self, key_or_uri: str, *, run_id: Optional[str] = None) -> bool:
        """Libera un lock (lo borra). Devuelve False si ya no existía.

        Si pasas `run_id`, solo lo libera si ese lock es suyo (cinturón de seguridad).
        """
        from google.api_core import exceptions as gax  # type: ignore[import-untyped]

        key = self._as_key(key_or_uri)
        blob = self._bucket.blob(key)
        if run_id is not None:
            try:
                current = json.loads(blob.download_as_bytes())
            except gax.NotFound:
                return False
            owner = current.get("run_id")
            if owner != run_id:
                raise LocksError(f"el lock {key} es de {owner!r}, no de {run_id!r}; no se libera")
        try:
            blob.delete()
            return True
        except gax.NotFound:
            return False

    # ------------------------------------------------------------- estado
    def occupancy(self, accelerator: str) -> list[dict]:
        """Estado de cada slot del acelerador: libre, o con su payload."""
        out: list[dict] = []
        for key in self._keys(accelerator):
            blob = self._bucket.blob(key)
            if blob.exists():
                try:
                    payload = json.loads(blob.download_as_bytes())
                except Exception:  # noqa: BLE001 - lock corrupto: lo marcamos ocupado igualmente
                    payload = {}
                out.append({"slot": key, "free": False, **payload})
            else:
                out.append({"slot": key, "free": True})
        return out


# --------------------------------------------------------------------------- CLI


def _main(argv: Optional[list[str]] = None) -> int:
    """Comprobación manual contra GCS real (necesita ADC y red).

        python -m vroad_mlt.locks status  <a100|l4>
        python -m vroad_mlt.locks claim    <a100|l4> <run_id>
        python -m vroad_mlt.locks release  <a100|l4> <run_id>   # libera el slot de ese run
        python -m vroad_mlt.locks selftest <a100|l4>            # bajo _test_locks/, no toca los reales

    `status/claim/release` usan los locks REALES (`locks/...`). `selftest` usa un
    root de prueba aislado y limpia al final.
    """
    import sys
    import uuid

    from vroad_mlt.config import get_settings

    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) < 2:
        print(_main.__doc__, file=sys.stderr)
        return 2
    op, accel, rest = args[0], args[1], args[2:]

    try:
        cfg = get_settings()
        if op == "status":
            locks = Locks.from_settings(cfg)
            for s in locks.occupancy(accel):
                print(s)
        elif op == "claim":
            locks = Locks.from_settings(cfg)
            got = locks.claim(accel, rest[0], who="manual-cli")
            print(f"reclamado: {got}" if got else "SIN SLOTS LIBRES")
        elif op == "release":
            locks = Locks.from_settings(cfg)
            keys = locks._keys(accel)
            freed = any(locks.release(k, run_id=rest[0]) for k in keys)
            print("liberado" if freed else "no había lock de ese run")
        elif op == "selftest":
            locks = Locks.from_settings(cfg, root=f"_test_locks/{uuid.uuid4().hex[:8]}")
            n = len(locks._keys(accel))
            claims = [locks.claim(accel, f"run-{i}", who="selftest") for i in range(n + 1)]
            got = [c for c in claims if c]
            assert len(got) == n, f"esperaba {n} claims, hubo {len(got)}"
            assert claims[-1] is None, "el slot extra debería estar lleno"
            for c in got:
                locks.release(c.key)
            print(f"OK selftest {accel}: {n} slots reclamados, +1 rechazado, todos liberados")
        else:
            print(f"operación desconocida: {op}", file=sys.stderr)
            return 2
    except Exception as e:  # noqa: BLE001 - herramienta manual: error legible
        print(f"ERROR: {type(e).__name__}: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
