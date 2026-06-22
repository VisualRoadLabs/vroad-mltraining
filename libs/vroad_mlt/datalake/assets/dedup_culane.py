"""vroad_mlt.datalake.assets.dedup_culane — descarte de frames casi idénticos (SOLO CULane).

CLRerNet descubrió que CULane tiene clips con frames casi idénticos y los quitó.
Mecanismo: un `.npz` (`train_diffs.npz`, clave `data`) con la diferencia de cada
frame respecto al anterior, ALINEADO índice-a-índice con `train.txt`. Se CONSERVA el
frame si su diferencia >= umbral. Verificado: umbral 15.0 reproduce exactamente
`train_gt_new.txt` (88880 -> 55698 frames).

Adaptación al bucket: las rutas del .npz/train.txt son nativas de CULane
(`/driver_23_30frame/05151649_0422.MP4/00000.jpg`); en `bkt-prod-public-usc1` la
imagen vive en `culane/images/driver_.../00000.jpg`. La clave de comparación es la
parte tras `/images/` (sin `/` inicial), común a ambos.

Es un one-off: vive en `assets/` para no tocar el resto, y SOLO se aplica a CULane.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence, Union

__all__ = [
    "DEFAULT_DIFF_THRESHOLD",
    "normalize_culane_key",
    "culane_key_from_gcs_uri",
    "build_keep_set",
    "load_diffs",
    "load_train_keys",
    "keep_set_from_files",
]

# Umbral por defecto: reproduce train_gt_new.txt (verificado contra los ficheros locales).
DEFAULT_DIFF_THRESHOLD = 15.0


def normalize_culane_key(path: str) -> str:
    """Ruta nativa de CULane -> clave canónica (sin espacios ni `/` inicial)."""
    return path.strip().lstrip("/")


def culane_key_from_gcs_uri(gcs_uri: str) -> str:
    """Clave CULane (`driver_.../00000.jpg`) desde la URI de la imagen en el bucket."""
    marker = "/images/"
    i = gcs_uri.find(marker)
    if i < 0:
        raise ValueError(f"URI sin '/images/': {gcs_uri!r}")
    return normalize_culane_key(gcs_uri[i + len(marker):])


def build_keep_set(
    train_keys: Sequence[str], diffs: Sequence[float], threshold: float = DEFAULT_DIFF_THRESHOLD
) -> set[str]:
    """Claves a CONSERVAR: frame `i` si `diffs[i] >= threshold`. Pura.

    `train_keys` (de train.txt) y `diffs` (de train_diffs.npz) deben estar alineados.
    """
    if len(train_keys) != len(diffs):
        raise ValueError(
            f"train_keys ({len(train_keys)}) y diffs ({len(diffs)}) deben tener la misma longitud"
        )
    return {
        normalize_culane_key(train_keys[i])
        for i in range(len(train_keys))
        if float(diffs[i]) >= threshold
    }


# ------------------------------------------------------------------------ IO


def load_diffs(src: Any) -> Any:
    """Carga el array `data` de un `.npz` (ruta, bytes o file-like). Necesita numpy."""
    import numpy as np

    with np.load(src) as z:
        return z["data"]


def load_train_keys(train_txt: Union[str, Path]) -> list[str]:
    """Lee `train.txt` -> lista de claves (la 1ª columna de cada línea, normalizada)."""
    text = Path(train_txt).read_text(encoding="utf-8")
    return [normalize_culane_key(ln.split()[0]) for ln in text.splitlines() if ln.strip()]


def keep_set_from_files(
    train_txt: Union[str, Path], diffs_npz: Any, threshold: float = DEFAULT_DIFF_THRESHOLD
) -> set[str]:
    """Conjunto de claves a conservar leyendo `train.txt` + `train_diffs.npz`."""
    return build_keep_set(load_train_keys(train_txt), list(load_diffs(diffs_npz)), threshold)


# --------------------------------------------------------------------------- CLI


def _main(argv: Union[list, None] = None) -> int:
    """Comprobación manual contra tus ficheros locales de CULane.

        python -m vroad_mlt.datalake.assets.dedup_culane verify <list_dir> [threshold]

    Compara el keep-set generado con `train_gt_new.txt` (debe coincidir al umbral 15).
    """
    import sys

    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) < 2 or args[0] != "verify":
        print(_main.__doc__, file=sys.stderr)
        return 2
    d = Path(args[1])
    threshold = float(args[2]) if len(args) > 2 else DEFAULT_DIFF_THRESHOLD

    keep = keep_set_from_files(d / "train.txt", str(d / "train_diffs.npz"), threshold)
    print(f"keep (umbral {threshold}): {len(keep)} frames")
    new_txt = d / "train_gt_new.txt"
    if new_txt.exists():
        ref = {normalize_culane_key(ln.split()[0]) for ln in new_txt.read_text(encoding="utf-8").splitlines() if ln.strip()}
        print(f"train_gt_new.txt        : {len(ref)} frames")
        print("OK coincide" if keep == ref else "DIFIERE del esperado")
        return 0 if keep == ref else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
