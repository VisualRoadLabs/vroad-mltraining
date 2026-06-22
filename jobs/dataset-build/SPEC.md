# Spec de entrada de `dataset-build`

Formato del JSON que recibe el job `job-dev-dataset-build-usc1` para materializar un
dataset (shards WebDataset + manifiesto) desde el Data Lake. **Lo genera el dashboard**
y lo consume el runner. Aprobado 2026-06-22.

```json
{
  "manifest_id": "culane-mix-curves",
  "version": "3",
  "shard_maxcount": 10000,
  "sources": [
    {
      "source": "public",
      "dataset": "culane",
      "version": "1",
      "splits": ["train", "val", "test"],
      "filters": {},
      "dedup": false
    },
    {
      "source": "user",
      "dataset": "user",
      "version": "2",
      "splits": ["train"],
      "filters": { "road_geometry": "curve" }
    }
  ]
}
```

## Campos

| Campo | Significado |
|---|---|
| `manifest_id` / `version` | Identidad del manifiesto resultante (`manifests/<id>@<version>.json`). |
| `shard_maxcount` | Nº máx. de samples por `.tar` (rotación de `ShardWriter`). Opcional (default 10000). |
| `sources[]` | Lista de fuentes que se mezclan en el manifiesto. |
| `sources[].source` | `public` \| `user`. |
| `sources[].dataset` | Nombre del dataset en `tbl_images.dataset` (p. ej. `culane`, `curvelanes`, o `user`). |
| `sources[].version` | Versión de la fuente (va al manifiesto). |
| `sources[].splits` | Splits a materializar de esa fuente. |
| `sources[].filters` | Filtros de `tbl_classifications` (`weather`/`scene`/`timeofday`/`road_geometry`). Los **valores** disponibles se descubren con `SELECT DISTINCT` (no se hardcodean). |
| `sources[].dedup` | **Solo CULane:** `true` descarta frames casi idénticos (ver abajo). Default `false`. |

## Reglas del runner

- **Usuario:** lee de `bkt-prod-user-usc1`. Se **conservan** las imágenes bien anotadas
  (las que **NO** aparecen en `tbl_label_review_status`) **más** las que sí aparecen con
  `status='reviewed'`. Se **descartan** solo las que están en revisión y aún no son `reviewed`
  (LEFT JOIN + `WHERE r.image_id IS NULL OR r.status='reviewed'`).
- **Públicos sin GT en un split** (p. ej. CurveLanes test): el runner **salta** los samples cuyo
  `.lines.json` no exista y reporta los `counts` reales materializados.
- **`.lines.json`**: se deriva de la URI de la imagen (`/images/` → `/label/`, `.jpg` → `.lines.json`).
- **Salida:** `shards/<dataset>@<version>/<split>-NNNNN.tar` + `manifests/<id>@<version>.json`
  (+ `…curve_keys.json` si hay filtro de curva) en `bkt-dev-datasets-usc1`.

## Dedup de CULane (`dedup: true`)

One-off `vroad_mlt.datalake.assets.dedup_culane`. CLRerNet detectó clips con frames casi idénticos:
se conserva el frame `i` si `train_diffs.npz[i] >= 15.0` (umbral verificado: reproduce
`train_gt_new.txt`, 88880 → 55698). Necesita los assets de CULane (`train.txt` + `train_diffs.npz`)
accesibles por el job (subirlos a GCS, p. ej. `bkt-dev-datasets-usc1/_assets/culane/`).
La clave de comparación es la parte tras `/images/` (común al .npz nativo y al bucket).
