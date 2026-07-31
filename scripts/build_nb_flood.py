#!/usr/bin/env python
"""Author notebooks/flood_timeseries.ipynb (source of truth for the notebook)."""

import nbformat as nbf

nb = nbf.v4.new_notebook()
C, M = nbf.v4.new_code_cell, nbf.v4.new_markdown_cell
cells = []

cells.append(M("""\
# Flood time series: Örnsköldsvik / Gottne (EMSR841), Sept 2025

Sentinel-1 GRD σ⁰ over the EMS AOI, geocoded with **sarsen** + GLO-30 (20 m, UTM 33N),
water masks from VV backscatter, scored against Copernicus EMS labels.

**Data situation (post-download).** Eleven GRD scenes across six relative orbits,
including complete pre/during/post series on the two label-anchor tracks:

- **rel 131 ASC**: Sept 4 (pre) → **Sept 10 (FEP label source, zero offset)** → Sept 16 (post).
- **rel 22 DESC**: Sept 3 (pre) → Sept 9 (during) → **Sept 15 (MONIT02 label source, zero offset)** → Sept 21 (post).
- Four other during-event passes (rel 168, 95, 102, 29) fill in the peak days.

Scenes with a same-track pre-event reference are scored with the **log-ratio**
change-detection baseline; the rest fall back to single-image Otsu. Every comparison
states the label-to-acquisition time offset — two rows have offset **0.0 h**.\
"""))

cells.append(C("""\
import sys, warnings
from pathlib import Path
sys.path.insert(0, "../src")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import geopandas as gpd
import rioxarray
import xarray as xr
import yaml
import matplotlib.pyplot as plt

from rawsar import grd, flood

ROOT = Path("..").resolve()
cfg = yaml.safe_load(open(ROOT / "configs/aoi_ornskoldsvik.yaml"))
BBOX = tuple(cfg["aoi"]["bbox"])
DEM = ROOT / "data/dem/aoi_glo30_utm33_20m.tif"
GTC_DIR = ROOT / "data/gtc"
EMS_DIR = ROOT / "data/ems"
plt.rcParams["figure.dpi"] = 110\
"""))

cells.append(M("## Acquisition inventory (from disk, grouped by relative orbit)"))

cells.append(C("""\
prods = [p for p in grd.scan(ROOT / "data/copernicus_data/S1") if p.kind == "GRDH" and not p.is_cog]
inv = pd.DataFrame([{
    "sensing": p.start, "sat": p.platform, "rel_orbit": p.relative_orbit,
    "datatake": p.datatake, "product": p.safe_name.split(".")[0],
} for p in prods]).sort_values("sensing").reset_index(drop=True)
inv\
"""))

cells.append(M("## DEM + geocoded σ⁰ (cached in `data/gtc/`)"))

cells.append(C("""\
dem_path = grd.prepare_dem(
    sorted((ROOT / "data/dem").glob("Copernicus_DSM_COG_10_N63_00_E01*_DEM.tif")),
    BBOX, DEM)

scenes = {}
for p in prods:
    out = GTC_DIR / f"{p.platform}_{p.relative_orbit:03d}_{p.start:%Y%m%d}_VV_sigma0.tif"
    da = grd.gtc_sigma0(p, "VV", dem_path, out, crop_bbox=BBOX)
    key = f"{p.start:%b %d %H:%M} {p.platform} rel{p.relative_orbit} "
    key += "ASC" if p.start.hour > 12 else "DESC"   # local pattern: ~16:xx ASC, ~05:xx DESC
    scenes[key] = {"da": da, "meta": p}
print(f"{len(scenes)} scenes geocoded on the {rioxarray.open_rasterio(dem_path).shape} DEM grid")\
"""))

cells.append(M("""\
## Time-series panel

Chronological σ⁰ (dB, VV). Water is dark. This is a *cross-track* series until the
per-track pairs download — incidence angle and viewing geometry differ between panels,
so panel-to-panel brightness changes are not purely hydrology.\
"""))

cells.append(C("""\
keys = sorted(scenes, key=lambda k: scenes[k]["meta"].start)
n = len(keys)
ncols = 3
nrows = -(-n // ncols)
# explicit north-up extent, robust to either y ordering (sarsen writes south-to-north)
da0 = scenes[keys[0]]["da"]
yy = da0.y.values
IM_ORIGIN = "lower" if yy[0] < yy[-1] else "upper"
full_extent = [float(da0.x[0]), float(da0.x[-1]), float(min(yy[0], yy[-1])), float(max(yy[0], yy[-1]))]
# note: sharex/sharey + constrained layout + fixed aspect silently drops titles on
# non-first columns (matplotlib quirk) -> no sharing, blank inner tick labels manually
fig, axes = plt.subplots(nrows, ncols, figsize=(15, 3.6 * nrows), layout="constrained")
for i, (ax, k) in enumerate(zip(np.ravel(axes), keys)):
    db = flood.to_db(scenes[k]["da"])
    ax.imshow(db.values, cmap="gray", vmin=-25, vmax=0,
              extent=full_extent, origin=IM_ORIGIN, aspect="equal")
    ax.set_title(k, fontsize=10)
    ax.set_ylim(full_extent[2], full_extent[3])
    if i % ncols: ax.set_yticklabels([])
    if i // ncols < nrows - 1: ax.set_xticklabels([])
for ax in np.ravel(axes)[n:]:
    ax.axis("off")
fig.suptitle("Sentinel-1 VV σ⁰ [dB], AOI01 Gottne — chronological, north up")\
"""))

cells.append(M("""\
## EMS reference labels

Four labelled EMS products exist for AOI01 (AOI02 was never produced). `observedEventA`
is the flood delineation of each product; MONIT products add `maximumFloodExtentA`
(cumulative). FEP has **no** flood-extent layer beyond `observedEventA`.\
"""))

cells.append(C("""\
EMS_PRODUCTS = {
    "FEP (S1 Sept 10 16:14)":      ("EMSR841_AOI01_FEP_PRODUCT_v1",  "FEP_PRODUCT",  "2025-09-10T16:14"),
    "DEL (ICEYE Sept 11 23:06)":   ("EMSR841_AOI01_DEL_PRODUCT_v1",  "DEL_PRODUCT",  "2025-09-11T23:06"),
    "MONIT01 (CSG Sept 12 16:57)": ("EMSR841_AOI01_DEL_MONIT01_v1",  "DEL_MONIT01",  "2025-09-12T16:57"),
    "MONIT02 (S1 Sept 15 05:06)":  ("EMSR841_AOI01_DEL_MONIT02_v1",  "DEL_MONIT02",  "2025-09-15T05:06"),
}
template = scenes[keys[0]]["da"]
# analysis region = EMS AOI polygon (labels are defined here; also excludes open sea)
aoi_gdf = gpd.read_file(EMS_DIR / "EMSR841_AOI01_DEL_MONIT02_v1/EMSR841_AOI01_DEL_MONIT02_areaOfInterestA_v1.json")
aoi_mask = flood.rasterize_labels(aoi_gdf, template)
print(f"analysis region (EMS AOI polygon): {float(aoi_mask.sum()) * 4e-4:.0f} km²")

labels = {}
for title, (d, stem, t) in EMS_PRODUCTS.items():
    gdf = gpd.read_file(EMS_DIR / d / f"EMSR841_AOI01_{stem}_observedEventA_v1.json")
    labels[title] = {"mask": flood.rasterize_labels(gdf, template), "time": pd.Timestamp(t), "n_polys": len(gdf)}
    mfe = EMS_DIR / d / f"EMSR841_AOI01_{stem}_maximumFloodExtentA_v1.json"
    if mfe.exists():
        labels[title]["max_extent"] = flood.rasterize_labels(gpd.read_file(mfe), template)

# cumulative "flood happened here at some point" reference: union of every EMS flood layer
ems_union = None
for v in labels.values():
    for layer in (v["mask"], v.get("max_extent")):
        if layer is not None:
            ems_union = layer if ems_union is None else (ems_union | layer)

pd.DataFrame({t: {"n_polys": v["n_polys"], "area_km2": round(float(v["mask"].sum()) * 4e-4, 2),
                  "has_max_extent": "max_extent" in v} for t, v in labels.items()}).T\
"""))

cells.append(M("""\
## Water masks and scores against EMS

Per acquisition: single-image Otsu-thresholded dark-water mask (capped at −14 dB).
Where a same-track pre-event reference exists, the **log-ratio** mask is used instead —
that's the WP4 image-domain baseline. Each detection is scored against the EMS product
nearest in time, with the offset stated. A permanent-water estimate (dark in the
pre-event rel-131 reference) is removed from both sides so scores measure *flood*, not rivers/lakes.\
"""))

cells.append(C("""\
# permanent water proxy: dark surface in the pre-event reference scene (Sept 4),
# restricted to the AOI polygon
ref_key = next(k for k in keys if scenes[k]["meta"].start.strftime("%m%d") == "0904")
ref_db = flood.to_db(scenes[ref_key]["da"]).where(aoi_mask)
perm_water, thr_pw = flood.water_mask_single(ref_db)
print(f"permanent-water proxy from '{ref_key}': thr={thr_pw:.1f} dB, "
      f"area={float(perm_water.sum()) * 4e-4:.1f} km² of "
      f"{float(aoi_mask.sum()) * 4e-4:.0f} km² AOI")\
"""))

cells.append(C("""\
def same_track_reference(meta, all_prods):
    '''Earliest pre-event GRD on the same relative orbit, if any (activates post-download).'''
    cands = [p for p in all_prods
             if p.relative_orbit == meta.relative_orbit and p.start < meta.start
             and p.start < pd.Timestamp(cfg["event"]["window_during"][0]).to_pydatetime()]
    return min(cands, key=lambda p: abs((meta.start - p.start).days)) if cands else None

rows = []
detections = {}
for k in keys:
    meta = scenes[k]["meta"]
    db = flood.to_db(scenes[k]["da"]).where(aoi_mask)
    ref = same_track_reference(meta, prods)
    if ref is not None:
        ref_da = grd.gtc_sigma0(ref, "VV", dem_path,
                                GTC_DIR / f"{ref.platform}_{ref.relative_orbit:03d}_{ref.start:%Y%m%d}_VV_sigma0.tif",
                                crop_bbox=BBOX)
        mask, thr = flood.water_mask_logratio(db, flood.to_db(ref_da).where(aoi_mask))
        method = f"log-ratio vs {ref.start:%b %d}"
    else:
        mask, thr = flood.water_mask_single(db)
        method = "single-image Otsu"
    mask = mask & ~perm_water
    detections[k] = mask
    valid = xr.apply_ufunc(np.isfinite, db) & aoi_mask
    t = pd.Timestamp(meta.start)
    nearest = min(labels, key=lambda L: abs(labels[L]["time"] - t))
    # truth = observedEvent ∪ maximumFloodExtent: MONIT products delineate almost no
    # standing water at their own timestamp (0.05 km² for MONIT02) but carry the
    # cumulative extent — score against everything the product asserts flooded.
    truth = labels[nearest]["mask"]
    if "max_extent" in labels[nearest]:
        truth = truth | labels[nearest]["max_extent"]
    truth = truth & ~perm_water
    m = flood.iou_f1(mask, truth, valid=valid)
    mu = flood.iou_f1(mask, ems_union & ~perm_water, valid=valid)
    rows.append({"scene": k, "method": method, "thr_dB": round(thr, 1),
                 "vs": nearest, "offset_h": round(abs((labels[nearest]["time"] - t).total_seconds()) / 3600, 1),
                 "IoU": round(m["iou"], 3), "F1": round(m["f1"], 3),
                 "IoU_union": round(mu["iou"], 3),
                 "pred_km2": round(m["pred_area_km2"], 1), "ems_km2": round(m["truth_area_km2"], 1)})
scores = pd.DataFrame(rows)
scores\
"""))

cells.append(M("""\
The per-scene numbers above are the honest baseline. Expected pattern: the largest
detected areas and best agreement around Sept 8–10; IoU is limited both by the
label-to-acquisition offset (flood recedes between S1 pass and EMS source image)
and by single-image detection where no same-track reference exists yet.\
"""))

cells.append(C("""\
# Visual: best-scoring scene vs its EMS reference (zoomed to the labelled valley)
best = scores.sort_values("IoU_union").iloc[-1]
k = best["scene"]
db = flood.to_db(scenes[k]["da"]).where(aoi_mask)
truth = (ems_union & ~perm_water)
pred = detections[k]

# zoom window = bounding box of EMS union labels + 2 km margin
ys, xs = np.where(np.asarray(ems_union))
y0, y1 = max(ys.min() - 100, 0), min(ys.max() + 100, ems_union.shape[0])
x0, x1 = max(xs.min() - 100, 0), min(xs.max() + 100, ems_union.shape[1])
sl = dict(y=slice(y0, y1), x=slice(x0, x1))

dbz = db.isel(**sl)
# explicit north-up extent, robust to either y ordering: left,right,bottom,top
yyz = dbz.y.values
extent = [float(dbz.x[0]), float(dbz.x[-1]), float(min(yyz[0], yyz[-1])), float(max(yyz[0], yyz[-1]))]
imargs = dict(extent=extent, origin="lower" if yyz[0] < yyz[-1] else "upper", aspect="equal")

fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), layout="constrained")
axes[0].imshow(dbz.values, cmap="gray", vmin=-25, vmax=0, **imargs)
axes[0].set_title(f"σ⁰ dB — {k}")
axes[1].imshow(np.where(pred.isel(**sl).values, 1.0, np.nan),
               cmap="Blues", vmin=0, vmax=1.5, **imargs)
axes[1].set_title(f"detected water−permanent ({best['method']})")
p = pred.isel(**sl).values; tr = truth.isel(**sl).values
overlay = np.zeros(p.shape + (4,))
overlay[p & tr] = (0, 0.7, 0, 0.9)     # green TP
overlay[p & ~tr] = (0.95, 0.6, 0, 0.5) # orange FP
overlay[~p & tr] = (0.85, 0, 0, 0.9)   # red FN
axes[2].imshow(overlay, **imargs)
axes[2].set_title(f"vs EMS union — IoU {best['IoU_union']} (offset to nearest: {best['offset_h']} h)")
for j, ax in enumerate(axes):
    ax.set_ylim(extent[2], extent[3])             # pin north up
    if j: ax.set_yticklabels([])\
"""))

cells.append(M("""\
## Takeaways (final run of 2026-07-16, full manifest on disk)

- The σ⁰ → geocode → threshold pipeline runs end-to-end on all 11 scenes at
  ~50 s/scene (sarsen GTC, AOI-cropped, 20 m grid, peak ~6.6 GB RSS).
- The EMS AOI is a **57 km² valley corridor**; 18 km² of it is permanent water.
  The EMS flood labels are small: 1.9 km² (FEP, Sept 10) → 0.05 km² observed +
  1.06 km² cumulative max extent (MONIT02, Sept 15). MONIT products are scored
  against observedEvent ∪ maximumFloodExtent.
- **The zero-offset baseline numbers (this is what WP4 must beat):**
  log-ratio at the FEP anchor (Sept 10, offset 0 h) scores **IoU 0.047 / F1 0.089**;
  at the MONIT02 anchor (Sept 15) **IoU 0.026 / F1 0.051**. The failure mode is
  plain **over-detection**: ~9 km² of "new dark surface" against 1.4 km² of labelled
  flood — the capped −3 dB log-ratio threshold sweeps in wet fields, shadows and
  geometry noise. Time offset was the excuse before the downloads; it no longer is.
- A pre-event scene (Sept 3) "detects" 9.3 km² by single-image Otsu — a good
  null-test showing how much of the dark-surface signal is not flood. The log-ratio
  method correctly returns ~0 for the other pre-event scene (Sept 4 is its own
  reference epoch).
- These are deliberately untuned baselines (fixed caps, no despeckling, no slope
  mask, GLO-30 DEM). The point of the table is an honest floor with provenance,
  not a competitive flood mapper.\
"""))

cells.append(M("""\
## Flood evolution in the labelled valley — Copernicus EMS labels on the raw GRD

Same zoom window as above, chronological, **no detection of our own** — each panel is
just the σ⁰ image (gray, water = dark) with the **Copernicus EMS flood polygons**
overlaid (red fill = union of all EMS flood layers; gold outline = the EMS AOI
corridor). The point is to see the official flood delineation against what the radar
actually measured on each date:

- **Sept 3–4 (pre-event)**: the red areas are still bright/gray in σ⁰ — dry land.
- **Sept 7–10 (rising water and peak — discharge crested Sept 10–11)**: the red areas
  turn dark — smooth open water where EMS later mapped the flood.
- **Sept 15–21 (recession)**: the red areas brighten again as water drains, leaving
  only the permanent river channel dark.

The panel title shows each scene's mean σ⁰ inside the EMS flood area — the single
number that tracks this darkening/brightening.\
"""))

cells.append(C("""\
# zoom window = bounding box of EMS union labels + 2 km margin (as in the figure above)
ys, xs = np.where(np.asarray(ems_union))
y0, y1 = max(ys.min() - 100, 0), min(ys.max() + 100, ems_union.shape[0])
x0, x1 = max(xs.min() - 100, 0), min(xs.max() + 100, ems_union.shape[1])
zoom = dict(y=slice(y0, y1), x=slice(x0, x1))
ems_zoom = np.asarray(ems_union.isel(**zoom))
aoi_zoom = np.asarray(aoi_mask.isel(**zoom))

# explicit north-up extent so the maps match a normal map view (Google-Maps-like:
# north up, west left); robust to either y ordering (sarsen writes south-to-north)
tmpl = scenes[keys[0]]["da"].isel(**zoom)
yyt = tmpl.y.values
extent = [float(tmpl.x[0]), float(tmpl.x[-1]), float(min(yyt[0], yyt[-1])), float(max(yyt[0], yyt[-1]))]
imargs = dict(extent=extent, origin="lower" if yyt[0] < yyt[-1] else "upper", aspect="equal")
xg, yg = tmpl.x.values, tmpl.y.values

ncols, nrows = 3, -(-len(keys) // 3)
fig, axes = plt.subplots(nrows, ncols, figsize=(15, 3.4 * nrows), layout="constrained")
ems_full = np.asarray(ems_union)  # for the per-scene mean-sigma0-in-flood-area number
for i, (ax, k) in enumerate(zip(np.ravel(axes), keys)):
    # background: sigma0 as measured (terrain context, not masked, no detection)
    db_full = flood.to_db(scenes[k]["da"])
    db = db_full.isel(**zoom)
    ax.imshow(db.values, cmap="gray", vmin=-25, vmax=0, **imargs)
    ax.contour(xg, yg, aoi_zoom, levels=[0.5], colors="gold", linewidths=0.5, alpha=0.8)
    fill = np.zeros(ems_zoom.shape + (4,))
    fill[ems_zoom.astype(bool)] = (0.9, 0.1, 0.1, 0.45)   # red: EMS flood union
    ax.imshow(fill, **imargs)
    ax.contour(xg, yg, ems_zoom, levels=[0.5], colors="red", linewidths=0.4, alpha=0.9)
    mean_db = float(db_full.values[ems_full.astype(bool)][np.isfinite(db_full.values[ems_full.astype(bool)])].mean())
    ax.set_title(f"{k}   σ⁰ in EMS flood area: {mean_db:.1f} dB", fontsize=9)
    ax.set_ylim(extent[2], extent[3])             # pin north up
    if i % ncols: ax.set_yticklabels([])
    if i // ncols < nrows - 1: ax.set_xticklabels([])
for ax in np.ravel(axes)[len(keys):]:
    ax.axis("off")
fig.suptitle("Copernicus EMS flood union (red) on Sentinel-1 σ⁰ — north up, chronological (water = dark)")
None\
"""))

cells.append(M("""\
## When: the radar signal inside the EMS flood area, over time

No detection here either — one number per scene: the **mean σ⁰ (dB) inside the EMS
flood-union polygons**. Open water is radar-dark, so this curve dips exactly while
the area is inundated and recovers as it drains. Plotted against the river discharge
and the EMS label times. This is the "when"; the panel grid above is the "where".\
"""))

cells.append(C("""\
t_scene = [pd.Timestamp(scenes[k]["meta"].start) for k in keys]
ems_bool = np.asarray(ems_union).astype(bool)
mean_db_flood, mean_db_dryref = [], []
dry_ref = np.asarray(aoi_mask).astype(bool) & ~ems_bool & ~np.asarray(perm_water).astype(bool)
for k in keys:
    v = flood.to_db(scenes[k]["da"]).values
    x = v[ems_bool];  mean_db_flood.append(float(x[np.isfinite(x)].mean()))
    x = v[dry_ref];   mean_db_dryref.append(float(x[np.isfinite(x)].mean()))

fig, ax = plt.subplots(figsize=(12, 4.5), layout="constrained")
ax.plot(t_scene, mean_db_flood, "o-", color="crimson", label="mean σ⁰ inside EMS flood area")
ax.plot(t_scene, mean_db_dryref, "s--", color="gray", alpha=0.7,
        label="mean σ⁰ in never-flooded AOI land (control)")
for title, v in labels.items():
    ax.axvline(v["time"], color="crimson", lw=0.8, ls=":", alpha=0.7)
    ax.annotate(title.split(" (")[0], (v["time"], ax.get_ylim()[1]), fontsize=8,
                color="crimson", rotation=90, va="top", xytext=(3, -2), textcoords="offset points")

# discharge on a secondary axis for the hydrological "when"
rows_q = [ln.split(";") for ln in
          (ROOT / "data/smhi/discharge_daily_1583_vastersel.csv").read_text().splitlines()
          if ln.startswith(("2025-08-2", "2025-08-3", "2025-09", "2025-10-0"))]
q = pd.Series({pd.Timestamp(p[0]): float(p[1]) for p in rows_q
               if len(p) > 1 and p[1].replace(".", "", 1).isdigit()})
ax2 = ax.twinx()
ax2.plot(q.index, q.values, color="tab:cyan", lw=1.5, alpha=0.8)
ax2.set_ylabel("Moälven discharge [m³/s]", color="tab:cyan")
ax2.tick_params(axis="y", colors="tab:cyan")

ax.set_ylabel("mean σ⁰ [dB]")
ax.set_title("Radar darkening inside the EMS flood area (red) vs dry-land control (gray) vs discharge (cyan)")
ax.legend(loc="lower left", fontsize=8)
fig.autofmt_xdate()
None\
"""))

cells.append(M("""\
Read together: the red curve (radar signal where Copernicus mapped flood) drops by
several dB exactly as the discharge wave arrives and recovers through the recession,
while the gray control (never-flooded land in the same images) stays flat — so the
dip is water on the ground, not a scene-brightness or geometry artefact. Every
number here comes straight from the GRD pixels and the EMS polygons; no detection
algorithm of ours is involved.\
"""))

nb.cells = cells
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
out = "notebooks/flood_timeseries.ipynb"
nbf.write(nb, out)
print("wrote", out)
