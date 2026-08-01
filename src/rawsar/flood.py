"""Flood mapping baseline: thresholding, log-ratio change detection, EMS label metrics."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import xarray as xr
from rasterio import features
from scipy.ndimage import uniform_filter


def to_db(sigma0: xr.DataArray) -> xr.DataArray:
    """dB with non-positive values (nodata borders of partial scenes) masked to NaN."""
    return 10.0 * np.log10(sigma0.where(sigma0 > 0))


def otsu_threshold(values: np.ndarray, bins: int = 256) -> float:
    """Otsu's method on finite values (dB). No skimage dependency needed."""
    v = values[np.isfinite(values)]
    hist, edges = np.histogram(v, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2
    w = hist.astype(float)
    p = w / w.sum()
    omega = np.cumsum(p)
    mu = np.cumsum(p * centers)
    mu_t = mu[-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma_b = (mu_t * omega - mu) ** 2 / (omega * (1 - omega))
    return float(centers[np.nanargmax(sigma_b)])


def water_mask_single(db: xr.DataArray, max_threshold_db: float = -14.0) -> tuple[xr.DataArray, float]:
    """Single-image dark-water mask: Otsu on dB, capped at a physical bound.

    Otsu finds the dominant land/water split; the cap protects against scenes
    where too little water is present for a bimodal histogram.
    """
    thr = min(otsu_threshold(db.values), max_threshold_db)
    return (db < thr) & np.isfinite(db), thr


def water_mask_logratio(
    db_event: xr.DataArray, db_ref: xr.DataArray, max_threshold_db: float = -3.0
) -> tuple[xr.DataArray, float]:
    """Change-detection water mask: event-minus-reference dB drop + Otsu."""
    ratio = db_event - db_ref
    thr = min(otsu_threshold(ratio.values), max_threshold_db)
    return (ratio < thr) & np.isfinite(ratio), thr


def rasterize_labels(gdf: gpd.GeoDataFrame, template: xr.DataArray) -> xr.DataArray:
    """Burn label polygons (any CRS) onto a georeferenced template grid as bool."""
    g = gdf.to_crs(template.rio.crs)
    arr = features.rasterize(
        ((geom, 1) for geom in g.geometry if geom is not None),
        out_shape=template.shape,
        transform=template.rio.transform(),
        fill=0,
        dtype="uint8",
    )
    return xr.DataArray(arr.astype(bool), coords=template.coords, dims=template.dims)


def iou_f1(pred: xr.DataArray, truth: xr.DataArray, valid: xr.DataArray | None = None) -> dict:
    p = np.asarray(pred.values, bool)
    t = np.asarray(truth.values, bool)
    if valid is not None:
        m = np.asarray(valid.values, bool)
        p, t = p & m, t & m
    tp = int((p & t).sum())
    fp = int((p & ~t).sum())
    fn = int((~p & t).sum())
    iou = tp / (tp + fp + fn) if tp + fp + fn else float("nan")
    f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "iou": iou, "f1": f1,
            "pred_area_km2": p.sum() * 0.0004, "truth_area_km2": t.sum() * 0.0004}


def load_ems_layer(path: str | Path) -> gpd.GeoDataFrame:
    return gpd.read_file(path)

def top_hotspot_windows(hit_mask: np.ndarray, window: int = 60, n: int = 3) -> list[tuple[slice, slice]]:
    """Top-n (y, x) window slices ranked by hit density in hit_mask, centers >= window apart.

    Greedy pick over a local-density map (box filter), so picks land on genuinely
    different parts of a corridor-shaped extent instead of collapsing onto one spot.
    """
    

    arr = np.asarray(hit_mask, dtype=float)
    density = uniform_filter(arr, size=window, mode="constant")
    ny, nx = arr.shape
    half = window // 2
    chosen: list[tuple[int, int]] = []
    for idx in np.argsort(density.ravel())[::-1]:
        cy, cx = np.unravel_index(idx, arr.shape)
        if any(abs(cy - py) < window and abs(cx - px) < window for py, px in chosen):
            continue
        chosen.append((int(cy), int(cx)))
        if len(chosen) == n:
            break
    return [(slice(max(cy - half, 0), min(cy + half, ny)),
              slice(max(cx - half, 0), min(cx + half, nx))) for cy, cx in chosen]
