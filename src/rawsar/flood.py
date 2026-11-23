"""Flood mapping baseline: thresholding, log-ratio change detection, EMS label metrics."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import geopandas as gpd
import numpy as np
import xarray as xr
from rasterio import features


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


def despeckle(db: xr.DataArray, size: int = 3) -> xr.DataArray:
    """Median-filter speckle reduction on a dB image, applied before thresholding.

    NaNs (outside-AOI / nodata) are temporarily filled with their nearest finite
    neighbour, so the fill never drags the median of AOI-edge pixels (the boundary
    behaves like scipy's mode="nearest"; a 0 dB fill brightened the AOI rim and erased
    convex corners), and are re-inserted afterwards.
    """
    from scipy.ndimage import distance_transform_edt, median_filter

    v = db.values
    finite = np.isfinite(v)
    if not finite.any():
        return xr.DataArray(v.copy(), coords=db.coords, dims=db.dims)
    filled = v
    if not finite.all():
        nearest = distance_transform_edt(~finite, return_distances=False, return_indices=True)
        filled = v[tuple(nearest)]
    filtered = median_filter(filled, size=size)
    filtered = np.where(finite, filtered, np.nan)
    return xr.DataArray(filtered, coords=db.coords, dims=db.dims)


def slope_deg(dem: xr.DataArray) -> xr.DataArray:
    """Terrain slope in degrees from a DEM on a regular grid (resolution via dem.rio)."""
    res_x, res_y = dem.rio.resolution()
    gy, gx = np.gradient(dem.values, abs(res_y), abs(res_x))
    slope = np.degrees(np.arctan(np.hypot(gx, gy)))
    return xr.DataArray(slope, coords=dem.coords, dims=dem.dims)


def water_mask_adaptive(signal: xr.DataArray, block: int = 100, k: float = 2.0) -> tuple[xr.DataArray, float]:
    """Locally-adaptive dark-pixel mask: water where pixel < local_mean - k*local_std.

    Unlike water_mask_single's one global Otsu cutoff, the threshold varies spatially
    (box-filter local mean/std via scipy.ndimage.uniform_filter) — useful where the
    backscatter baseline itself shifts across the scene (soil/vegetation/incidence
    angle), so a single global number either misses subtly-darker regions or
    over-triggers in naturally-darker ones.

    Local mean/std use only the finite pixels of each window (box sums normalised by the
    count of finite pixels), so NaN (outside-AOI) pixels never enter the statistics. A
    zero fill would bias every pixel here, not just the edge: the 100-px window is wider
    than the ~60-px AOI corridor. The returned threshold is the mean of the thresholds
    actually applied to the finite pixels.
    """
    from scipy.ndimage import uniform_filter

    v = signal.values
    finite = np.isfinite(v)
    filled = np.where(finite, v, 0.0).astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        count = uniform_filter(finite.astype(np.float64), size=block)
        local_mean = uniform_filter(filled, size=block) / count
        local_mean_sq = uniform_filter(filled**2, size=block) / count
        local_std = np.sqrt(np.clip(local_mean_sq - local_mean**2, 0, None))
        local_thr = local_mean - k * local_std
    mask = (v < local_thr) & finite
    thr = float(local_thr[finite].mean()) if finite.any() else float("nan")
    return xr.DataArray(mask, coords=signal.coords, dims=signal.dims), thr


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
        n_valid = int(m.sum())
    else:
        n_valid = p.size
    tp = int((p & t).sum())
    fp = int((p & ~t).sum())
    fn = int((~p & t).sum())
    tn = n_valid - tp - fp - fn
    iou = tp / (tp + fp + fn) if tp + fp + fn else float("nan")
    f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
    # precision/recall/fpr matter most for *ever-detected* scoring (see
    # temporal_union): recall = how much of an ever-flooded reference we ever
    # capture, fpr = how much of the never-flooded area we ever mis-flag.
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    fpr = fp / (fp + tn) if (fp + tn) else float("nan")
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "iou": iou, "f1": f1,
            "precision": precision, "recall": recall, "fpr": fpr,
            "pred_area_km2": p.sum() * 0.0004, "truth_area_km2": t.sum() * 0.0004}


def load_ems_layer(path: str | Path) -> gpd.GeoDataFrame:
    return gpd.read_file(path)


def temporal_union(masks: Iterable[xr.DataArray]) -> xr.DataArray:
    """OR-reduce boolean masks across time: True wherever *any* scene flagged water.

    Puts a per-scene detector on the same temporal footing as an EMS label union
    (observedEvent + maximumFloodExtent across products/dates), which already means
    "flooded at some point", not "flooded in this one image". Scoring a single
    scene's mask against that union unfairly penalises real flood that had already
    receded by that pass; unioning detections over time first fixes the comparison.
    """
    masks = list(masks)
    out = masks[0]
    for m in masks[1:]:
        out = out | m
    return out


def detection_frequency(masks: Iterable[xr.DataArray]) -> xr.DataArray:
    """Fraction of scenes each pixel was flagged as water, in [0, 1].

    A persistence diagnostic: real flood tends to be flagged repeatedly across
    during-event passes, while speckle/geometry false positives tend to flicker
    in a single scene and vanish in the next.
    """
    masks = list(masks)
    stacked = np.stack([np.asarray(m.values, bool) for m in masks], axis=0)
    freq = stacked.mean(axis=0)
    return xr.DataArray(freq, coords=masks[0].coords, dims=masks[0].dims)


def top_hotspot_windows(hit_mask: np.ndarray, window: int = 60, n: int = 3) -> list[tuple[slice, slice]]:
    """Top-n (y, x) window slices ranked by hit density in hit_mask, centers >= window apart.

    Greedy pick over a local-density map (box filter), so picks land on genuinely
    different parts of a corridor-shaped extent instead of collapsing onto one spot.
    """
    from scipy.ndimage import uniform_filter

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
