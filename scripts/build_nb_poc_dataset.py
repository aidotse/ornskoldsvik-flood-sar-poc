#!/usr/bin/env python
"""Author notebooks/poc_dataset.ipynb — open the POC store, batch it, plot it."""

import nbformat as nbf

nb = nbf.v4.new_notebook()
C, M = nbf.v4.new_code_cell, nbf.v4.new_markdown_cell
cells = []

cells.append(M("""\
# The POC dataset

One coherent zarr-v3 store (`data/zarr/poc.zarr`, ~0.55 GB) holding everything the
pre-study needs in one place, for one event:

| group | content |
|---|---|
| `sigma0/track_*` | geocoded VV σ⁰ stacks per relative orbit (time,y,x), 20 m UTM33 |
| `labels/` | EMS flood/AOI rasters on the σ⁰ grid + **raw-geometry label windows** |
| `raw/00809C_vv/chunk_*` | decoded I/Q for the 10 AOI echo chunks of the rel-22 pass + per-line headers |
| `tables/smhi` | precipitation (2 stations) + Moälven discharge |

Built by `scripts/build_zarr_store.py`; codec choices are the measured winners from
`scripts/bench_codecs.py` (zstd3 for I/Q, blosc-zstd3+byteshuffle for rasters).\
"""))

cells.append(C("""\
import sys, warnings
from pathlib import Path
sys.path.insert(0, "../src")
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import xarray as xr
import zarr
import xbatcher
import matplotlib.pyplot as plt

STORE = Path("../data/zarr/poc.zarr").resolve()
root = zarr.open_group(STORE, mode="r")
plt.rcParams["figure.dpi"] = 110
print(root.attrs.asdict())
print("groups:", sorted(root.group_keys()))\
"""))

cells.append(M("## An ML batch, straight from the store"))

cells.append(C("""\
ds = xr.open_zarr(STORE, group="sigma0/track_022")
labels = root["labels/ems_flood_union"][:]

da = ds.sigma0_vv
lab_da = xr.DataArray(labels, dims=("y", "x"), coords={"y": da.y, "x": da.x})
stacked = xr.Dataset({"sigma0": da.isel(time=0), "flood": lab_da})
gen = xbatcher.BatchGenerator(stacked, input_dims={"x": 256, "y": 256})
batches = list(gen)
positives = [b for b in batches if float(b.flood.sum()) > 0]
print(f"{len(batches)} chips of 256², {len(positives)} contain flood labels")

fig, axes = plt.subplots(2, 4, figsize=(14, 7), layout="constrained")
for ax, b in zip(np.ravel(axes), positives[:8]):
    db = 10 * np.log10(b.sigma0.where(b.sigma0 > 0))
    ax.imshow(db, cmap="gray", vmin=-25, vmax=0)
    m = np.ma.masked_where(~b.flood.values.astype(bool), b.flood.values)
    ax.imshow(m, cmap="autumn", alpha=0.8, vmin=0, vmax=1)
    ax.set_title(f"flood px: {int(b.flood.sum())}", fontsize=9)
    ax.axis("off")
fig.suptitle("xbatcher chips: σ⁰ (Sept 9, rel 22) + EMS flood-union label overlay")
None\
"""))

cells.append(M("""\
## The raw side of the same event

The store's raw group holds the decoded echoes whose focused image would show the
scene above — plus the label windows that tie EMS polygons to (azimuth-time,
slant-range) regions of the unfocused signal.\
"""))

cells.append(C("""\
raw = root["raw/00809C_vv"]
rows = []
for k in sorted(raw.group_keys()):
    g = raw[k]
    rows.append({"chunk": k, "swath": g.attrs["swath"],
                 "lines": g.attrs["n_lines"], "samples": g.attrs["n_samples"],
                 "MB_complex64": round(g["iq"].nbytes / 1e6, 0)})
pd.DataFrame(rows)\
"""))

cells.append(C("""\
# label windows in raw geometry (strategy b), vs the chunk azimuth extents
win_grp = root["labels/raw_windows_00809C"]
w = win_grp["DEL_MONIT02_maximumFloodExtent"][:]
az0 = pd.to_datetime(w[:, 0].astype("int64"))
az1 = pd.to_datetime(w[:, 1].astype("int64"))
GPS_EPOCH, LEAP = pd.Timestamp("1980-01-06"), pd.Timedelta(seconds=18)

fig, ax = plt.subplots(figsize=(12, 4.5), layout="constrained")
for i in range(len(w)):
    ax.plot([az0[i], az1[i]], [w[i, 2] * 1e3, w[i, 2] * 1e3], lw=0.8, color="crimson", alpha=0.5)
for k in sorted(raw.group_keys()):
    g = raw[k]
    ct = g["coarse_time"][:] + g["fine_time"][:]
    t0 = GPS_EPOCH + pd.Timedelta(seconds=float(ct.min())) - LEAP
    t1 = GPS_EPOCH + pd.Timedelta(seconds=float(ct.max())) - LEAP
    pri, swst, rank = float(g["pri"][:].mean()), float(g["swst"][:].mean()), float(g["rank"][:].mean())
    slant_ms = (rank * pri + swst) * 1e3
    ax.plot([t0, t1], [slant_ms, slant_ms], lw=5, alpha=0.6,
            color={"10": "tab:blue", "11": "tab:green", "12": "tab:orange"}[g.attrs["swath"]])
    ax.annotate(f"{k[-3:]} IW{int(g.attrs['swath'])-9}", (t0, slant_ms), fontsize=7,
                xytext=(0, 4), textcoords="offset points")
ax.set(xlabel="azimuth time (UTC)", ylabel="two-way slant time [ms]",
       title="EMS flood polygons (red) and stored echo chunks (thick) in RAW coordinates")
None\
"""))

cells.append(M("""\
Red strokes are the 426 MONIT02 flood polygons projected into raw coordinates
(each dilated by the ±0.4 s aperture kernel and the 52 µs chirp smear); thick bars
are the stored echo chunks per sub-swath. The flood sits in the far-range chunks —
a raw-domain model needs exactly those (azimuth, slant) neighbourhoods as input,
and this figure *is* the weak-label assignment.

## Event context from the same store\
"""))

cells.append(C("""\
t = xr.open_zarr(STORE, group="tables/smhi")
fig, ax = plt.subplots(figsize=(11, 3.5), layout="constrained")
ax.bar(t.time, t.precip_ornskoldsvik_mm, width=0.8, alpha=0.7, label="precip Örnsköldsvik [mm]")
ax2 = ax.twinx()
ax2.plot(t.time, t.discharge_vastersel_m3s, color="tab:blue", lw=2, label="Moälven discharge [m³/s]")
ax2.set_ylabel("m³/s", color="tab:blue")
ax.set_ylabel("mm/day")
ax.set_title("Same store, same event: rain and the 9× discharge peak (Sept 11)")
fig.legend(loc="upper left", bbox_to_anchor=(0.08, 0.95), fontsize=8)
None\
"""))

cells.append(M("""\
## What this store demonstrates

1. **Colocated processed + raw + labels + context** for one event, openable with
   nothing but `zarr`/`xarray` — the shape of the open dataset the application
   promises (align group naming with EOPF's `measurements/conditions` split before
   publishing).
2. Chip-oriented chunking serves ML reads directly (xbatcher above); raw chunks are
   line-block-oriented for signal-domain consumers.
3. The weak-label pathway (map polygons → raw windows) is concrete, stored, and
   backed by the measured 0.66 km geolocation error bars (`docs/labels.md`).
4. What's missing, deliberately: the two zero-offset anchor passes (in the pending
   download manifest) — the store schema already has their slots (per-track sigma0
   time dimension, per-datatake raw groups).\
"""))

nb.cells = cells
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, "notebooks/poc_dataset.ipynb")
print("wrote notebooks/poc_dataset.ipynb")
