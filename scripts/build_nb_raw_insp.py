#!/usr/bin/env python
"""Author notebooks/03_raw_inspection.ipynb."""

import nbformat as nbf

nb = nbf.v4.new_notebook()
C, M = nbf.v4.new_code_cell, nbf.v4.new_markdown_cell
cells = []

cells.append(M("""\
# Cracking open Sentinel-1 Level-0

A first honest look inside unfocused L0 I/Q for the Örnsköldsvik POC.
Primary product here is **S1C Sept 9 (rel 22 DESC, datatake 00809C)** — the
label-anchor track, with its same-datatake SLC on disk for comparison —
plus **S1A Sept 10 (rel 131 ASC, datatake 0796C3)** as the known-decoder-territory
cross-check. That S1A pass is the plan's first-choice target: it is the **FEP
label-source acquisition** (zero label-to-acquisition time offset).

Decoder: `sentinel1decoder` 2.0.0 (Rust core). **De-risk result up front: S1C L0
decodes fine** — metadata, ephemeris and FDBAQ echoes all parse (details below);
the plan's fallback (`s1isp`) was not needed.\
"""))

cells.append(C("""\
import sys, glob, time, zipfile
from pathlib import Path
sys.path.insert(0, "../src")
import warnings; warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sentinel1decoder as s1d
from rawsar import l0

ROOT = Path("..").resolve()
S1C_SAFE = Path(glob.glob(str(ROOT / "data/extracted/S1C_IW_RAW*00809C*.SAFE"))[0])
# S1A: prefer the Sept 10 rel-131 FEP-anchor pass; fall back to any extracted S1A RAW
s1a_all = sorted(glob.glob(str(ROOT / "data/extracted/S1A_IW_RAW*.SAFE")))
S1A_SAFE = Path(next((p for p in s1a_all if "20250910" in p), s1a_all[0]))
plt.rcParams["figure.dpi"] = 110
print("S1C:", S1C_SAFE.name)
print("S1A:", S1A_SAFE.name)\
"""))

cells.append(M("""\
## L0 SAFE layout

A Level-0 SAFE is tiny in structure: a manifest, and per polarisation a measurement
`.dat` (the CCSDS space-packet stream — the actual downlink), an `-annot.dat`
(fixed-size per-packet annotations added on ground) and an `-index.dat`.\
"""))

cells.append(C("""\
for p in sorted(S1C_SAFE.rglob("*")):
    if p.is_file():
        print(f"{p.stat().st_size/1e6:9.1f} MB  {p.relative_to(S1C_SAFE)}")\
"""))

cells.append(M("""\
## Packet headers: what the instrument tells us per pulse

Every radar pulse (PRI) produces one space packet. The secondary header carries the
commanded radar configuration — this is everything a raw-domain model can know
about a line without decoding samples.\
"""))

cells.append(C("""\
f_c = l0.open_l0(S1C_SAFE)     # S1C VV
md = f_c.packet_metadata
print(f"{len(md)} packets, {len(md.columns)} header fields")
print("Signal types:", {str(k): int(v) for k, v in md['Signal Type'].value_counts().items()})
print("Swaths:", {str(k): int(v) for k, v in md['Swath Number'].value_counts().items()},
      " (10/11/12 = IW1/2/3, 93/43 = cal/noise ops)")
print("BAQ modes:", {str(k): int(v) for k, v in md['BAQ Mode'].value_counts().items()})

echo = md[md["Signal Type"].astype(str) == "Echo"]
per_swath = echo.groupby(echo["Swath Number"].astype(int)).agg(
    packets=("PRI", "size"),
    PRF_Hz=("PRI", lambda s: 1 / s.astype(float).mean()),
    SWST_ms=("SWST", lambda s: 1e3 * s.astype(float).mean()),
    SWL_ms=("SWL", lambda s: 1e3 * s.astype(float).mean()),
    rank=("Rank", lambda s: int(s.astype(int).mode()[0])),
    n_quads=("Number of Quads", lambda s: int(s.astype(int).mode()[0])),
)
per_swath\
"""))

cells.append(M("""\
The three IW sub-swaths run different PRFs, sampling windows and ranks (the rank is
how many PRIs ago the received pulse was transmitted — light travel time exceeds
the PRI at these ranges). `Number of Quads` × 2 = complex samples per echo line.

**IW bursts vs packets:** azimuth bursts are ~1.2 s blocks of consecutive echoes in
one sub-swath; the instrument cycles IW1→IW2→IW3. `sentinel1decoder` groups the
stream into "acquisition chunks" (contiguous same-configuration packet runs); a
chunk boundary is *not* a CCSDS framing feature — the packets just keep counting.\
"""))

cells.append(C("""\
track = l0.chunk_subsatellite_track(f_c)
track["duration_s"] = track.n_packets * float(md["PRI"].astype(float).mean())
print(track[track.signal == "Echo"].head(12))
aoi_lat = (63.26, 63.55)
near = track[(track.signal == "Echo") & (track.subsat_lat > 62.8) & (track.subsat_lat < 64.0)]
print(f"\\nEcho chunks with sub-satellite latitude near the AOI ({aoi_lat}):")
print(near)\
"""))

cells.append(M("""\
> The sub-satellite point is orbit-only geolocation. The imaged swath lies ~250–400 km
> to the right of it, and each azimuth *sample* mixes returns from a ±2–3 km synthetic
> aperture. This table answers "roughly which seconds of the datatake matter for the
> AOI" — the label-attachment question proper is Phase 6.

## Sub-commutated ephemeris

Position/velocity/attitude arrive spread across 64 consecutive packets (1 word each);
the decoder reassembles them.\
"""))

cells.append(C("""\
eph = f_c.ephemeris
r = np.linalg.norm(eph[[f"{a}-axis position ECEF" for a in "XYZ"]].to_numpy(float), axis=1)
v = np.linalg.norm(eph[[f"{a}-axis velocity ECEF" for a in "XYZ"]].to_numpy(float), axis=1)
print(f"{len(eph)} ephemeris solutions | orbit radius {r.mean()/1e3:.0f} km | speed {v.mean():.0f} m/s")
eph.head(3)\
"""))

cells.append(M("""\
## Decoding echoes — and the S1C question

FDBAQ decode of full echo chunks, S1A and S1C side by side. This was the plan's
flagged risk ("S1C support unverified"); measured result below.\
"""))

cells.append(C("""\
f_a = l0.open_l0(S1A_SAFE)
stats = []
decoded = {}
s1a_tag = f"S1A ({S1A_SAFE.name.split('_')[5][:8]})"
for tag, f in [("S1C (rel 22, Sept 9)", f_c), (s1a_tag, f_a)]:
    ids = l0.echo_chunk_ids(f, swath=10)
    i = ids[len(ids) // 2]
    m = f.get_acquisition_chunk_metadata(i)
    t0 = time.time()
    d = f.get_acquisition_chunk_data(i, try_load_from_file=False)
    dt = time.time() - t0
    decoded[tag] = (d, m)
    npk = len(m)
    stats.append({"product": tag, "chunk": i, "lines": d.shape[0], "samples": d.shape[1],
                  "decode_s": round(dt, 2), "MB": round(d.nbytes / 1e6, 1),
                  "mean_I": round(float(d.real.mean()), 3), "std": round(float(d.std()), 1)})
pd.DataFrame(stats)\
"""))

cells.append(C("""\
# What raw echoes look like: amplitude image + I/Q histogram (FDBAQ reconstruction levels)
fig, axes = plt.subplots(2, 2, figsize=(14, 8), layout="constrained")
for row, (tag, (d, m)) in enumerate(decoded.items()):
    a = np.abs(d)
    axes[row, 0].imshow(a, cmap="gray", vmax=np.percentile(a, 99), aspect="auto")
    axes[row, 0].set_title(f"{tag} — |raw echo| (az×rg {d.shape[0]}×{d.shape[1]})")
    axes[row, 0].set_ylabel("azimuth line")
    axes[row, 1].hist(d.real[::7, ::13].ravel(), bins=201, alpha=0.6, label="I")
    axes[row, 1].hist(d.imag[::7, ::13].ravel(), bins=201, alpha=0.6, label="Q")
    axes[row, 1].set_title("I/Q sample histogram"); axes[row, 1].legend(); axes[row, 1].set_yscale("log")
axes[1, 0].set_xlabel("range sample")
None\
"""))

cells.append(M("""\
Unfocused echoes are visually structureless (Gaussian-like speckle sum over the whole
antenna footprint) — the information is there but spread by the range chirp and the
synthetic aperture. The histograms show FDBAQ's non-uniform reconstruction levels:
this is decoded *quantised thermal-noise-plus-clutter*, high-entropy by design
(relevant to Phase 5 compression measurements).

## Per-line and per-packet diagnostics\
"""))

cells.append(C("""\
d, m = decoded["S1C (rel 22, Sept 9)"]
fig, axes = plt.subplots(1, 3, figsize=(15, 4), layout="constrained")
axes[0].plot(np.abs(d).mean(axis=1))
axes[0].set(title="mean |sample| per line (azimuth)", xlabel="line")
axes[1].plot(np.abs(d).mean(axis=0))
axes[1].set(title="mean |sample| per range sample", xlabel="range sample")
lens = m["Packet Data Length"].to_numpy(int)
axes[2].plot(lens)
axes[2].set(title="packet length per line (FDBAQ = variable rate)", xlabel="line", ylabel="bytes")
None\
"""))

cells.append(M("""\
Packet length per line is the visible fingerprint of FDBAQ's scene-adaptive bit-rate
(brighter clutter → more bits). Per-block BRC codes are not exposed by
`sentinel1decoder`'s public API — noted for `s1isp` if a BRC map becomes necessary.

## Minimal range compression + small-patch azimuth focus

Matched filter from the *nominal* chirp parameters in the headers (TXPSF/TXPRR/TXPL),
then a deliberately naive azimuth compression (zero Doppler centroid, no RCMC, no
TOPS deramp) on a small patch. Proof of decode, not a processor.\
"""))

cells.append(C("""\
r0 = m.iloc[0]
rep, fs = l0.nominal_chirp(r0)
print(f"chirp: {len(rep)} samples @ {fs/1e6:.2f} MHz, TXPL={float(r0['Tx Pulse Length'])*1e6:.1f} µs, "
      f"bandwidth≈{abs(float(r0['Tx Ramp Rate'])) * float(r0['Tx Pulse Length'])/1e6:.1f} MHz")

nbytes_chunk = int(m["Packet Data Length"].astype(int).sum() + 6 * len(m))
t0 = time.time(); rc = l0.range_compress(d, rep); t_rc = time.time() - t0

prf = 1 / float(r0["PRI"])
sr = l0.slant_ranges(r0, d.shape[1], fs)
v_eff = l0.effective_velocity(f_c.ephemeris)
t0 = time.time(); foc = l0.azimuth_focus_patch(rc, prf, v_eff, sr); t_az = time.time() - t0

def kurt(x):
    a = np.abs(x).ravel(); a = (a - a.mean()) / a.std(); return float((a ** 4).mean())
print(f"range compression: {t_rc:.2f}s | azimuth focus: {t_az:.2f}s")
print(f"kurtosis (peakiness): raw {kurt(d):.1f} → range-compressed {kurt(rc):.1f} → focused {kurt(foc):.1f}")\
"""))

cells.append(C("""\
fig, axes = plt.subplots(1, 3, figsize=(16, 5), layout="constrained")
for ax, (img, title) in zip(axes, [(d, "raw"), (rc, "range-compressed"), (foc, "quick-focused")]):
    a = np.abs(img)
    ax.imshow(a, cmap="gray", vmax=np.percentile(a, 99), aspect="auto")
    ax.set_title(title)
axes[0].set_ylabel("azimuth line")
for ax in axes: ax.set_xlabel("range sample")
None\
"""))

cells.append(M("## Sanity check against the operational SLC (same datatake)"))

cells.append(C("""\
import xarray as xr
# same-datatake SLC — already extracted on disk (rasterio can't read measurement tiffs from zips)
slc_safe = glob.glob(str(ROOT / "data/copernicus_data/S1/S1C_IW_SLC__1SDV_20250909T050504*7A6D.SAFE"))[0]
slc = xr.open_dataset(slc_safe, engine="sentinel-1", group="IW1/VV")
print("SLC IW1/VV mosaic:", dict(slc.sizes), "— ~9 TOPS bursts stacked in line order")
# our decoded chunk sits mid-datatake; take the corresponding mid-scene SLC patch
lc = slc.sizes["line"] // 2
sl = slc.measurement.isel(line=slice(lc, lc + 1500), pixel=slice(0, 4000))
a = np.abs(sl.values)
fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout="constrained")
axes[0].imshow(np.abs(foc[:, :4000]), cmap="gray",
               vmax=np.percentile(np.abs(foc[:, :4000]), 99), aspect="auto")
axes[0].set_title("our quick focus (patch)")
axes[1].imshow(a, cmap="gray", vmax=np.percentile(a, 99), aspect="auto")
axes[1].set_title("operational IW1 SLC burst (patch)")
None\
"""))

cells.append(M("""\
Expected outcome, stated plainly: large-scale energy distribution and range structure
should correspond; our patch stays visibly softer in azimuth because the quick focus
ignores TOPS steering, Doppler centroid and RCMC. Geometry agreement is at the
"same coastline/valley visible" level, not sample-level.

## OBJ5 seed numbers: bytes / wall-time / rough FLOPs\
"""))

cells.append(C("""\
n_az, n_rg = d.shape
nfft_rg = int(2 ** np.ceil(np.log2(n_rg + len(rep) - 1)))
flops_rc = n_az * 3 * 5 * nfft_rg * np.log2(nfft_rg)      # fwd+inv FFT + mult, 5N log N per FFT
flops_az = n_rg * 2 * 5 * n_az * np.log2(n_az) + 6 * n_az * n_rg
rows = [
    {"stage": "FDBAQ decode (Rust)", "input_MB": round(nbytes_chunk / 1e6, 1),
     "output_MB": round(d.nbytes / 1e6, 1), "wall_s": stats[0]["decode_s"], "GFLOP": None},
    {"stage": "range compression", "input_MB": round(d.nbytes / 1e6, 1),
     "output_MB": round(rc.nbytes / 1e6, 1), "wall_s": round(t_rc, 2), "GFLOP": round(flops_rc / 1e9, 1)},
    {"stage": "azimuth focus (naive)", "input_MB": round(rc.nbytes / 1e6, 1),
     "output_MB": round(foc.nbytes / 1e6, 1), "wall_s": round(t_az, 2), "GFLOP": round(flops_az / 1e9, 1)},
]
seed = pd.DataFrame(rows)
seed.to_csv(ROOT / "docs/obj5_seed_numbers.csv", index=False)
print("one IW1 burst-sized chunk of one polarisation; container CPU (linux/arm64), numpy FFT, single process")
seed\
"""))

cells.append(M("""\
## Findings

1. **S1C L0 decodes with `sentinel1decoder` 2.0.0** — metadata, ephemeris, FDBAQ
   echoes; enum names even carry S1C/S1D cases. The plan's flagged risk is retired;
   `s1isp` remains an optional cross-check, not a dependency.
2. The compressed downlink expands ~5× on FDBAQ decode (input MB → complex64 MB
   above); raw-domain models would ingest quantised, high-entropy samples.
3. Everything a model can know per line without decoding samples is in the packet
   headers (PRF/SWST/SWL/rank/BAQ/beam addresses + sub-commutated ephemeris) —
   that is the metadata a raw-echo dataset must carry (Phase 5 schema input).
4. Range compression is cheap and already reveals structure (kurtosis jump);
   azimuth is where the real processor cost and the TOPS complications live.
5. Decode→compress→naive-focus wall-times and FLOP estimates are in
   `docs/obj5_seed_numbers.csv` (Apple-Silicon-class container CPU — caveat for OBJ5).\
"""))

nb.cells = cells
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, "notebooks/raw_inspection.ipynb")
print("wrote notebooks/raw_inspection.ipynb")
