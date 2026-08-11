# rawsar — Örnsköldsvik flood POC (EMSR841)

Pre-work for the FFI pre-study *"Novel raw Synthetic Aperture Radar data processing
for rapid detection of road conditions"*. This repo uses the Västernorrland
flood (Sept 2025, Copernicus EMS activation **EMSR841**) as a proxy event because it
has published flood maps.

## Setup

```bash
uv venv .venv && uv pip install -e ".[dev]"
cp .env.example .env       # fill in CDSE creds (free account, dataspace.copernicus.eu)
pytest                     # 8 L0 parser tests, run on committed fixtures only
```

## Get the data

```bash
python scripts/fetch_all_data.py           # dry run: lists everything missing
python scripts/fetch_all_data.py --yes     # fetch it all (~55 GB)
python scripts/fetch_all_data.py --skip-s1 --yes   # public sources only, no account
```

One command puts every notebook input on disk, in five stages: the GLO-30 DEM tiles
(public AWS S3), the SMHI precipitation/discharge series (public API), the four EMS
label products (public HTTPS), the 20-product Sentinel-1 manifest (the only stage
needing credentials), and the RAW/SLC SAFE extractions the notebooks glob for. Every
stage is a no-op once its output exists, so re-running costs nothing.

`scripts/download_manifest.py` remains the Sentinel-1-only operator script with the
same `--yes` gate.

## Run it

```bash
python scripts/gtc_all.py            # geocode every GRD to sigma0 (~50 s/scene, ~6.6 GB RAM)
python scripts/build_zarr_store.py   # assemble the POC zarr store
python scripts/build_nb02.py         # author the notebooks (also build_nb03 / build_nb07)
cd notebooks && jupyter nbconvert --to notebook --execute --inplace 02_flood_timeseries.ipynb
```

Notebooks are **generated from the `build_nb0*.py` scripts**, not hand-edited, so the
reviewable source is plain Python rather than opaque `.ipynb` JSON. They are committed
with outputs, so they can be read without re-running anything.

## Layout

- `notebooks/` — `02_flood_timeseries` (GRD σ⁰ + EMS scoring), `03_raw_inspection`
  (inside L0), `07_poc_dataset` (the zarr store). All executed top-to-bottom.
- `src/rawsar/` — `cdse.py` (catalogue search + resumable downloads), `ems.py` (label
  API), `grd.py` (σ⁰ terrain correction via sarsen), `flood.py` (water masks, metrics),
  `l0.py` (raw decode, range compression, orbit geolocation, raw-coordinate labels).
- `scripts/` — every number quoted below traces to one of these.
- `configs/aoi_ornskoldsvik.yaml` — single source of truth for the AOI polygon, event
  windows, target orbits, download manifest, DEM tiles and SMHI stations. Nothing about
  location, dates or products is hardcoded in code.
- `tests/` — L0 parser tests plus ~8 MB of real committed packet fixtures per satellite.
- `data/` — gitignored (~83 GB S1 archive + derived stores), rebuilt by
  `scripts/fetch_all_data.py`.
- `docs/` — measured CSV artifacts and figures: `codec_benchmarks.csv`,
  `label_attach_errorbars.csv`, `obj5_seed_numbers.csv`,
  `flood_state_vs_change_comparison.csv`, `figures/event_timeline.png`.

## The event

Sept 6–7 2025 skyfall over Västernorrland (70–80 mm). Moälven discharge at Västersel,
4.6 km from the labels, rose from a 14 m³/s August mean to a **126 m³/s peak on Sept
11** (≈9×). Roughly 40 roads closed, a train derailment, one fatality on a washed-away
road — the road-condition use case this pre-study targets. Copernicus EMS produced four
flood-delineation products for AOI01; a fifth for a neighbouring AOI was never produced
("remote sensing limitations", straight from the EMS API).

Two EMS products were derived *from* Sentinel-1 passes this repo holds, so those
comparisons carry a **zero label-to-acquisition time offset**: FEP ↔ S1A Sept 10
(rel 131 ASC) and MONIT02 ↔ S1A Sept 15 (rel 22 DESC). Those two tracks are the spine
of the POC.


