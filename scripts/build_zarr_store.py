#!/usr/bin/env python
"""Build the POC zarr-v3 store: sigma0 stacks, EMS labels, decoded I/Q, SMHI tables.

Layout (data/zarr/poc.zarr):
  sigma0/track_<rel>/     xarray (time,y,x) float32, chunks (1,512,512), blosc-zstd3+byteshuffle
  labels/                 (y,x) uint8 masks per EMS product + union; GeoJSON in attrs
  raw/<datatake>_<pol>/chunk_<id>/  iq (line,sample) complex64, zstd3 (measured best);
                          per-line packet-header columns; chunk attrs = swath, times
  tables/smhi             precipitation + discharge time series

Chunking rationale (see docs/notes.md): chips of (1 time, 512, 512) for rasters =
ML read pattern; raw I/Q chunked (line-block 256, full range line) because range
lines are the atomic unit a raw-domain model consumes.
Codec choices are the measured winners from scripts/bench_codecs.py.
"""

import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import rioxarray
import zarr
from zarr.codecs import BloscCodec, ZstdCodec

from rawsar import flood, l0

ROOT = Path(__file__).resolve().parents[1]
STORE = ROOT / "data/zarr/poc.zarr"

RASTER_COMP = BloscCodec(cname="zstd", clevel=3, shuffle="shuffle")
IQ_COMP = ZstdCodec(level=3)


def add_sigma0_stacks(root: zarr.Group) -> xr.DataArray:
    template = None
    by_track: dict[int, list] = {}
    for tif in sorted((ROOT / "data/gtc").glob("*_VV_sigma0.tif")):
        sat, rel, date, *_ = tif.stem.split("_")
        da = rioxarray.open_rasterio(tif).squeeze("band", drop=True)
        template = template if template is not None else da
        by_track.setdefault(int(rel), []).append((pd.Timestamp(date), sat, da))
    for rel, items in sorted(by_track.items()):
        items.sort()
        stack = xr.concat([d for _, _, d in items], dim="time")
        stack = stack.assign_coords(time=[t for t, _, _ in items])
        ds = stack.to_dataset(name="sigma0_vv")
        ds["satellite"] = ("time", [s for _, s, _ in items])
        ds.attrs.update(crs="EPSG:32633", description=f"sigma0 VV GTC, rel orbit {rel}",
                        source="sarsen GTC on GLO-30 20m; scripts/gtc_all.py")
        ds.to_zarr(STORE, group=f"sigma0/track_{rel:03d}", mode="w",
                   encoding={"sigma0_vv": {"chunks": (1, 512, 512), "compressors": [RASTER_COMP]}})
        print(f"sigma0/track_{rel:03d}: {dict(ds.sizes)}")
    return template


def add_labels(root: zarr.Group, template) -> None:
    ems_dir = ROOT / "data/ems"
    grp = root.create_group("labels", overwrite=True)
    grp.attrs["crs"] = "EPSG:32633"
    grp.attrs["grid"] = "same 20m UTM33 grid as sigma0 stacks"
    union = None
    import geopandas as gpd
    for d in sorted(ems_dir.iterdir()):
        for layer in ("observedEventA", "maximumFloodExtentA", "areaOfInterestA"):
            js = list(d.glob(f"*_{layer}_v1.json"))
            if not js:
                continue
            gdf = gpd.read_file(js[0])
            mask = flood.rasterize_labels(gdf, template).values.astype("uint8")
            name = f"{d.name.replace('EMSR841_AOI01_', '').replace('_v1', '')}_{layer[:-1]}"
            arr = grp.create_array(name, shape=mask.shape, dtype="uint8",
                                   chunks=(512, 512), compressors=[RASTER_COMP], overwrite=True)
            arr[:] = mask
            arr.attrs["n_polygons"] = int(len(gdf))
            arr.attrs["source_file"] = js[0].name
            if layer != "areaOfInterestA":
                union = mask if union is None else (union | mask)
    u = grp.create_array("ems_flood_union", shape=union.shape, dtype="uint8",
                         chunks=(512, 512), compressors=[RASTER_COMP], overwrite=True)
    u[:] = union
    print(f"labels: {len(list(grp.array_keys()))} rasters (union {union.sum() * 4e-4:.2f} km²)")


# The two label-anchor datatakes: (RAW safe glob, GRD-for-GCPs safe glob, description)
DATATAKES = [
    ("S1C_IW_RAW*00809C*.SAFE", "S1C_IW_GRDH*00809C_1E3D.SAFE",
     "S1C rel-22 DESC 2025-09-09 (MONIT02 anchor track)"),
    ("S1A_IW_RAW*0796C3*.SAFE", "S1A_IW_GRDH*0796C3_FC8C.SAFE",
     "S1A rel-131 ASC 2025-09-10 (FEP label-source pass, zero offset)"),
]


def add_raw(root: zarr.Group) -> None:
    """AOI-relevant echo chunks for each anchor datatake, all three IW swaths."""
    for raw_glob, grd_glob, desc in DATATAKES:
        hits = glob.glob(str(ROOT / "data/extracted" / raw_glob))
        if not hits:
            print(f"add_raw: {raw_glob} not extracted, skipping ({desc})")
            continue
        _add_raw_one(root, Path(hits[0]), grd_glob, desc)


def _add_raw_one(root: zarr.Group, safe: Path, grd_glob: str, desc: str) -> None:
    f = l0.open_l0(safe, "vv")
    track = l0.chunk_subsatellite_track(f)

    # AOI azimuth-time window from the same-datatake GRD's GCP grid (the honest way:
    # sub-satellite lat/lon alone is ~1° lat off because the swath looks ~450 km sideways)
    grd_safe = glob.glob(str(ROOT / "data/extracted" / grd_glob))[0]
    gcp = xr.open_dataset(grd_safe, engine="sentinel-1", group="IW/VV/gcp")
    in_aoi = ((gcp.latitude >= 63.20) & (gcp.latitude <= 63.61)
              & (gcp.longitude >= 17.6) & (gcp.longitude <= 18.9))
    az = gcp.azimuth_time.where(in_aoi.any("slant_range_time")).dropna("azimuth_time")
    gps_epoch = pd.Timestamp("1980-01-06")
    leap = pd.Timedelta(seconds=18)  # GPS-UTC offset in 2025
    t0 = (pd.Timestamp(az.values[0]) - gps_epoch + leap).total_seconds() - 2.0
    t1 = (pd.Timestamp(az.values[-1]) - gps_epoch + leap).total_seconds() + 2.0
    sel = track[(track.signal == "Echo") & (track.t_mid >= t0) & (track.t_mid <= t1)]
    print(f"AOI azimuth window {t1-t0:.1f}s -> {len(sel)} echo chunks")

    datatake = safe.name.replace(".SAFE", "").split("_")[-2]
    grp = root.create_group(f"raw/{datatake}_vv", overwrite=True)
    grp.attrs.update(
        product=safe.name, polarisation="VV", description=desc,
        selection="echo chunks overlapping the AOI azimuth-time window derived from "
                  "the same-datatake GRD GCPs (+-2s margin)",
        aoi_azimuth_window_gps_s=[t0, t1],
    )
    keep_cols = ["Space Packet Count", "PRI Count", "Coarse Time", "Fine Time", "PRI",
                 "SWST", "SWL", "Rank", "Number of Quads"]
    for cid, row in sel.iterrows():
        iq = f.get_acquisition_chunk_data(int(cid), try_load_from_file=False)
        md = f.get_acquisition_chunk_metadata(int(cid))
        g = grp.create_group(f"chunk_{int(cid):03d}", overwrite=True)
        a = g.create_array("iq", shape=iq.shape, dtype="complex64",
                           chunks=(256, iq.shape[1]), compressors=[IQ_COMP], overwrite=True)
        a[:] = iq
        for col in keep_cols:
            v = md[col].to_numpy(float)
            c = g.create_array(col.lower().replace(" ", "_"), shape=v.shape, dtype="float64",
                               compressors=[IQ_COMP], overwrite=True)
            c[:] = v
        g.attrs.update(swath=str(row.swath), n_lines=int(iq.shape[0]),
                       n_samples=int(iq.shape[1]), subsat_lat=float(row.subsat_lat),
                       subsat_lon=float(row.subsat_lon))
        print(f"raw chunk_{int(cid):03d}: swath {row.swath} {iq.shape} "
              f"subsat_lat={row.subsat_lat:.2f}")
        del iq


def add_label_windows(root: zarr.Group) -> None:
    """Strategy (b): EMS polygons as raw-geometry (azimuth, slant) windows, per anchor pass."""
    import geopandas as gpd

    for _raw_glob, grd_glob, desc in DATATAKES:
        hits = glob.glob(str(ROOT / "data/extracted" / grd_glob))
        if not hits:
            print(f"add_label_windows: {grd_glob} not extracted, skipping")
            continue
        datatake = Path(hits[0]).name.replace(".SAFE", "").split("_")[-2]
        gcp = xr.open_dataset(hits[0], engine="sentinel-1", group="IW/VV/gcp")
        grp = root.require_group("labels").create_group(f"raw_windows_{datatake}", overwrite=True)
        grp.attrs.update(
            pass_product=desc,
            method="docs/labels.md strategy (b): GCP-grid projection + physics dilation",
            azimuth_dilation_s=0.4, slant_dilation_s=52.4e-6,
            columns=["azimuth_t0_utc_ns", "azimuth_t1_utc_ns", "slant_t0_s", "slant_t1_s"],
        )
        for d, layer in [("EMSR841_AOI01_DEL_MONIT02_v1", "maximumFloodExtentA"),
                         ("EMSR841_AOI01_FEP_PRODUCT_v1", "observedEventA")]:
            js = list((ROOT / "data/ems" / d).glob(f"*_{layer}_v1.json"))[0]
            win = l0.l0_label_windows(gcp, gpd.read_file(js).geometry)
            arr = np.column_stack([win.azimuth_t0.astype("int64"), win.azimuth_t1.astype("int64"),
                                   win.slant_t0_s, win.slant_t1_s])
            name = f"{d.replace('EMSR841_AOI01_', '').replace('_v1', '')}_{layer[:-1]}"
            a = grp.create_array(name, shape=arr.shape, dtype="float64",
                                 compressors=[IQ_COMP], overwrite=True)
            a[:] = arr
            a.attrs["n_polygons_projected"] = int(len(win))
            a.attrs["source_file"] = js.name
            print(f"raw_windows_{datatake}/{name}: {len(win)} windows")


def add_tables(root: zarr.Group) -> None:
    sys.path.insert(0, str(ROOT / "scripts"))
    from event_timeline import read_smhi_csv

    frames = {}
    for f in sorted((ROOT / "data/smhi").glob("precip_daily_*.csv")):
        frames[f"precip_{f.stem.split('_')[-1]}_mm"] = read_smhi_csv(f).set_index("date")["mm"]
    dis = ROOT / "data/smhi/discharge_daily_1583_vastersel.csv"
    rows = [ln.split(";") for ln in dis.read_text().splitlines()
            if ln[:4] in ("2025", "2026") and ";" in ln]
    q = pd.Series({pd.Timestamp(p[0]): float(p[1]) for p in rows
                   if len(p) > 1 and p[1].replace(".", "", 1).isdigit()})
    frames["discharge_vastersel_m3s"] = q
    df = pd.DataFrame(frames).loc["2025-08-01":"2025-10-31"]
    ds = xr.Dataset({k: ("time", v.to_numpy()) for k, v in df.items()},
                    coords={"time": df.index.to_numpy()})
    ds.attrs["source"] = "SMHI open data (metobs param 5, hydroobs param 1), CC BY 4.0"
    ds.to_zarr(STORE, group="tables/smhi", mode="w")
    print(f"tables/smhi: {dict(ds.sizes)}, {list(ds.data_vars)}")


def main() -> None:
    STORE.parent.mkdir(parents=True, exist_ok=True)
    root = zarr.open_group(STORE, mode="a")
    root.attrs.update(
        title="Örnsköldsvik flood POC (EMSR841) — raw+processed Sentinel-1",
        created="2026-07-16",
        event="Västernorrland flood, Sept 2025",
        conventions="ad-hoc v0; align with EOPF/GeoZarr naming before publishing",
    )
    template = add_sigma0_stacks(root)
    add_labels(root, template)
    add_raw(root)
    add_label_windows(root)
    add_tables(root)
    zarr.consolidate_metadata(STORE)  # xarray to_zarr leaves stale root consolidation
    # size report
    total = sum(p.stat().st_size for p in STORE.rglob("*") if p.is_file())
    print(f"\nstore total: {total/1e9:.2f} GB at {STORE}")


if __name__ == "__main__":
    main()
