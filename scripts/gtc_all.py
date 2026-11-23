#!/usr/bin/env python
"""Geocode every on-disk GRD (non-COG) to sigma0 GTC over the AOI. One product per call.

The parent first builds the AOI DEM from the configured GLO-30 tiles (grd.prepare_dem,
cached), then runs one child process per product. It exits non-zero if any product
failed, so `gtc_all.py && build_zarr_store.py` stops instead of building from nothing.
"""
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rawsar import grd  # noqa: E402

CFG = yaml.safe_load((ROOT / "configs/aoi_ornskoldsvik.yaml").read_text())
BBOX = tuple(CFG["aoi"]["bbox"])
DEM = ROOT / "data/dem/aoi_glo30_utm33_20m.tif"
S1_DIR = ROOT / CFG["download"]["target_dir"]


def gtc_path(p: grd.S1Zip) -> Path:
    return ROOT / f"data/gtc/{p.platform}_{p.relative_orbit:03d}_{p.start:%Y%m%d}_VV_sigma0.tif"


def grd_products() -> list[grd.S1Zip]:
    return [p for p in grd.scan(S1_DIR) if p.kind == "GRDH" and not p.is_cog]


if len(sys.argv) > 1:  # child mode: geocode the one product whose zip path is given
    zip_path = Path(sys.argv[1])
    p = next(q for q in grd_products() if q.path == zip_path)
    out = gtc_path(p)
    grd.gtc_sigma0(p, "VV", DEM, out, crop_bbox=BBOX, extract_dir=ROOT / "data/extracted")
    print("ok", out)
else:
    grd.prepare_dem([ROOT / "data/dem" / f"{t}.tif" for t in CFG["dem"]["tiles"]], BBOX, DEM)
    failed = []
    for p in grd_products():
        out = gtc_path(p)
        if out.exists() and out.stat().st_size > 0:
            print("cached", out)
            continue
        r = subprocess.run([sys.executable, __file__, str(p.path)])
        if r.returncode != 0:
            print(f"FAILED {p.safe_name}", file=sys.stderr)
            failed.append(p.safe_name)
    if failed:
        sys.exit(f"{len(failed)} product(s) failed to geocode: {', '.join(failed)}")
