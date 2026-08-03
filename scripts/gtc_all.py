#!/usr/bin/env python
"""Geocode every on-disk GRD (non-COG) to sigma0 GTC over the AOI. One product per call."""
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, "src")
from rawsar import grd

prods = [p for p in grd.scan(Path("data/copernicus_data/S1")) if p.kind == "GRDH" and not p.is_cog]
if len(sys.argv) > 1:  # child mode: geocode one product by index
    import yaml

    from rawsar.grd import gtc_sigma0
    p = prods[int(sys.argv[1])]
    bbox = tuple(yaml.safe_load(open("configs/aoi_ornskoldsvik.yaml"))["aoi"]["bbox"])
    out = Path(f"data/gtc/{p.platform}_{p.relative_orbit:03d}_{p.start:%Y%m%d}_VV_sigma0.tif")
    gtc_sigma0(p, "VV", Path("data/dem/aoi_glo30_utm33_20m.tif"), out, crop_bbox=bbox)
    print("ok", out)
else:
    for i, p in enumerate(prods):
        out = Path(f"data/gtc/{p.platform}_{p.relative_orbit:03d}_{p.start:%Y%m%d}_VV_sigma0.tif")
        if out.exists() and out.stat().st_size > 0:
            print("cached", out)
            continue
        r = subprocess.run([sys.executable, __file__, str(i)])
        if r.returncode != 0:
            print(f"FAILED idx={i} {p.safe_name}", file=sys.stderr)
