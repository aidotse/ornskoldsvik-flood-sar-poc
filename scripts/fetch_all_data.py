#!/usr/bin/env python
"""One command to put every input the three notebooks need on disk.

    python scripts/fetch_all_data.py          # dry run: what's missing, how many GB
    python scripts/fetch_all_data.py --yes    # fetch it

Only the Sentinel-1 products need credentials (free CDSE account -> .env); the DEM,
the EMS labels and the SMHI series are all public. Every stage is a no-op when its
output is already on disk, so re-running is cheap and safe.

Stages, in dependency order:
  1. GLO-30 DEM tiles      public AWS S3        -> data/dem/
  2. SMHI precip/discharge public REST API      -> data/smhi/
  3. EMS label zips        public HTTPS         -> data/copernicus_data/ + data/ems/
  4. Sentinel-1 products   CDSE (needs .env)    -> data/copernicus_data/S1/
  5. SAFE extraction       local unzip          -> data/extracted/, data/copernicus_data/S1/

Stage 5 exists because rasterio cannot read measurement files from inside a zip:
notebook 03 and the zarr builder both glob for extracted *.SAFE directories. GRD
SAFEs are extracted on demand by grd.gtc_sigma0(), so only the RAW passes and the
one SLC that notebook 03 compares against are handled here.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import zipfile
from pathlib import Path

import requests
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rawsar import cdse, ems  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
UA = {"User-Agent": "rawsar-poc/0.1 (research; contact via repo)"}

SMHI_PRECIP = ("https://opendata-download-metobs.smhi.se/api/version/1.0"
               "/parameter/5/station/{station}/period/corrected-archive/data.csv")
SMHI_DISCHARGE = ("https://opendata-download-hydroobs.smhi.se/api/version/1.0"
                  "/parameter/1/station/{station}/period/corrected-archive/data.csv")

# Products that must exist as extracted .SAFE directories, and where they go.
# RAW: notebook 03's two decode subjects + the zarr store's two raw anchors.
# SLC: notebook 03 sanity-checks its own range compression against this one.
EXTRACT_RAW = ["S1C_IW_RAW__0SDV_*00809C*.zip", "S1A_IW_RAW__0SDV_*0796C3*.zip"]
EXTRACT_IN_PLACE = ["S1C_IW_SLC__1SDV_*00809C_7A6D.zip"]


def _get(url: str, dest: Path, desc: str) -> None:
    """Stream a public URL to dest (atomic via .part), with a progress bar."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, headers=UA, stream=True, timeout=120) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0))
        with open(tmp, "wb") as fh, tqdm(total=total or None, unit="B", unit_scale=True,
                                          desc=desc[:48]) as bar:
            for chunk in r.iter_content(1 << 20):
                fh.write(chunk)
                bar.update(len(chunk))
    tmp.rename(dest)


def _extract_atomic(zp: Path, target: Path, member_root: str | None = None) -> None:
    """Unzip zp so that `target` only appears once the extraction has completed.

    Extracts into a hidden sibling `.<name>.partial` directory and renames it into place,
    so an interrupted unzip (Ctrl-C, full disk, bad zip) is redone on the next run instead
    of being mistaken for a finished one. member_root: the zip's top-level folder to move
    into place (SAFE zips); None for flat zips whose contents become `target` itself.
    """
    tmp = target.with_name(f".{target.name}.partial")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    with zipfile.ZipFile(zp) as z:
        z.extractall(tmp)
    if member_root is None:
        tmp.rename(target)
    else:
        (tmp / member_root).rename(target)
        shutil.rmtree(tmp)


def fetch_dem(cfg: dict, do: bool) -> list[str]:
    """GLO-30 tiles from public S3. grd.prepare_dem() merges them (scripts/gtc_all.py does it first)."""
    out_dir = ROOT / "data/dem"
    missing = [t for t in cfg["dem"]["tiles"] if not (out_dir / f"{t}.tif").exists()]
    for tile in missing if do else []:
        _get(cfg["dem"]["url_template"].format(tile=tile), out_dir / f"{tile}.tif", tile)
    return [f"DEM tile {t}" for t in missing]


def fetch_smhi(cfg: dict, do: bool) -> list[str]:
    """Daily precipitation + discharge CSVs. Saved verbatim; parsed by header later."""
    out_dir = ROOT / "data/smhi"
    todo = []
    for kind, url, prefix in (("precip", SMHI_PRECIP, "precip_daily"),
                               ("discharge", SMHI_DISCHARGE, "discharge_daily")):
        for st in cfg["smhi"][kind]:
            dest = out_dir / f"{prefix}_{st['station']}_{st['name']}.csv"
            if not dest.exists():
                todo.append((url.format(station=st["station"]), dest))
    for url, dest in todo if do else []:
        _get(url, dest, dest.name)
    return [f"SMHI {d.name}" for _u, d in todo]


def fetch_ems(cfg: dict, do: bool) -> list[str]:
    """EMS label zips (public) + extract each into data/ems/<product>/."""
    ems_dir = ROOT / cfg["download"]["ems_dir"]
    wanted = cfg.get("ems_downloads", [])
    missing = [w for w in wanted if not (ROOT / "data/ems" / w).is_dir()]
    if not do:
        return [f"EMS {w}" for w in missing]

    if missing:
        act = ems.activation(cfg["aoi"]["ems_activation"])
        for d in ems.product_downloads(act):
            if d["download"] and any(w.lower() in d["download"].lower() for w in missing):
                zip_path = ems.fetch(d["download"], ems_dir)
                print(f"  fetched {zip_path.name} ({zip_path.stat().st_size / 1e6:.1f} MB)")
            elif not d["download"] and d["not_produced_reason"]:
                print(f"  note: {d['aoi']} {d['type']} not produced ({d['not_produced_reason']})")
    # extract every EMS zip on disk whose target folder is absent (zips are flat)
    for zp in sorted(ems_dir.glob("EMSR841_*.zip")):
        target = ROOT / "data/ems" / zp.stem
        if target.is_dir():
            continue
        _extract_atomic(zp, target)
        print(f"  extracted {zp.stem}/")
    return [f"EMS {w}" for w in missing]


def fetch_s1(cfg: dict, do: bool) -> list[str]:
    """Resolve the manifest against CDSE OData; download what's missing (needs .env)."""
    s1_dir = ROOT / cfg["download"]["target_dir"]
    todo, total = [], 0.0
    for item in cfg["manifest"]:
        p = cdse.find_product(item["name"])
        if not (s1_dir / (p.name.removesuffix(".SAFE") + ".zip")).exists():
            todo.append(p)
            total += p.gb
    cap = cfg["download"]["hard_cap_gb"]
    if total > cap:
        sys.exit(f"ABORT: {total:.1f} GB exceeds hard cap {cap} GB")
    if not do or not todo:
        return [f"S1 {p.name[:52]} ({p.gb:.2f} GB)" for p in todo]

    try:
        tokens = cdse.TokenManager()
    except RuntimeError as e:
        sys.exit(f"\nSTOP: {e}\nEverything public is fetched; S1 products need CDSE "
                 "credentials in .env (free account at dataspace.copernicus.eu).")
    for p in todo:
        with tqdm(total=p.size, unit="B", unit_scale=True, desc=p.name[:48]) as bar:
            cdse.download(p, s1_dir, tokens, progress=bar.update)
    return [f"S1 {p.name[:52]} ({p.gb:.2f} GB)" for p in todo]


def extract_safes(cfg: dict, do: bool) -> list[str]:
    """Unzip the RAW passes and the one SLC that must exist as real directories."""
    s1_dir = ROOT / cfg["download"]["target_dir"]
    jobs = ([(pat, ROOT / "data/extracted") for pat in EXTRACT_RAW]
            + [(pat, s1_dir) for pat in EXTRACT_IN_PLACE])
    todo = []
    for pattern, dest_dir in jobs:
        for zp in sorted(s1_dir.glob(pattern)):
            if not (dest_dir / f"{zp.stem}.SAFE").exists():
                todo.append((zp, dest_dir))
    for zp, dest_dir in todo if do else []:
        _extract_atomic(zp, dest_dir / f"{zp.stem}.SAFE", member_root=f"{zp.stem}.SAFE")
        print(f"  extracted {zp.stem}.SAFE -> {dest_dir.relative_to(ROOT)}/")
    return [f"extract {zp.stem}.SAFE" for zp, _d in todo]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--yes", action="store_true", help="actually fetch (default: dry run)")
    ap.add_argument("--skip-s1", action="store_true", help="public sources only, no CDSE")
    ap.add_argument("--config", default=ROOT / "configs/aoi_ornskoldsvik.yaml", type=Path)
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())

    stages = [("DEM tiles (public S3)", fetch_dem),
              ("SMHI series (public API)", fetch_smhi),
              ("EMS labels (public HTTPS)", fetch_ems)]
    if not args.skip_s1:
        stages.append(("Sentinel-1 products (CDSE, needs .env)", fetch_s1))
    stages.append(("SAFE extraction (local)", extract_safes))

    pending = 0
    for title, fn in stages:
        print(f"\n=== {title} ===")
        missing = fn(cfg, args.yes)
        pending += len(missing)
        if not missing:
            print("  already on disk")
        elif not args.yes:
            for m in missing:
                print(f"  MISSING  {m}")

    if not args.yes:
        print(f"\n{pending} item(s) missing. Re-run with --yes to fetch.")
        print("Note: SAFE extraction counts only what can be seen now — more will be "
              "queued once the S1 zips land.")
    else:
        print("\nDone. Every notebook input is on disk; re-running this is a no-op.")
        print("Next: python scripts/gtc_all.py, then the build_nb0*.py scripts.")


if __name__ == "__main__":
    main()
