#!/usr/bin/env python
"""Print the Phase-1 download manifest (sizes from live OData) and, with --yes, transfer.

Usage:
    python scripts/download_manifest.py            # dry run: table + total GB
    python scripts/download_manifest.py --yes      # actually download (needs .env creds)
    python scripts/download_manifest.py --yes --skip-ems --skip-orbits
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rawsar import cdse, ems  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def sensing_iso(name: str) -> str:
    m = re.search(r"_(\d{8}T\d{6})_", name)
    assert m, name
    t = m.group(1)
    return f"{t[:4]}-{t[4:6]}-{t[6:8]}T{t[9:11]}:{t[11:13]}:{t[13:15]}.000Z"


def datatake_sensings(cfg: dict, s1_dir: Path) -> set[tuple[str, str]]:
    """(platform, sensing_iso) for every manifest product AND every product already on disk."""
    out = set()
    names = [m["name"] for m in cfg["manifest"]]
    names += [p.name for p in s1_dir.glob("S1*_IW_*.zip")]
    for n in names:
        out.add((n[:3], sensing_iso(n)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--yes", action="store_true", help="actually download")
    ap.add_argument("--skip-ems", action="store_true")
    ap.add_argument("--skip-orbits", action="store_true")
    ap.add_argument("--config", default=ROOT / "configs/aoi_ornskoldsvik.yaml", type=Path)
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    s1_dir = ROOT / cfg["download"]["target_dir"]
    ems_dir = ROOT / cfg["download"]["ems_dir"]

    print("Resolving manifest against CDSE OData ...")
    products = []
    total = 0.0
    for item in cfg["manifest"]:
        p = cdse.find_product(item["name"])
        on_disk = (s1_dir / (p.name.removesuffix(".SAFE") + ".zip")).exists()
        products.append((item, p, on_disk))
        if not on_disk:
            total += p.gb
        flag = "on disk" if on_disk else f"{p.gb:6.2f} GB"
        print(f"  [{item['track']:>3} {item['role']:<30}] {p.name:<75} {flag}")

    orbits = []
    if not args.skip_orbits:
        print("Resolving AUX_POEORB ...")
        for platform, iso in sorted(datatake_sensings(cfg, s1_dir)):
            try:
                o = cdse.find_poeorb(platform, iso)
            except LookupError as e:
                print(f"  WARNING: {e}")
                continue
            if not any(o.name == prev.name for prev in orbits):
                orbits.append(o)
        for o in orbits:
            print(f"  {o.name}  {o.size/1e6:.1f} MB")
        total += sum(o.gb for o in orbits)

    cap = cfg["download"]["hard_cap_gb"]
    print(f"\nTOTAL to transfer: {total:.1f} GB (hard cap {cap} GB)")
    if total > cap:
        sys.exit(f"ABORT: manifest exceeds hard cap {cap} GB")
    if not args.yes:
        print("Dry run. Re-run with --yes to download.")
        return

    if not args.skip_ems:
        print("\nFetching EMS zips (public) ...")
        act = ems.activation(cfg["aoi"]["ems_activation"])
        wanted = cfg.get("ems_downloads", [])
        for d in ems.product_downloads(act):
            if d["download"] and any(w.lower() in d["download"].lower() for w in wanted):
                path = ems.fetch(d["download"], ems_dir)
                print(f"  {path.name} ({path.stat().st_size/1e6:.1f} MB)")
            elif not d["download"] and d["not_produced_reason"]:
                print(f"  note: {d['aoi']} {d['type']} not produced ({d['not_produced_reason']})")

    try:
        tokens = cdse.TokenManager()
    except RuntimeError as e:
        sys.exit(f"\nSTOP before CDSE transfer: {e}\n"
                 "EMS zips above (if any) were fetched; S1/orbit downloads need credentials.")

    if orbits:
        orb_dir = ROOT / "data/aux_poeorb"
        print(f"\nDownloading {len(orbits)} orbit files -> {orb_dir}")
        for o in orbits:
            path = cdse.download(o, orb_dir, tokens)
            print(f"  ok {path.name}")

    todo = [(i, p) for i, p, on_disk in products if not on_disk]
    print(f"\nDownloading {len(todo)} S1 products -> {s1_dir}")
    for _item, p in todo:
        with tqdm(total=p.size, unit="B", unit_scale=True, desc=p.name[:52]) as bar:
            path = cdse.download(p, s1_dir, tokens, progress=bar.update)
        print(f"  ok {path.name}  md5 verified={bool(p.md5)}")

    print("\nDone. Re-running this script should now be a no-op for everything above.")


if __name__ == "__main__":
    main()
