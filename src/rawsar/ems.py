"""Copernicus EMS rapid-mapping access for EMSR841 (public, no auth)."""

from __future__ import annotations

from pathlib import Path

import requests

API = "https://rapidmapping.emergency.copernicus.eu/backend/dashboard-api/public-activations/"
BASE = "https://rapidmapping.emergency.copernicus.eu"


def activation(code: str = "EMSR841") -> dict:
    r = requests.get(API, params={"code": code}, timeout=60)
    r.raise_for_status()
    data = r.json()
    results = data.get("results", data if isinstance(data, list) else [data])
    for act in results:
        if act.get("code") == code:
            return act
    raise LookupError(f"activation {code} not found")


def product_downloads(act: dict) -> list[dict]:
    """Flatten the activation JSON into one row per product.

    Note: a product can exist in the JSON but have an empty downloadPath — e.g.
    EMSR841 AOI02 DEL: "Not produced because of remote sensing limitations".
    """
    out = []
    for aoi in act.get("aois", []):
        for prod in aoi.get("products", []):
            imgs = [
                {"sensor": i.get("sensorName"), "acquired": i.get("acquisitionTime")}
                for i in prod.get("images", [])
            ]
            out.append(
                {
                    "aoi": f"AOI{prod.get('aoiNumber', 0):02d}",
                    "aoi_name": aoi.get("name"),
                    "type": prod.get("type"),
                    "monitoring_number": prod.get("monitoringNumber"),
                    "download": prod.get("downloadPath") or None,
                    "status": (prod.get("version") or {}).get("statusCode"),
                    "not_produced_reason": (prod.get("version") or {}).get("reason") or None,
                    "source_imagery": imgs,
                }
            )
    return out


def fetch(url: str, dest_dir: Path) -> Path:
    if url.startswith("/"):
        url = BASE + url
    dest = dest_dir / url.rsplit("/", 1)[-1]
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest_dir.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        tmp = dest.with_suffix(dest.suffix + ".part")
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
        tmp.rename(dest)
    return dest
