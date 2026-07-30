"""Copernicus Data Space Ecosystem access: OData search + authenticated download.

Search is unauthenticated; download needs a free CDSE account with credentials in
`.env` (CDSE_USERNAME / CDSE_PASSWORD). Tokens come from Keycloak and expire after
~10 min, so long transfers refresh them mid-download.
"""

from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass
from pathlib import Path

import requests
from dotenv import load_dotenv

CATALOGUE = "https://catalogue.dataspace.copernicus.eu/odata/v1"
DOWNLOAD = "https://download.dataspace.copernicus.eu/odata/v1"
TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
CHUNK = 8 * 1024 * 1024


@dataclass
class Product:
    id: str
    name: str
    size: int
    md5: str | None
    s3_path: str | None

    @property
    def gb(self) -> float:
        return self.size / 1e9


def _get(url: str, **kwargs) -> requests.Response:
    for attempt in range(5):
        try:
            r = requests.get(url, timeout=kwargs.pop("timeout", 60), **kwargs)
            if r.status_code < 500:
                return r
        except requests.RequestException:
            if attempt == 4:
                raise
        time.sleep(2**attempt)
    r.raise_for_status()
    return r


def find_product(name: str) -> Product:
    """Look up a single product by exact SAFE name."""
    r = _get(CATALOGUE + "/Products", params={"$filter": f"Name eq '{name}'"})
    r.raise_for_status()
    vals = r.json()["value"]
    if len(vals) != 1:
        raise LookupError(f"expected exactly 1 catalogue hit for {name}, got {len(vals)}")
    v = vals[0]
    md5 = next(
        (c["Value"] for c in v.get("Checksum", []) if c.get("Algorithm") == "MD5" and c.get("Value")),
        None,
    )
    return Product(id=v["Id"], name=v["Name"], size=v["ContentLength"], md5=md5, s3_path=v.get("S3Path"))


def find_poeorb(platform: str, sensing_iso: str) -> Product:
    """Find the precise orbit (AUX_POEORB) covering a sensing instant for S1A/S1B/S1C."""
    flt = (
        f"startswith(Name,'{platform}_OPER_AUX_POEORB') "
        f"and ContentDate/Start lt {sensing_iso} and ContentDate/End gt {sensing_iso}"
    )
    r = _get(CATALOGUE + "/Products", params={"$filter": flt, "$orderby": "PublicationDate desc"})
    r.raise_for_status()
    vals = r.json()["value"]
    if not vals:
        raise LookupError(f"no POEORB for {platform} @ {sensing_iso}")
    v = vals[0]  # newest publication wins
    md5 = next(
        (c["Value"] for c in v.get("Checksum", []) if c.get("Algorithm") == "MD5" and c.get("Value")),
        None,
    )
    return Product(id=v["Id"], name=v["Name"], size=v["ContentLength"], md5=md5, s3_path=v.get("S3Path"))


class TokenManager:
    """Keycloak password-grant token with refresh; re-authenticates when refresh expires."""

    def __init__(self) -> None:
        load_dotenv()
        self.user = os.environ.get("CDSE_USERNAME")
        self.password = os.environ.get("CDSE_PASSWORD")
        if not self.user or not self.password:
            raise RuntimeError("CDSE_USERNAME / CDSE_PASSWORD not set (create .env from .env.example)")
        self._access: str | None = None
        self._refresh: str | None = None
        self._expires_at = 0.0

    def _grant(self, data: dict) -> None:
        r = requests.post(TOKEN_URL, data={"client_id": "cdse-public", **data}, timeout=30)
        r.raise_for_status()
        tok = r.json()
        self._access = tok["access_token"]
        self._refresh = tok.get("refresh_token")
        self._expires_at = time.time() + tok.get("expires_in", 600) - 60

    def token(self) -> str:
        if self._access and time.time() < self._expires_at:
            return self._access
        if self._refresh:
            try:
                self._grant({"grant_type": "refresh_token", "refresh_token": self._refresh})
                return self._access  # type: ignore[return-value]
            except requests.HTTPError:
                pass
        self._grant({"grant_type": "password", "username": self.user, "password": self.password})
        return self._access  # type: ignore[return-value]


def _md5_of(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while chunk := f.read(CHUNK):
            h.update(chunk)
    return h.hexdigest()


def download(product: Product, dest_dir: Path, tokens: TokenManager, progress=None) -> Path:
    """Download product zip with resume + MD5 verify. No-op if already present and valid."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    stem = product.name.removesuffix(".SAFE")
    final = dest_dir / (stem if stem.endswith(".EOF") else stem + ".zip")
    part = final.with_suffix(final.suffix + ".part")

    if final.exists():
        if final.stat().st_size == product.size and (not product.md5 or _md5_of(final) == product.md5.lower()):
            return final
        final.unlink()  # size/checksum mismatch: redownload

    url = f"{DOWNLOAD}/Products({product.id})/$value"
    offset = part.stat().st_size if part.exists() else 0
    stalls = 0
    while True:
        start_offset = offset
        headers = {"Authorization": f"Bearer {tokens.token()}"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        with requests.get(url, headers=headers, stream=True, timeout=120, allow_redirects=True) as r:
            if offset and r.status_code == 200:
                offset = 0  # server ignored Range: restart
            elif offset and r.status_code == 416:
                break  # already complete
            r.raise_for_status()
            mode = "ab" if offset else "wb"
            with open(part, mode) as f:
                try:
                    for chunk in r.iter_content(CHUNK):
                        f.write(chunk)
                        offset += len(chunk)
                        if progress:
                            progress(len(chunk))
                except (requests.RequestException, ConnectionError):
                    pass  # fall through to stall check; Range resume on next loop
        if offset >= product.size:
            break
        stalls = 0 if offset > start_offset else stalls + 1
        if stalls >= 5:
            raise OSError(f"{product.name}: no progress after {stalls} attempts at byte {offset}")
        time.sleep(2**stalls)

    if part.stat().st_size != product.size:
        raise OSError(f"{product.name}: got {part.stat().st_size} bytes, expected {product.size}")
    if product.md5 and _md5_of(part) != product.md5.lower():
        part.unlink()
        raise OSError(f"{product.name}: MD5 mismatch after download (partial removed; rerun)")
    part.rename(final)
    return final
