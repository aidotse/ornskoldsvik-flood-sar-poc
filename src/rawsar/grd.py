"""GRD handling: product discovery, σ⁰ calibration, sarsen terrain correction.

All products are read directly from their zips via xarray-sentinel
(``zip://<name>.SAFE::<path>.zip``) — nothing is extracted to disk.
"""

from __future__ import annotations

import re
import shutil
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import rioxarray  # noqa: F401  (registers .rio accessor)
import xarray as xr
from rasterio.enums import Resampling
from sarsen import apps
from sarsen.sentinel1 import Sentinel1SarProduct

# Anchored at the repository, not the process cwd: notebooks run from notebooks/ and
# must extract to the same data/extracted that build_zarr_store/label_attach_errorbars read.
DEFAULT_EXTRACT_DIR = Path(__file__).resolve().parents[2] / "data" / "extracted"

NAME_RE = re.compile(
    r"(?P<platform>S1[ABC])_IW_(?P<kind>RAW__0|SLC__1|GRDH_1|OCN__2)S(?P<pols>\w{2})_"
    r"(?P<start>\d{8}T\d{6})_(?P<stop>\d{8}T\d{6})_(?P<orbit>\d{6})_(?P<datatake>[0-9A-F]{6})_"
    r"(?P<crc>[0-9A-F]{4})"
)


@dataclass
class S1Zip:
    path: Path
    platform: str
    kind: str  # RAW / SLC / GRDH / OCN
    start: datetime
    stop: datetime
    absolute_orbit: int
    datatake: str
    relative_orbit: int
    safe_name: str
    is_cog: bool = False

    @property
    def zip_spec(self) -> str:
        """fsspec chain string understood by xarray-sentinel."""
        return f"zip://{self.safe_name}::{self.path}"


def _relative_orbit(absolute_orbit: int, platform: str) -> int:
    """S1 relative orbit from absolute orbit (constants from S1 mission docs)."""
    if platform == "S1A":
        return (absolute_orbit - 73) % 175 + 1
    if platform == "S1B":
        return (absolute_orbit - 27) % 175 + 1
    if platform == "S1C":
        return (absolute_orbit - 172) % 175 + 1
    raise ValueError(platform)


def scan(s1_dir: Path) -> list[S1Zip]:
    """Inventory all S1 product zips in a directory, sorted by sensing start."""
    out = []
    for p in sorted(s1_dir.glob("S1*_IW_*.zip")):
        m = NAME_RE.match(p.name)
        if not m:
            continue
        with zipfile.ZipFile(p) as z:
            safe_name = z.namelist()[0].split("/")[0]
        d = m.groupdict()
        out.append(
            S1Zip(
                path=p,
                platform=d["platform"],
                kind=d["kind"].split("_")[0].rstrip("0123456789"),
                start=datetime.strptime(d["start"], "%Y%m%dT%H%M%S"),
                stop=datetime.strptime(d["stop"], "%Y%m%dT%H%M%S"),
                absolute_orbit=int(d["orbit"]),
                datatake=d["datatake"],
                relative_orbit=_relative_orbit(int(d["orbit"]), d["platform"]),
                safe_name=safe_name,
                is_cog=p.name.endswith("_COG.zip"),
            )
        )
    return sorted(out, key=lambda s: s.start)


class Sigma0Product(Sentinel1SarProduct):
    """Sentinel1SarProduct that calibrates to σ⁰ instead of β⁰ and can crop to an AOI.

    sarsen's terrain_correction geocodes whatever ``beta_nought()`` returns;
    swapping the LUT is the cleanest way to get a σ⁰ GTC out of it. The radar-
    geometry crop (via the GCP grid) keeps the full-scene calibrated image from
    ever materialising — without it a full IW GRD OOMs a small machine.
    """

    def __init__(self, *args, crop_bbox: tuple[float, float, float, float] | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.crop_bbox = crop_bbox

    def _crop_slices(self, buffer_deg: float = 0.01) -> tuple[slice, slice]:
        """(azimuth line, range pixel) index slices covering every GCP cell that overlaps crop_bbox.

        A cell is kept when the lon/lat bounding box of its four GCP corners overlaps the
        bbox (padded by ``buffer_deg`` for terrain displacement). Keeping only rows/columns
        that contain a GCP inside the bbox (+1 cell) clipped bbox corners whenever the
        coarse, rotated GCP lattice had no point near them, leaving NaN in the GTC there.
        The cell selection always contains that old "GCPs inside + 1 cell" window.
        """
        assert self.crop_bbox is not None
        w, s, e, n = self.crop_bbox
        w, s, e, n = w - buffer_deg, s - buffer_deg, e + buffer_deg, n + buffer_deg
        gcp = self.gcp
        lat = gcp.latitude.values  # (azimuth GCP row, range GCP column)
        lon = gcp.longitude.values

        def cell_corners(a: np.ndarray) -> np.ndarray:
            return np.stack([a[:-1, :-1], a[1:, :-1], a[:-1, 1:], a[1:, 1:]])

        lat_c, lon_c = cell_corners(lat), cell_corners(lon)
        hit = (lon_c.max(0) >= w) & (lon_c.min(0) <= e) & (lat_c.max(0) >= s) & (lat_c.min(0) <= n)
        if not hit.any():
            raise ValueError(f"AOI {self.crop_bbox} does not intersect product GCP grid")
        rows, cols = np.nonzero(hit)
        lines = gcp.line.values
        pixels = gcp.pixel.values
        i0, i1 = rows.min(), rows.max() + 1  # cell i spans GCP rows i..i+1
        j0, j1 = cols.min(), cols.max() + 1
        return slice(int(lines[i0]), int(lines[i1]) + 1), slice(int(pixels[j0]), int(pixels[j1]) + 1)

    def beta_nought(self, persist: bool = False) -> xr.DataArray:  # noqa: D102
        import xarray_sentinel

        measurement = self.measurement.data_vars["measurement"]
        if self.crop_bbox is not None:
            az_sl, rg_sl = self._crop_slices()
            measurement = measurement.isel(azimuth_time=az_sl, ground_range=rg_sl)
        sigma0 = xarray_sentinel.calibrate_intensity(measurement, self.calibration.sigmaNought)
        if persist:
            sigma0 = sigma0.persist()
        return sigma0.drop_vars(["pixel", "line"])


def prepare_dem(
    tile_paths: list[Path],
    bbox: tuple[float, float, float, float],
    out_path: Path,
    crs: str = "EPSG:32633",
    resolution: float = 20.0,
    buffer_deg: float = 0.03,
) -> Path:
    """Merge GLO-30 tiles, clip to AOI bbox (+buffer), reproject to a UTM grid.

    Bilinear resampling: GLO-30 posts are ~31 m x 28 m at 63°N, and nearest-neighbour onto
    the 20 m grid duplicates rows/columns, terracing the DEM (striped slope masks).
    """
    from rioxarray.merge import merge_arrays

    if out_path.exists():
        return out_path
    tiles = [rioxarray.open_rasterio(p).squeeze("band", drop=True) for p in tile_paths]
    dem = merge_arrays(tiles) if len(tiles) > 1 else tiles[0]
    w, s, e, n = bbox
    dem = dem.rio.clip_box(w - buffer_deg, s - buffer_deg, e + buffer_deg, n + buffer_deg)
    dem = dem.rio.reproject(crs, resolution=resolution, resampling=Resampling.bilinear)
    dem = dem.where(dem > -1000, 0.0).astype("float32")  # nodata/sea -> 0 m
    dem.rio.write_nodata(np.nan, inplace=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dem.rio.to_raster(out_path, compress="deflate")
    return out_path


def extract_safe(product: S1Zip, extract_dir: Path) -> Path:
    """Unzip a SAFE next to its siblings (cached); sarsen's rasterio reads need real files.

    The zip is unpacked into a hidden ``.<name>.partial`` directory that is renamed into
    place only when complete, so an interrupted extraction is redone instead of reused.
    """
    safe = extract_dir / product.safe_name
    if not safe.exists():
        tmp = extract_dir / f".{product.safe_name}.partial"
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir(parents=True)
        with zipfile.ZipFile(product.path) as z:
            z.extractall(tmp)
        (tmp / product.safe_name).rename(safe)
        shutil.rmtree(tmp)
    return safe


def gtc_sigma0(
    product: S1Zip,
    pol: str,
    dem_path: Path,
    out_tif: Path,
    chunks: int = 1024,
    extract_dir: Path = DEFAULT_EXTRACT_DIR,
    crop_bbox: tuple[float, float, float, float] | None = None,
) -> xr.DataArray:
    """Geocoded (GTC) σ⁰ for one polarisation onto the DEM grid. Cached by out_tif.

    sarsen writes to ``<out_tif stem>.partial.tif``, renamed onto out_tif only after the
    terrain correction finished, so an interrupted run never leaves a file that a later
    call would serve as a cache hit. Hit or miss, the file-backed raster is returned:
    sarsen's own return value is an unpersisted dask graph that would redo the whole
    terrain correction on every access.
    """
    if out_tif.exists() and out_tif.stat().st_size == 0:
        out_tif.unlink()  # leftover from an interrupted run
    if not out_tif.exists():
        safe = extract_safe(product, extract_dir)
        sar = Sigma0Product(str(safe), measurement_group=f"IW/{pol.upper()}", crop_bbox=crop_bbox)
        out_tif.parent.mkdir(parents=True, exist_ok=True)
        partial = out_tif.with_name(out_tif.stem + ".partial.tif")
        partial.unlink(missing_ok=True)
        apps.terrain_correction(
            sar,
            dem_urlpath=str(dem_path),
            output_urlpath=str(partial),
            chunks=chunks,
        )
        partial.replace(out_tif)
    return rioxarray.open_rasterio(out_tif).squeeze("band", drop=True)
