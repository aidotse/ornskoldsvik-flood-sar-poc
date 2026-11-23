"""Sentinel-1 Level-0 helpers on top of ``sentinel1decoder``.

Everything here operates on the extracted L0 SAFE (``*.dat`` measurement files).
The heavy lifting (packet parsing, FDBAQ decode) is sentinel1decoder's Rust core;
this module adds product discovery, chunk selection, the nominal-chirp matched
filter, and a deliberately minimal range-Doppler focus for small patches —
proof of decode, not a processor.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import sentinel1decoder as s1d
from sentinel1decoder.utilities import range_dec_to_sample_rate

F_REF = s1d.constants.F_REF  # 37.53472224 MHz reference frequency
C = s1d.constants.SPEED_OF_LIGHT_MPS
WAVELENGTH = s1d.constants.TX_WAVELENGTH_M


def find_dat(safe_dir: Path, pol: str = "vv") -> Path:
    """Locate the measurement .dat (not annot/index) for a polarisation."""
    cands = [
        p
        for p in safe_dir.glob(f"s1?-iw-raw-s-{pol.lower()}*.dat")
        if not p.name.endswith(("annot.dat", "index.dat"))
    ]
    if len(cands) != 1:
        raise FileNotFoundError(f"{safe_dir}: expected 1 {pol} measurement .dat, got {cands}")
    return cands[0]


def open_l0(safe_dir: Path, pol: str = "vv") -> s1d.Level0File:
    return s1d.Level0File(str(find_dat(safe_dir, pol)))


def echo_chunk_ids(f: s1d.Level0File, swath: int | None = None) -> list[int]:
    """Acquisition chunks that are pure echo (optionally one IW swath: 10/11/12)."""
    out = []
    for i in f.acquisition_chunks:
        m = f.get_acquisition_chunk_metadata(i)
        if {str(s) for s in m["Signal Type"]} != {"Echo"}:
            continue
        if swath is not None and set(m["Swath Number"]) != {swath}:
            continue
        out.append(i)
    return out


def nominal_chirp(md_row: pd.Series, oversample: int = 1) -> tuple[np.ndarray, float]:
    """Nominal TX chirp replica from packet metadata; returns (replica, fs).

    Uses the commanded pulse parameters (TXPSF, TXPRR, TXPL); the downlinked
    replicas (Tx Cal packets) would be the higher-fidelity alternative.
    """
    fs = range_dec_to_sample_rate(md_row["Range Decimation"]) * oversample
    txprr = float(md_row["Tx Ramp Rate"])  # Hz/s
    txpsf = float(md_row["Tx Pulse Start Frequency"])  # Hz
    txpl = float(md_row["Tx Pulse Length"])  # s
    n = int(round(txpl * fs))
    t = np.arange(n) / fs
    phase = 2 * np.pi * (txpsf * t + 0.5 * txprr * t**2)
    return np.exp(1j * phase).astype(np.complex64), fs


def range_compress(data: np.ndarray, replica: np.ndarray) -> np.ndarray:
    """Matched-filter range compression (frequency domain, per line)."""
    n = data.shape[1]
    nfft = int(2 ** np.ceil(np.log2(n + len(replica) - 1)))
    rep_f = np.conj(np.fft.fft(replica, nfft))
    out = np.fft.ifft(np.fft.fft(data, nfft, axis=1) * rep_f[None, :], axis=1)
    return out[:, : n].astype(np.complex64)


def effective_velocity(ephemeris: pd.DataFrame) -> float:
    """Approximate effective velocity: sqrt(v_ground * v_sat) ≈ platform speed scaled.

    Good enough for a small-patch quick-look focus; a real processor computes
    this per range gate from the orbit and geodetic geometry.
    """
    v = ephemeris[["X-axis velocity ECEF", "Y-axis velocity ECEF", "Z-axis velocity ECEF"]].to_numpy()
    v_sat = float(np.linalg.norm(v, axis=1).mean())
    # LEO scaling factor for effective velocity in the hyperbolic range model
    return v_sat * 0.94


def azimuth_focus_patch(
    rc: np.ndarray,
    prf: float,
    v_eff: float,
    slant_range_m: np.ndarray,
) -> np.ndarray:
    """Minimal azimuth matched filter (range-Doppler, zero Doppler centroid).

    Ignores range cell migration and TOPS azimuth steering — valid only for a
    small patch and as an energy/geometry sanity check against the SLC.
    """
    n_az, n_rg = rc.shape
    f_az = np.fft.fftfreq(n_az, d=1.0 / prf)
    ka = 2 * v_eff**2 / (WAVELENGTH * slant_range_m)  # azimuth FM rate per gate (Hz/s)
    rc_f = np.fft.fft(rc, axis=0)
    # azimuth history exp(-j4πR(η)/λ) ≈ exp(-jπ·Ka·η²) has spectrum ∝ exp(+jπf²/Ka) (the
    # convention range_compress's conj(replica) filter already relies on), so the matched
    # filter is its conjugate; exp(+jπf²/Ka) would double the azimuth chirp instead
    h = np.exp(-1j * np.pi * (f_az[:, None] ** 2) / ka[None, :])
    return np.fft.ifft(rc_f * h, axis=0).astype(np.complex64)


def ecef_to_latlon(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Geodetic lat/lon (deg) from ECEF (m); iterative WGS84, fine for km-level use."""
    a = s1d.constants.WGS84_SEMI_MAJOR_AXIS_M
    b = s1d.constants.WGS84_SEMI_MINOR_AXIS_M
    e2 = 1 - (b / a) ** 2
    lon = np.arctan2(y, x)
    p = np.hypot(x, y)
    lat = np.arctan2(z, p * (1 - e2))
    for _ in range(5):
        n = a / np.sqrt(1 - e2 * np.sin(lat) ** 2)
        h = p / np.cos(lat) - n
        lat = np.arctan2(z, p * (1 - e2 * n / (n + h)))
    return np.degrees(lat), np.degrees(lon)


def chunk_subsatellite_track(f: s1d.Level0File) -> pd.DataFrame:
    """Per acquisition chunk: mid sensing time + sub-satellite lat/lon from ephemeris.

    This is orbit-only geolocation (no look geometry) — the ground swath sits
    hundreds of km to the side; use it for coarse 'which chunk is near the AOI'
    questions, not for label attachment.
    """
    eph = f.ephemeris
    t_eph = eph["POD Solution Data Timestamp"].to_numpy(float)
    rows = []
    for i in f.acquisition_chunks:
        m = f.get_acquisition_chunk_metadata(i)
        t_mid = float(m["Coarse Time"].to_numpy(float).mean() + m["Fine Time"].to_numpy(float).mean())
        pos = [np.interp(t_mid, t_eph, eph[c].to_numpy(float)) for c in
               ("X-axis position ECEF", "Y-axis position ECEF", "Z-axis position ECEF")]
        lat, lon = ecef_to_latlon(*(np.array([p]) for p in pos))
        sig = {str(s) for s in m["Signal Type"]}
        rows.append({"chunk": i, "n_packets": len(m), "t_mid": t_mid,
                     "signal": "/".join(sorted(sig)), "swath": "/".join(str(s) for s in sorted(set(m["Swath Number"]))),
                     "subsat_lat": float(lat[0]), "subsat_lon": float(lon[0])})
    return pd.DataFrame(rows).set_index("chunk")


def l0_label_windows(
    gcp: xr.Dataset,  # noqa: F821 — xarray dataset (GRD GCP grid, group IW/*/gcp)
    polygons,
    aperture_s: float = 0.4,
    txpl_s: float = 52.4e-6,
) -> pd.DataFrame:
    """Forward-project map-space label polygons into raw (azimuth-time, slant-range) windows.

    Strategy (b) of docs/labels.md: each polygon's vertices are mapped through the
    GCP grid's (lat,lon)->(azimuth_time, slant_range_time) relation (scattered
    linear interpolation), then dilated by the physics kernels:
      - azimuth: ±aperture_s seconds (±~2.5 km synthetic-aperture mixing),
      - slant: +txpl_s two-way seconds (uncompressed chirp smear; one-sided,
        the echo of a scatterer arrives from t to t+TXPL).
    Returns one row per polygon; times are UTC ns (azimuth) and two-way seconds (slant).
    """
    from scipy.interpolate import griddata

    lat = gcp.latitude.values.ravel()
    lon = gcp.longitude.values.ravel()
    az = np.repeat(gcp.azimuth_time.values.astype("datetime64[ns]").astype("int64"),
                   gcp.sizes["slant_range_time"])
    sr = np.tile(gcp.slant_range_time.values, gcp.sizes["azimuth_time"])
    pts = np.column_stack([lon, lat])

    rows = []
    for i, geom in enumerate(polygons):
        try:
            xy = np.asarray(geom.exterior.coords) if geom.geom_type == "Polygon" else \
                np.concatenate([np.asarray(g.exterior.coords) for g in geom.geoms])
        except AttributeError:
            continue
        az_i = griddata(pts, az, xy[:, :2], method="linear")
        sr_i = griddata(pts, sr, xy[:, :2], method="linear")
        ok = np.isfinite(az_i) & np.isfinite(sr_i)
        if not ok.any():
            continue  # polygon outside this product's footprint
        t0 = np.datetime64(int(az_i[ok].min()), "ns") - np.timedelta64(int(aperture_s * 1e9), "ns")
        t1 = np.datetime64(int(az_i[ok].max()), "ns") + np.timedelta64(int(aperture_s * 1e9), "ns")
        rows.append({"polygon": i,
                     "azimuth_t0": t0, "azimuth_t1": t1,
                     "slant_t0_s": float(sr_i[ok].min()),
                     "slant_t1_s": float(sr_i[ok].max() + txpl_s)})
    return pd.DataFrame(rows)


def slant_ranges(md_row: pd.Series, n_samples: int, fs: float) -> np.ndarray:
    """Slant range (m) for each range sample from SWST + rank·PRI."""
    pri = float(md_row["PRI"])
    swst = float(md_row["SWST"])
    rank = int(md_row["Rank"])
    t0 = rank * pri + swst
    t = t0 + np.arange(n_samples) / fs
    return C * t / 2.0
