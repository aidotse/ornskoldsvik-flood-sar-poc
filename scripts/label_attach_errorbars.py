#!/usr/bin/env python
"""Quantify method-(a) raw-echo geolocation error against GCP truth.

For each AOI echo chunk stored in the POC zarr (S1C rel-22 Sept 9, VV):
  prediction = orbit+timing only — ephemeris position/velocity at the chunk's mid
    packet time, right-of-track look direction, look angle from the mid slant-range
    gate via spherical law of cosines. No annotations used.
  truth = same-datatake GRD GCP grid interpolated at (mid azimuth time, mid
    slant-range time).
Reports the horizontal error and its along/cross-track split, plus the irreducible
azimuth-aperture kernel for context. Writes docs/label_attach_errorbars.csv.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import zarr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rawsar import l0  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
GPS_EPOCH = pd.Timestamp("1980-01-06")
LEAP = pd.Timedelta(seconds=18)  # GPS-UTC, 2025

A = 6378137.0
B = 6356752.314245


def lla_to_ecef(lat, lon, h=0.0):
    lat, lon = np.radians(lat), np.radians(lon)
    e2 = 1 - (B / A) ** 2
    n = A / np.sqrt(1 - e2 * np.sin(lat) ** 2)
    return np.array([(n + h) * np.cos(lat) * np.cos(lon),
                     (n + h) * np.cos(lat) * np.sin(lon),
                     (n * (1 - e2) + h) * np.sin(lat)])


def predict_ground(pos, vel, slant_range):
    """Orbit-only ground point: right-of-track look, WGS84 ellipsoid intersection.

    Bisection on the look angle until the point at the given slant range lies on
    the ellipsoid. (A scalar 'local earth radius' here costs 15-20 km cross-track —
    measured before this fix; the ellipsoid solve is the honest minimum geometry.)
    """
    u = vel / np.linalg.norm(vel)
    up = pos / np.linalg.norm(pos)
    right = np.cross(u, up)
    right /= np.linalg.norm(right)

    def height(theta):
        g = pos + slant_range * (np.sin(theta) * right - np.cos(theta) * up)
        return (g[0] ** 2 + g[1] ** 2) / A**2 + g[2] ** 2 / B**2 - 1.0

    lo, hi = np.radians(10.0), np.radians(60.0)
    if height(lo) > 0:  # already outside at the near end: no intersection sanity
        raise ValueError("no ellipsoid intersection at look>10deg")
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if height(mid) > 0:
            hi = mid
        else:
            lo = mid
    g = pos + slant_range * (np.sin(lo) * right - np.cos(lo) * up)
    lat, lon = l0.ecef_to_latlon(*(np.array([c]) for c in g))
    return float(lat[0]), float(lon[0])


# (store raw group, RAW safe glob, GRD-for-GCP-truth safe glob)
DATATAKES = [
    ("raw/00809C_vv", "S1C_IW_RAW*00809C*.SAFE", "S1C_IW_GRDH*00809C_1E3D.SAFE"),
    ("raw/0796C3_vv", "S1A_IW_RAW*0796C3*.SAFE", "S1A_IW_GRDH*0796C3_FC8C.SAFE"),
]


def main() -> None:

    store = zarr.open_group(ROOT / "data/zarr/poc.zarr", mode="r")
    all_rows = []
    for group, raw_glob, grd_glob in DATATAKES:
        try:
            raw = store[group]
        except KeyError:
            print(f"{group} not in store, skipping")
            continue
        all_rows += measure_datatake(store, raw, group, raw_glob, grd_glob)

    df = pd.DataFrame(all_rows)
    df.to_csv(ROOT / "docs/label_attach_errorbars.csv", index=False)
    for g, sub in df.groupby("datatake"):
        print(f"\n{g}: median |err| {sub.err_km.median():.2f} km "
              f"(along {sub.along_km.abs().median():.2f}, cross {sub.cross_km.median():.2f})")
    print("azimuth aperture kernel ±2.5 km is irreducible on top")
    print("wrote docs/label_attach_errorbars.csv")


def measure_datatake(store, raw, group: str, raw_glob: str, grd_glob: str) -> list[dict]:
    import glob

    f = l0.open_l0(Path(glob.glob(str(ROOT / "data/extracted" / raw_glob))[0]))
    eph = f.ephemeris
    t_eph = eph["POD Solution Data Timestamp"].to_numpy(float)
    P = eph[[f"{a}-axis position ECEF" for a in "XYZ"]].to_numpy(float)
    V = eph[[f"{a}-axis velocity ECEF" for a in "XYZ"]].to_numpy(float)

    grd_safe = glob.glob(str(ROOT / "data/extracted" / grd_glob))[0]
    gcp = xr.open_dataset(grd_safe, engine="sentinel-1", group="IW/VV/gcp")

    rows = []
    for name in sorted(raw.group_keys()):
        g = raw[name]
        cid = int(name.split("_")[1])
        md = f.get_acquisition_chunk_metadata(cid)
        r0 = md.iloc[0]
        t_mid = float(md["Coarse Time"].to_numpy(float).mean()
                      + md["Fine Time"].to_numpy(float).mean())
        from sentinel1decoder.utilities import range_dec_to_sample_rate
        fs = range_dec_to_sample_rate(r0["Range Decimation"])
        n_samp = g["iq"].shape[1]
        # two-way time of mid range gate
        t_slant_mid = float(r0["Rank"]) * float(r0["PRI"]) + float(r0["SWST"]) + n_samp / (2 * fs)
        slant = l0.C * t_slant_mid / 2

        pos = np.array([np.interp(t_mid, t_eph, P[:, i]) for i in range(3)])
        vel = np.array([np.interp(t_mid, t_eph, V[:, i]) for i in range(3)])
        plat, plon = predict_ground(pos, vel, slant)

        t_utc = GPS_EPOCH + pd.Timedelta(seconds=t_mid) - LEAP
        truth = gcp.interp(azimuth_time=np.datetime64(t_utc), slant_range_time=t_slant_mid)
        tlat, tlon = float(truth.latitude), float(truth.longitude)

        err = np.linalg.norm(lla_to_ecef(plat, plon) - lla_to_ecef(tlat, tlon))
        # along/cross decomposition using track direction on ground
        u = vel / np.linalg.norm(vel)
        dvec = lla_to_ecef(plat, plon) - lla_to_ecef(tlat, tlon)
        along = float(np.dot(dvec, u))
        cross = float(np.sqrt(max(err**2 - along**2, 0)))
        rows.append({"datatake": group.split("/")[1], "chunk": name, "swath": g.attrs["swath"],
                     "pred_lat": round(plat, 4), "pred_lon": round(plon, 4),
                     "gcp_lat": round(tlat, 4), "gcp_lon": round(tlon, 4),
                     "err_km": round(err / 1e3, 2), "along_km": round(along / 1e3, 2),
                     "cross_km": round(cross / 1e3, 2)})
        print(rows[-1])
    return rows


if __name__ == "__main__":
    main()
