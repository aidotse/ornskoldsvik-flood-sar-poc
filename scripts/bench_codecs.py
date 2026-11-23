#!/usr/bin/env python
"""Codec benchmarks on REAL data: decoded L0 I/Q and geocoded sigma0.

Measures compression ratio and encode/decode throughput for zarr-v3 codec stacks:
zstd (levels) x {no shuffle, byte-shuffle, bit-shuffle} on
  (a) complex64 I/Q as-is,
  (b) I/Q quantized to 2xint16 (scale = 99.99th percentile / 32760),
  (c) sigma0 float32 GTC raster.
Writes docs/codec_benchmarks.csv and prints the table.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from zarr.codecs import BloscCodec, ZstdCodec

ROOT = Path(__file__).resolve().parents[1]


def bench_stack(name: str, arr: np.ndarray, compressors: list) -> dict:
    """Round-trip arr through a zarr in-memory store with the given compressor stack."""
    import zarr

    store = zarr.storage.MemoryStore()
    t0 = time.time()
    z = zarr.create_array(store, name="x", shape=arr.shape, dtype=arr.dtype,
                          chunks=arr.shape, compressors=compressors, overwrite=True)
    z[:] = arr
    t_enc = time.time() - t0
    nbytes_stored = sum(len(buf) for buf in store._store_dict.values())
    t0 = time.time()
    back = z[:]
    t_dec = time.time() - t0
    assert back.shape == arr.shape
    return {
        "data": name,
        "ratio": round(arr.nbytes / nbytes_stored, 3),
        "enc_MBps": round(arr.nbytes / 1e6 / t_enc, 0),
        "dec_MBps": round(arr.nbytes / 1e6 / t_dec, 0),
    }


def main() -> None:
    import glob

    from rawsar import l0

    # (a) real decoded I/Q: one echo chunk, S1C rel-22 (label-anchor track)
    f = l0.open_l0(Path(glob.glob(str(ROOT / "data/extracted/S1C_IW_RAW*.SAFE"))[0]))
    ids = l0.echo_chunk_ids(f, swath=10)
    iq = f.get_acquisition_chunk_data(ids[len(ids) // 2], try_load_from_file=False)
    iq = iq[:512]  # 512 lines x ~23722 samples ~ 97 MB: plenty for stable numbers

    # (b) int16 quantization (I and Q interleaved on last axis)
    scale = np.percentile(np.abs(np.concatenate([iq.real, iq.imag])), 99.99) / 32760.0
    iq16 = np.stack([np.clip(np.round(iq.real / scale), -32768, 32767),
                     np.clip(np.round(iq.imag / scale), -32768, 32767)],
                    axis=-1).astype(np.int16)
    # quantization error for the record
    err = np.abs((iq16[..., 0] * scale + 1j * iq16[..., 1] * scale) - iq)
    snr_db = 10 * np.log10(float((np.abs(iq) ** 2).mean()) / float((err ** 2).mean()))

    # (c) sigma0 raster
    import rioxarray
    s0 = rioxarray.open_rasterio(ROOT / "data/gtc/S1C_022_20250909_VV_sigma0.tif").values.astype("float32")

    stacks = {
        "zstd1": [ZstdCodec(level=1)],
        "zstd3": [ZstdCodec(level=3)],
        "zstd9": [ZstdCodec(level=9)],
        "blosc-zstd3+byteshuf": [BloscCodec(cname="zstd", clevel=3, shuffle="shuffle")],
        "blosc-zstd3+bitshuf": [BloscCodec(cname="zstd", clevel=3, shuffle="bitshuffle")],
        "blosc-zstd9+bitshuf": [BloscCodec(cname="zstd", clevel=9, shuffle="bitshuffle")],
    }
    datasets = {"iq_complex64": iq, "iq_int16(2x)": iq16, "sigma0_float32": s0}

    rows = []
    for cname, codecs in stacks.items():
        for dname, arr in datasets.items():
            r = bench_stack(dname, arr, codecs)
            r["codec"] = cname
            rows.append(r)
            print(f"{cname:<22} {dname:<16} ratio={r['ratio']:<7} "
                  f"enc={r['enc_MBps']:>6} MB/s dec={r['dec_MBps']:>6} MB/s")

    df = pd.DataFrame(rows)[["codec", "data", "ratio", "enc_MBps", "dec_MBps"]]
    df.attrs["int16_quantization_snr_db"] = snr_db
    out = ROOT / "docs/codec_benchmarks.csv"
    out.parent.mkdir(parents=True, exist_ok=True)  # docs/ is gitignored, absent in a fresh clone
    df.to_csv(out, index=False)
    print(f"\nint16 quantization SNR: {snr_db:.1f} dB (scale from 99.99th pct)")
    print("wrote", out)


if __name__ == "__main__":
    main()
