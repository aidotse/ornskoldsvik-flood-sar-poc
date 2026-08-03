#!/usr/bin/env python
"""Slice the first N space packets out of a real L0 .dat into a small pytest fixture.

CCSDS packets are variable length: 6-byte primary header whose last 2 bytes hold
(packet data length - 1); total packet size = 6 + data_length + 1.
"""

import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rawsar import l0  # noqa: E402


def slice_packets(dat: Path, out: Path, n_packets: int, max_bytes: int) -> None:
    with open(dat, "rb") as f:
        data = bytearray()
        for _ in range(n_packets):
            hdr = f.read(6)
            if len(hdr) < 6:
                break
            (data_len_minus1,) = struct.unpack(">H", hdr[4:6])
            body = f.read(data_len_minus1 + 1)
            if len(data) + 6 + len(body) > max_bytes:
                break
            data += hdr + body
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(bytes(data))
    print(f"{out}: {len(data)/1e6:.2f} MB")


if __name__ == "__main__":
    import glob

    for pat, name in [
        ("data/extracted/S1A_IW_RAW*.SAFE", "s1a_l0_head.dat"),
        ("data/extracted/S1C_IW_RAW*.SAFE", "s1c_l0_head.dat"),
    ]:
        safe = Path(glob.glob(pat)[0])
        slice_packets(l0.find_dat(safe), Path("tests/fixtures") / name, n_packets=3000, max_bytes=8_000_000)
