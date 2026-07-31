"""Parser sanity tests on small real-data fixtures (first packets of real L0 files)."""

from pathlib import Path

import numpy as np
import pytest
import sentinel1decoder as s1d

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(params=["s1a_l0_head.dat", "s1c_l0_head.dat"], ids=["S1A", "S1C"])
def l0file(request):
    p = FIXTURES / request.param
    if not p.exists():
        pytest.skip("fixture not generated (scripts/make_l0_fixture.py)")
    return s1d.Level0File(str(p))


def test_metadata_parses(l0file):
    md = l0file.packet_metadata
    assert len(md) > 100
    # physical plausibility of commanded timing for IW
    pri = md["PRI"].to_numpy(float)
    assert (pri > 4e-4).all() and (pri < 8e-4).all(), "IW PRI must be ~0.5-0.7 ms"
    assert set(int(s) for s in md["Swath Number"].unique()) <= {10, 11, 12, 43, 93}


def test_packet_stream_is_contiguous(l0file):
    md = l0file.packet_metadata
    counts = md["Space Packet Count"].to_numpy(int)
    assert (np.diff(counts) == 1).all(), "sliced packet stream must be gap-free"


def test_echo_decode_statistics(l0file):
    echo_chunks = [
        i
        for i in l0file.acquisition_chunks
        if {str(s) for s in l0file.get_acquisition_chunk_metadata(i)["Signal Type"]} == {"Echo"}
    ]
    if not echo_chunks:
        pytest.skip("fixture holds no complete echo chunk")
    d = l0file.get_acquisition_chunk_data(echo_chunks[0], try_load_from_file=False)
    assert d.dtype == np.complex64
    assert d.ndim == 2 and d.shape[0] > 10
    finite = np.isfinite(d).all()
    assert finite
    # FDBAQ-decoded thermal-noise+clutter echoes: zero-mean, std in a broad sane band
    assert abs(d.real.mean()) < 1.0 and abs(d.imag.mean()) < 1.0
    assert 1.0 < d.std() < 200.0


def test_ephemeris_subcommutation(l0file):
    eph = l0file.ephemeris
    assert len(eph) > 0
    r = np.linalg.norm(
        eph[["X-axis position ECEF", "Y-axis position ECEF", "Z-axis position ECEF"]].to_numpy(float),
        axis=1,
    )
    # S1 orbit radius ~ 7070 km
    assert ((r > 7.0e6) & (r < 7.2e6)).all()
