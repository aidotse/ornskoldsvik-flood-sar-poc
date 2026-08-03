#!/usr/bin/env python
"""Event timeline: SMHI daily precipitation + S1 passes + EMS product times.

Writes docs/figures/event_timeline.png and prints the numbers it plots.
SMHI CSVs have variable preamble/column layouts — parse by header row, not position.
"""

import io
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def read_smhi_csv(path: Path) -> pd.DataFrame:
    """Return DataFrame(date, mm) from an SMHI metobs CSV (any station layout)."""
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    hdr = next(i for i, ln in enumerate(lines) if ln.startswith(("Från Datum", "Representativt dygn", "Datum")))
    df = pd.read_csv(io.StringIO("\n".join(lines[hdr:])), sep=";")
    datecol = next(c for c in df.columns if "Representativt" in c or c == "Datum")
    valcol = next(c for c in df.columns if "Nederbörd" in c)
    out = pd.DataFrame({"date": pd.to_datetime(df[datecol], errors="coerce"),
                        "mm": pd.to_numeric(df[valcol], errors="coerce")}).dropna()
    return out


def main() -> None:
    win = slice("2025-08-20", "2025-10-05")
    stations = {}
    for f in sorted((ROOT / "data/smhi").glob("precip_daily_*.csv")):
        name = f.stem.split("_")[-1]
        s = read_smhi_csv(f).set_index("date")["mm"].loc[win]
        stations[name] = s
        top = s.sort_values(ascending=False).head(5)
        print(f"{name}: {len(s)} days in window; wettest:",
              {d.strftime('%m-%d'): float(v) for d, v in top.items()})

    s1_passes = {  # sensing UTC, on disk (o) / manifest (m)
        "2025-09-03 05:06": "m", "2025-09-04 16:13": "o", "2025-09-07 05:21": "o",
        "2025-09-08 05:14": "o", "2025-09-08 16:30": "o", "2025-09-09 05:05": "o",
        "2025-09-09 16:21": "o", "2025-09-10 16:14": "m", "2025-09-15 05:06": "m",
        "2025-09-16 16:13": "m", "2025-09-21 05:05": "m",
    }
    ems = {"FEP": "2025-09-10 16:14", "DEL(ICEYE)": "2025-09-11 23:06",
           "MONIT01": "2025-09-12 16:57", "MONIT02": "2025-09-15 05:06"}

    fig, ax = plt.subplots(figsize=(13, 5), layout="constrained")
    w = 0.4
    for k, (name, s) in enumerate(stations.items()):
        ax.bar(s.index + pd.Timedelta(days=w * k - w / 2), s.values, width=w,
               label=f"SMHI {name}", alpha=0.8)
    ymax = max(s.max() for s in stations.values()) * 1.15
    for t, kind in s1_passes.items():
        ts = pd.Timestamp(t)
        ax.axvline(ts, color="k" if kind == "o" else "gray",
                   ls="-" if kind == "o" else ":", lw=1, alpha=0.7)
    for i, (name, t) in enumerate(ems.items()):
        ts = pd.Timestamp(t)
        ax.axvline(ts, color="crimson", lw=1.2, alpha=0.8)
        ax.annotate(name, (ts, ymax * (0.99 - 0.08 * i)), color="crimson", fontsize=8,
                    ha="left", xytext=(3, 0), textcoords="offset points",
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75))
    ax.plot([], [], color="k", lw=1, label="S1 pass (on disk)")
    ax.plot([], [], color="gray", ls=":", lw=1, label="S1 pass (manifest)")
    ax.plot([], [], color="crimson", lw=1.2, label="EMS product source time")
    # Moälven discharge at Västersel (4.6 km from Gottne), secondary axis
    dis_csv = ROOT / "data/smhi/discharge_daily_1583_vastersel.csv"
    if dis_csv.exists():
        rows = [ln.split(";") for ln in dis_csv.read_text().splitlines()
                if ln.startswith(("2025-08", "2025-09", "2025-10"))]
        q = pd.Series({pd.Timestamp(p[0]): float(p[1]) for p in rows if len(p) > 1
                       and p[1].replace(".", "", 1).isdigit()}).loc[win]
        ax2 = ax.twinx()
        ax2.plot(q.index, q.values, color="tab:blue", lw=1.8)
        ax2.set_ylabel("Moälven discharge, Västersel [m³/s]", color="tab:blue")
        ax2.tick_params(axis="y", colors="tab:blue")
        print(f"discharge: Aug mean {q.loc['2025-08'].mean():.1f} → peak "
              f"{q.max():.0f} m³/s on {q.idxmax():%b %d}")
        ax.plot([], [], color="tab:blue", lw=1.8, label="discharge (right axis)")

    ax.set(ylabel="daily precipitation [mm]",
           title="EMSR841 Örnsköldsvik flood — rain, discharge, S1 passes, EMS labels")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.legend(loc="upper right", fontsize=8)
    out = ROOT / "docs/figures/event_timeline.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print("wrote", out)


if __name__ == "__main__":
    main()
