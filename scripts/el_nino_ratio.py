"""The notebook's "El Nino" experiment, as written and with the noise fixed.

The notebook compares a control run with one whose wet-regime noise amplitude
D_P is 1.5 times larger, and plots the ratio of their daily-precipitation PDFs
(its cell 21). This repeats that comparison under the notebook's scheme and
under the same scheme with the noise multiplied by sqrt(dt), at dt = 1 min with
MN19's Fig. 2b parameters, on bins common to both runs.

Usage: python -u scripts/el_nino_ratio.py [--years N]
Writes outputs/el_nino_ratio_<N>yr.json and figs/el_nino_ratio_<N>yr.pdf.
"""

import argparse
import json

import cmcrameri.cm as cmc
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from puffins import plotting as pplt  # noqa: E402

import check_against_mv19 as chk  # noqa: E402
import stoch_rain  # noqa: E402

SCHEMES = {"notebook as written": False, "noise times sqrt(dt)": True}
FACTOR = 1.5
EDGES = 10.0 ** np.arange(-1, 2.5001, 0.2)  # mm, wet-day daily totals
THRESHOLDS = [20, 50, 100]  # mm


def daily_wet(d_wet_factor, sqrt_noise, years, seed):
    P = chk.P
    si = stoch_rain.notebook_units(
        q_c=P["q_c"], b=P["b"], q_min=P["q_min"], evap=P["evap"], conv=P["conv"],
        rain_rate=P["rain_rate"], d_wet=P["d_wet"] * d_wet_factor, d_dry=P["d_dry"],
        dt_hours=1 / 60, q_init=chk.Q_INIT)
    res = stoch_rain.integrate(si, round(365 * years * 1440), sqrt_noise=sqrt_noise,
                               wet_source=True, floor_restore=False, seed=seed)
    day = res["day_rain_m"] / stoch_rain.MM
    return day[day > 0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", type=int, default=100)
    args = parser.parse_args()
    centers = np.sqrt(EDGES[:-1] * EDGES[1:])
    out = {}
    for k, (name, sqrt_noise) in enumerate(SCHEMES.items()):
        ctrl = daily_wet(1.0, sqrt_noise, args.years, seed=200 + 2 * k)
        nino = daily_wet(FACTOR, sqrt_noise, args.years, seed=201 + 2 * k)
        n_c, _ = np.histogram(ctrl, EDGES)
        n_n, _ = np.histogram(nino, EDGES)
        ratio = (n_n / nino.size) / (n_c / ctrl.size)
        exceed = {t: [float((ctrl > t).mean()), float((nino > t).mean())] for t in THRESHOLDS}
        out[name] = dict(n_wet_days=[int(ctrl.size), int(nino.size)], counts_ctrl=n_c.tolist(),
                         counts_nino=n_n.tolist(), ratio=ratio.tolist(), exceed=exceed)
        print(f"\n{name}: {ctrl.size} control and {nino.size} El Nino wet days")
        print(f"{'bin center (mm)':>16s}{'ctrl count':>12s}{'nino count':>12s}{'ratio':>8s}")
        for c, a, b, r in zip(centers, n_c, n_n, ratio):
            print(f"{c:>16.3g}{a:>12d}{b:>12d}{r:>8.3f}")
        for t, (pc, pn) in exceed.items():
            print(f"P(daily > {t} mm | wet day): control {pc:.4f}, El Nino {pn:.4f}, "
                  f"ratio {pn / pc:.3f}")

    path = chk.ROOT / "outputs" / f"el_nino_ratio_{args.years}yr.json"
    path.write_text(json.dumps(dict(edges_mm=EDGES.tolist(), runs=out), indent=1))
    print(f"wrote {path}")

    plt.rcParams.update(pplt.plt_rc_params_custom)
    plt.rcParams["pdf.fonttype"] = 42
    fig, ax = plt.subplots(figsize=(4.5, 3.2), constrained_layout=True)
    pplt.mark_y0(ax=ax, y0=1)
    for (name, res), color in zip(out.items(), [cmc.batlow(0.72), cmc.batlow(0.1)]):
        ok = (np.array(res["counts_ctrl"]) >= 10) & (np.array(res["counts_nino"]) >= 10)
        ax.semilogx(centers[ok], np.array(res["ratio"])[ok], "o", ms=3, color=color)
        y_last = np.array(res["ratio"])[ok][-1]
        ax.annotate(name, xy=(centers[ok][-1], y_last), xytext=(5, 0),
                    textcoords="offset points", color=color, va="center")
    ax.set(xlabel="daily precipitation, wet days (mm)",
           ylabel=f"PDF ratio, D$_P$ x {FACTOR} / control", xlim=(0.08, 1500))
    fig_path = chk.ROOT / "figs" / f"el_nino_ratio_{args.years}yr.pdf"
    fig.savefig(fig_path)
    print(f"wrote {fig_path}")


if __name__ == "__main__":
    main()
