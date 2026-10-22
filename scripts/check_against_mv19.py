"""Integrate the on-off model under the notebook's scheme and the paper's, and
compare both with Martinez-Villalobos and Neelin (2019, J. Atmos. Sci.; MN19).

All configurations use MN19's Fig. 2b parameters (q_c = 65 mm, b = 0.2 mm,
E = 0.1 mm/h, C = 0.2 mm/h, R_0 = 9 mm/h, D_P = 17 and D_E = 3 mm h^-1/2,
a floor at q = 1 mm) and start from q = 64.9 mm, as the notebook does:

  notebook        the notebook's scheme: noise times dt, E + C kept while
                  raining, q clamped to 1 mm at the floor; dt = 1 min
  noise fixed     as notebook, with the noise times sqrt(dt)
  paper 1 min     MN19's scheme: noise times sqrt(dt), E + C = 0 while raining,
                  q restored to its previous value at the floor; dt = 1 min
  paper 0.6 s     as paper 1 min, at MN19's Fig. 2 time step of 0.6 s
  noise fixed,    as noise fixed, with the notebook's "El Nino" change,
  El Nino         D_P times 1.5

Usage: python -u scripts/check_against_mv19.py [--years N]
Writes outputs/check_against_mv19_<N>yr.json and figs/check_against_mv19_<N>yr.pdf.
"""

import argparse
import json
import math
from pathlib import Path

import cmcrameri.cm as cmc
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from puffins import plotting as pplt  # noqa: E402

import stoch_rain  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
P = stoch_rain.PAPER_FIG2
Q_INIT = 64.9  # mm, the notebook's initial condition

CONFIGS = {
    "notebook": dict(dt_hours=1 / 60, sqrt_noise=False, wet_source=True, floor_restore=False),
    "noise fixed": dict(dt_hours=1 / 60, sqrt_noise=True, wet_source=True, floor_restore=False),
    "paper 1 min": dict(dt_hours=1 / 60, sqrt_noise=True, wet_source=False, floor_restore=True),
    "paper 0.6 s": dict(dt_hours=0.6 / 3600, sqrt_noise=True, wet_source=False,
                        floor_restore=True),
    "noise fixed, El Nino": dict(dt_hours=1 / 60, sqrt_noise=True, wet_source=True,
                                 floor_restore=False, d_wet_factor=1.5),
}

# MN19 Fig. 2b: fit of eq. (2) to the on-off model's daily precipitation
MN19_DAILY_FIT = dict(tau_P=0.50, P_L=29.7)


def continuum_theory(d_wet, d_dry, rain_rate, src, wet_source, b):
    """First-passage predictions behind MN19 eq. (10), eq. (B2) and appendix B.

    A wet spell is a first passage over the distance b against the drift
    v = R_0 - (E + C) when the scheme keeps E + C while raining, v = R_0
    otherwise: an inverse Gaussian with mean duration b / v and variance
    b D_P^2 / v^3, so the cutoff t_L = 2 var / mean = 2 D_P^2 / v^2. The
    accumulation is R_0 times the duration. With v = R_0 these are MN19's
    s_bar = b, s_L = 2 D_P^2 / R_0, t_bar = b / R_0, t_L = 2 D_P^2 / R_0^2.
    Dry spells are first passages over b with drift E + C (appendix B), the
    floor at 1 mm ignored. Units: mm, hours.
    """
    v = rain_rate - src if wet_source else rain_rate
    return dict(
        accum_mean_mm=rain_rate * b / v,
        accum_cutoff_mm=2 * rain_rate * d_wet**2 / v**2,
        wet_mean_min=60 * b / v,
        wet_cutoff_h=2 * d_wet**2 / v**2,
        dry_mean_min=60 * b / src,
        dry_cutoff_h=2 * d_dry**2 / src**2,
    )


def appendix_a_fit(x, per_decade=10, min_count=10):
    """MN19 appendix A: regress log p on log x and x over log-spaced bins.

    Returns the power-law exponent tau = -c2 and cutoff x_L = -1/c3 of
    p ~ x^-tau exp(-x / x_L), using bins with at least ``min_count`` values.
    """
    edges = 10.0 ** np.arange(math.floor(np.log10(x.min())),
                              math.ceil(np.log10(x.max())) + 1e-9, 1 / per_decade)
    counts, _ = np.histogram(x, edges)
    keep = counts >= min_count
    centers = np.sqrt(edges[:-1] * edges[1:])[keep]
    pdf = counts[keep] / (x.size * np.diff(edges)[keep])
    design = np.column_stack([np.ones(centers.size), np.log(centers), centers])
    c, *_ = np.linalg.lstsq(design, np.log(pdf), rcond=None)
    return dict(tau=-c[1], cutoff=-1 / c[2] if c[2] < 0 else math.inf, n_bins=int(keep.sum()))


def summarize(res, rain_rate):
    dt_h = res["dt"] / 3600
    accum = res["accum_m"] / stoch_rain.MM
    wet_h = res["event_steps"] * dt_h
    dry_h = res["dry_steps"][1:] * dt_h  # the first dry spell starts at t = 0
    day = res["day_rain_m"] / stoch_rain.MM
    wet = day > 0
    all_day = day >= rain_rate * 24 * (1 - 1e-9)
    n_wet = wet.sum()
    events_wet_days = res["day_events"][wet]
    daily_wet = day[wet]
    q_hist = res["q_hist"]
    q_edges_mm = 10.0 ** stoch_rain.HIST_LOG10_EDGES
    frac_q_over_100 = q_hist[q_edges_mm[:-1] >= 100].sum() / q_hist.sum()
    days = res["n_steps"] * res["dt"] / 86400
    bud = {k: v / stoch_rain.MM / days for k, v in res["budget_m"].items()}
    return dict(
        budget_source_mm_day=bud["source"],
        budget_rain_mm_day=bud["rain"],
        budget_noise_mm_day=bud["noise"],
        budget_floor_mm_day=bud["floor"],
        budget_q_change_mm_day=bud["q_change"],
        budget_residual_mm_day=bud["q_change"] - (bud["source"] - bud["rain"] + bud["noise"]
                                                  + bud["floor"]),
        n_events=int(accum.size),
        n_days=int(day.size),
        accum_mean_mm=accum.mean(),
        accum_cutoff_mm=2 * accum.var() / accum.mean(),  # MN19 eq. (10)
        accum_max_mm=accum.max(),
        wet_mean_min=60 * wet_h.mean(),
        wet_cutoff_h=2 * wet_h.var() / wet_h.mean(),
        wet_max_h=wet_h.max(),
        dry_mean_min=60 * dry_h.mean(),
        dry_cutoff_h=2 * dry_h.var() / dry_h.mean(),
        frac_time_raining=res["n_rain_steps"] / res["n_steps"],
        mean_precip_mm_day=day.mean(),
        wet_day_frac=n_wet / day.size,
        all_day_rain_frac_of_wet_days=all_day.sum() / n_wet,
        events_per_wet_day_mean=events_wet_days.mean(),
        events_per_wet_day_var=events_wet_days.var(),
        daily_wet_mean_mm=daily_wet.mean(),
        daily_wet_max_mm=daily_wet.max(),
        daily_PL_mom_mm=daily_wet.var() / daily_wet.mean(),  # MN19 eq. (D6)
        daily_tauP_mom=1 - daily_wet.mean() ** 2 / daily_wet.var(),
        daily_fit=appendix_a_fit(daily_wet),
        accum_fit=appendix_a_fit(accum),
        q_max_mm=res["q_max_m"] / stoch_rain.MM,
        frac_minutes_q_over_100mm=frac_q_over_100,
        n_floor=res["n_floor"],
        n_steps=res["n_steps"],
        wall_s=res["wall_s"],
    )


def log_pdf(x, lo, hi, per_decade=10, min_count=10):
    edges = 10.0 ** np.arange(lo, hi + 1e-9, 1 / per_decade)
    counts, _ = np.histogram(x, edges)
    pdf = counts / (x.size * np.diff(edges))
    centers = np.sqrt(edges[:-1] * edges[1:])
    keep = counts >= min_count
    return centers[keep], pdf[keep]


def make_figure(results, path):
    plt.rcParams.update(pplt.plt_rc_params_custom)
    plt.rcParams["pdf.fonttype"] = 42
    shown = ["notebook", "noise fixed", "paper 0.6 s"]
    colors = dict(zip(shown, [cmc.batlow(x) for x in (0.72, 0.45, 0.1)]))
    labels = {"notebook": "notebook as written (1 min)",
              "noise fixed": "noise times sqrt(dt) (1 min)",
              "paper 0.6 s": "MN19 scheme (0.6 s)"}
    fig, axs = plt.subplots(1, 3, figsize=(10, 3.4), constrained_layout=True)

    for name in shown:
        res, color = results[name], colors[name]
        x, p = log_pdf(res["accum_m"] / stoch_rain.MM, -3, 5)
        axs[0].loglog(x, p, ".", color=color, label=labels[name])
        day = res["day_rain_m"] / stoch_rain.MM
        x, p = log_pdf(day[day > 0], -3, 3)
        axs[1].loglog(x, p, ".", color=color)
        edges = 10.0 ** stoch_rain.HIST_LOG10_EDGES
        h = res["q_hist"]
        pdf = h / (h.sum() * np.diff(edges))
        keep = h >= 10
        axs[2].loglog(np.sqrt(edges[:-1] * edges[1:])[keep], pdf[keep], ".", color=color)

    # MN19 eq. (9) with s_bar = b and s_L = 2 D_P^2 / R_0 (their eq. 10)
    s = np.logspace(-3, 3, 400)
    s_bar, s_L = P["b"], 2 * P["d_wet"] ** 2 / P["rain_rate"]
    B = s_bar / np.sqrt(np.pi * s_L) * np.exp(2 * s_bar / s_L)
    axs[0].loglog(s, B * np.exp(-s_bar**2 / (s_L * s)) * np.exp(-s / s_L) * s**-1.5, "k--",
                  lw=0.8, label=f"MN19 eq. (9), $s_L$ = {s_L:.0f} mm")
    # MN19 eq. (2) at their Fig. 2b fitted parameters
    tau, PL = MN19_DAILY_FIT["tau_P"], MN19_DAILY_FIT["P_L"]
    x = np.logspace(-3, 2.5, 400)
    A = PL ** (tau - 1) / math.gamma(1 - tau)
    axs[1].loglog(x, A * x**-tau * np.exp(-x / PL), "k--", lw=0.8)
    axs[1].text(0.03, 0.03, f"dashed: MN19 eq. (2) at their\nfitted $\\tau_P$ = {tau}, "
                f"$P_L$ = {PL} mm", transform=axs[1].transAxes, ha="left", va="bottom")
    axs[2].axvline(P["q_c"], color="0.4", lw=0.5)
    axs[2].text(P["q_c"] * 1.08, 2e-8, "$q_c$", color="0.4")

    axs[0].set(xlabel="event accumulation (mm)", ylabel="PDF (mm$^{-1}$)", ylim=(1e-9, 1e3))
    axs[1].set(xlabel="daily precipitation, wet days (mm)", ylim=(1e-7, 1e2))
    axs[2].set(xlabel="column water vapor (mm)", ylim=(1e-8, 1))
    for n, (ax, title) in enumerate(zip(axs, ["accumulations", "daily precipitation",
                                              "CWV, sampled each minute"])):
        pplt.panel_label(n, ax=ax, extra_text=title)
    axs[0].legend(loc="lower left", frameon=False, fontsize="small")
    fig.savefig(path)
    print(f"wrote {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", type=int, default=100)
    args = parser.parse_args()
    n_days = 365 * args.years

    results, summaries = {}, {}
    for k, (name, cfg) in enumerate(CONFIGS.items()):
        d_wet = P["d_wet"] * cfg.get("d_wet_factor", 1.0)
        si = stoch_rain.notebook_units(
            q_c=P["q_c"], b=P["b"], q_min=P["q_min"], evap=P["evap"], conv=P["conv"],
            rain_rate=P["rain_rate"], d_wet=d_wet, d_dry=P["d_dry"],
            dt_hours=cfg["dt_hours"], q_init=Q_INIT)
        n_steps = round(n_days * 86400 / si["dt"])
        print(f"{name}: {n_steps} steps of {si['dt']:.3g} s", flush=True)
        res = stoch_rain.integrate(
            si, n_steps, sqrt_noise=cfg["sqrt_noise"], wet_source=cfg["wet_source"],
            floor_restore=cfg["floor_restore"], seed=k + 1, progress=n_steps > 1e9)
        results[name] = res
        # Noise amplitude the scheme actually applies: the notebook's per-step
        # increment D dt (dt in s) equals a continuum D sqrt(dt_s) x sqrt(dt)
        scale = 1.0 if cfg["sqrt_noise"] else math.sqrt(si["dt"])
        theory = continuum_theory(d_wet * scale, P["d_dry"] * scale, P["rain_rate"],
                                  P["evap"] + P["conv"], cfg["wet_source"], P["b"])
        summaries[name] = dict(config=cfg, seed=k + 1, years=args.years,
                               effective_d_wet=d_wet * scale,
                               effective_d_dry=P["d_dry"] * scale,
                               theory=theory, measured=summarize(res, P["rain_rate"]))

    rows = ["accum_mean_mm", "accum_cutoff_mm", "wet_mean_min", "wet_cutoff_h",
            "dry_mean_min", "dry_cutoff_h"]
    print("\nmeasured, with the continuum first-passage prediction in brackets")
    print(f"{'':28s}" + "".join(f"{n:>24s}" for n in CONFIGS))
    for r in rows:
        cells = [f"{s['measured'][r]:.4g} [{s['theory'][r]:.4g}]" for s in summaries.values()]
        print(f"{r:28s}" + "".join(f"{c:>24s}" for c in cells))
    print("\nmeasured only")
    other = ["effective_d_wet", "effective_d_dry"]
    measured_only = [k for k in next(iter(summaries.values()))["measured"]
                     if k not in rows and not k.endswith("_fit")]
    for r in other + measured_only:
        vals = [s[r] if r in s else s["measured"][r] for s in summaries.values()]
        print(f"{r:28s}" + "".join(f"{v:>24.4g}" for v in vals))
    for fit in ["daily_fit", "accum_fit"]:
        for key in ["tau", "cutoff", "n_bins"]:
            vals = [s["measured"][fit][key] for s in summaries.values()]
            print(f"{fit + ' ' + key:28s}" + "".join(f"{v:>24.4g}" for v in vals))
    print(f"\nMN19 Fig. 2b daily fit at 0.6 s: tau_P = {MN19_DAILY_FIT['tau_P']}, "
          f"P_L = {MN19_DAILY_FIT['P_L']} mm")

    out = ROOT / "outputs" / f"check_against_mv19_{args.years}yr.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(summaries, indent=1, default=float))
    print(f"wrote {out}")
    fig_path = ROOT / "figs" / f"check_against_mv19_{args.years}yr.pdf"
    fig_path.parent.mkdir(exist_ok=True)
    make_figure(results, fig_path)


if __name__ == "__main__":
    main()
