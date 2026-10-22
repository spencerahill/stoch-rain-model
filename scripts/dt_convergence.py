"""How the on-off model's statistics depend on the time step, under MN19's scheme.

MN19 (Martinez-Villalobos and Neelin 2019) integrate their Fig. 2 parameters at
dt = 0.6 s, saying that step "is needed to generate the smallest accumulation
and daily precipitation values"; the notebook uses 1 min. This integrates the
paper's scheme (noise times sqrt(dt), E + C = 0 while raining, floor restore)
at dt = 60, 6, 0.6 and 0.06 s for 10 years each and prints the statistics
beside the continuum first-passage predictions (MN19 eqs. 10 and B2).

Usage: python -u scripts/dt_convergence.py [--years N]
Writes outputs/dt_convergence_<N>yr.json.
"""

import argparse
import json

import check_against_mv19 as chk
import stoch_rain

DTS = [60.0, 6.0, 0.6, 0.06]  # s
ROWS = ["accum_mean_mm", "accum_cutoff_mm", "wet_mean_min", "wet_cutoff_h", "dry_mean_min",
        "dry_cutoff_h", "events_per_wet_day_mean", "wet_day_frac", "daily_wet_mean_mm",
        "daily_PL_mom_mm", "daily_tauP_mom", "mean_precip_mm_day", "budget_floor_mm_day",
        "budget_residual_mm_day", "wall_s"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", type=int, default=10)
    args = parser.parse_args()
    P = chk.P
    theory = chk.continuum_theory(P["d_wet"], P["d_dry"], P["rain_rate"],
                                  P["evap"] + P["conv"], False, P["b"])
    out = {}
    for k, dt in enumerate(DTS):
        si = stoch_rain.notebook_units(
            q_c=P["q_c"], b=P["b"], q_min=P["q_min"], evap=P["evap"], conv=P["conv"],
            rain_rate=P["rain_rate"], d_wet=P["d_wet"], d_dry=P["d_dry"],
            dt_hours=dt / 3600, q_init=chk.Q_INIT)
        n_steps = round(365 * args.years * 86400 / si["dt"])
        print(f"dt = {dt} s: {n_steps} steps", flush=True)
        res = stoch_rain.integrate(si, n_steps, sqrt_noise=True, wet_source=False,
                                   floor_restore=True, seed=100 + k, progress=n_steps > 1e9)
        out[f"{dt} s"] = chk.summarize(res, P["rain_rate"])

    print(f"\n{'':26s}" + "".join(f"{name:>12s}" for name in out) + f"{'continuum':>12s}")
    for r in ROWS:
        vals = "".join(f"{s[r]:>12.4g}" for s in out.values())
        print(f"{r:26s}{vals}{theory[r] if r in theory else float('nan'):>12.4g}")
    # A barrier crossed between samples is overshot by about 0.58 sigma sqrt(dt)
    # (Siegmund's correction), so a discretization error from it falls by
    # sqrt(10) = 3.16 per tenfold cut in dt
    print("\ndeparture from the continuum, and its ratio to the next smaller dt's")
    for r in ["accum_mean_mm", "wet_mean_min", "dry_mean_min"]:
        dev = [s[r] - theory[r] for s in out.values()]
        ratios = [a / b for a, b in zip(dev[:-1], dev[1:])]
        print(f"{r:26s}" + "".join(f"{d:>12.4g}" for d in dev) + "   ratios "
              + " ".join(f"{x:.3g}" for x in ratios))
    path = chk.ROOT / "outputs" / f"dt_convergence_{args.years}yr.json"
    path.write_text(json.dumps(dict(theory=theory, runs=out), indent=1, default=float))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
