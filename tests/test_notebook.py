"""The model function in stoch_rain_model.ipynb.

The function is read out of the notebook itself, so these tests cover the code a
student runs. The independent reference is the numba port in scripts/stoch_rain.py,
which tests/test_port.py checks bit for bit against the 2023 notebook's class, run
here with its switches set to MN19's scheme.
"""

import json
import math
import sys
import time
from pathlib import Path

import numba
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import stoch_rain  # noqa: E402

PARAMS = dict(q_c=65.0, b=0.2, q_min=1.0, E=0.1, C=0.2, R_0=9.0, D_E=3.0, D_P=17.0, q_init=64.9)


def notebook_namespace():
    """Execute the notebook cells that only define functions and constants."""
    nb = json.loads((ROOT / "stoch_rain_model.ipynb").read_text())
    sources = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    ns = {"np": np, "math": math, "time": time}
    for start in ["def simulate(", "BUDGET_TERMS", "KAPPA", "MN19_FIG2B_DAILY_FIT"]:
        (src,) = [s for s in sources if s.startswith(start)]
        exec(src, ns)
    return ns


NB = notebook_namespace()
SIMULATE, SIMULATE_FAST = NB["simulate"], numba.njit(NB["simulate"])


@pytest.mark.parametrize("dt_seconds, n_days", [(60.0, 20), (0.6, 1)])
def test_pure_python_and_numba_identical(dt_seconds, n_days):
    n_save = round(n_days * 86400 / dt_seconds)
    out_py = SIMULATE(n_days, dt_seconds / 3600, rng=np.random.default_rng(5), n_save=n_save,
                      **PARAMS)
    out_nb = SIMULATE_FAST(n_days, dt_seconds / 3600, rng=np.random.default_rng(5),
                           n_save=n_save, **PARAMS)
    assert len(out_py[2]) > 10  # rain events occurred
    for a, b in zip(out_py, out_nb):
        assert np.array_equal(np.asarray(a), np.asarray(b))


@pytest.mark.parametrize("dt_seconds, n_days", [(60.0, 30), (0.6, 2)])
def test_matches_port_in_mn19_mode(dt_seconds, n_days):
    # Same seed, so both draw the same standard normals in the same order
    dt_h = dt_seconds / 3600
    n_steps = round(n_days * 86400 / dt_seconds)
    (_, _, accum, wet_steps, dry_steps, _, q_saved, rain_saved, _) = SIMULATE_FAST(
        n_days, dt_h, rng=np.random.default_rng(11), n_save=n_steps, **PARAMS)
    si = stoch_rain.notebook_units(q_c=PARAMS["q_c"], b=PARAMS["b"], q_min=PARAMS["q_min"],
                                   evap=PARAMS["E"], conv=PARAMS["C"], rain_rate=PARAMS["R_0"],
                                   d_wet=PARAMS["D_P"], d_dry=PARAMS["D_E"], dt_hours=dt_h,
                                   q_init=PARAMS["q_init"])
    port = stoch_rain.integrate(si, n_steps, sqrt_noise=True, wet_source=False,
                                floor_restore=True, seed=11, store_q=True)
    assert len(accum) > 10
    np.testing.assert_allclose(q_saved, port["q_series"] / stoch_rain.MM, rtol=0, atol=1e-9)
    assert np.count_nonzero(rain_saved) == port["n_rain_steps"]
    np.testing.assert_allclose(np.array(accum), port["accum_m"] / stoch_rain.MM, rtol=1e-12)
    assert np.array_equal(np.array(wet_steps), port["event_steps"])
    assert np.array_equal(np.array(dry_steps), port["dry_steps"])


def test_moisture_budget_closes():
    # q_end - q_init = source - rain + noise + boundary, to rounding; the boundary is
    # raised to 55 mm so that every term is exercised within one year
    out = SIMULATE_FAST(365, 1 / 60, rng=np.random.default_rng(3), n_save=0,
                        **dict(PARAMS, q_min=55.0))
    change, source, rain, noise, boundary = out[8]
    for term in [source, rain, noise, boundary]:
        assert abs(term) > 0
    assert abs(change - (source - rain + noise + boundary)) < 1e-9 * rain
    assert rain == pytest.approx(out[0].sum(), rel=1e-12)


@pytest.mark.parametrize("dt_seconds", [60.0, 6.0])
def test_noise_variance_is_D_squared_per_hour(dt_seconds):
    # Pure diffusion (no rain, no drift, boundary far below): the variance of q
    # after one day is D_E^2 x 24 h = 216 mm^2 at any time step
    p = dict(PARAMS, q_c=1e9, q_min=0.0, E=0.0, C=0.0, q_init=1000.0)
    ends = [SIMULATE_FAST(1, dt_seconds / 3600, rng=np.random.default_rng(s), n_save=0, **p)[8][0]
            for s in range(1500)]
    assert np.var(ends) / 216.0 == pytest.approx(1, abs=0.1)


def test_wet_regime_has_no_E_plus_C():
    # No noise, raining from the start: one 1-min step removes exactly R_0 dt = 0.15 mm
    p = dict(PARAMS, D_E=0.0, D_P=0.0, q_init=66.0)
    out = SIMULATE_FAST(1, 1 / 60, rng=np.random.default_rng(0), n_save=1, **p)
    assert out[6][0] == pytest.approx(66.0 - 0.15, abs=1e-12)


def test_lower_boundary_keeps_previous_value():
    # No noise, drying at 10 mm/h: the first 1-min step would take q from 1.05 to
    # 0.883 mm, below the 1 mm boundary, so q stays at 1.05 mm all day, and each of
    # the 1440 discarded steps adds 10/60 mm to the boundary term of the budget
    p = dict(PARAMS, D_E=0.0, D_P=0.0, E=0.0, C=-10.0, q_init=1.05)
    out = SIMULATE_FAST(1, 1 / 60, rng=np.random.default_rng(0), n_save=1440, **p)
    assert np.all(out[6] == 1.05)
    assert out[8][4] == pytest.approx(1440 * 10 / 60, rel=1e-9)


def full_run(n_days, dt_seconds, seed, **changes):
    n_save = round(n_days * 86400 / dt_seconds)
    out = SIMULATE_FAST(n_days, dt_seconds / 3600, rng=np.random.default_rng(seed),
                        n_save=n_save, **dict(PARAMS, **changes))
    return out, round(86400 / dt_seconds)


def test_events_in_day_counts_starts_and_continuations():
    # Rebuilt from the saved precipitation series: each step that starts an event,
    # plus each day that begins while an event is under way
    out, steps_per_day = full_run(3650, 60.0, seed=7)
    rain = out[7] > 0
    starts = rain & ~np.r_[False, rain[:-1]]
    day_begins_raining = np.zeros(rain.size, dtype=bool)
    day_begins_raining[::steps_per_day] = rain[::steps_per_day]
    assert (day_begins_raining & ~starts).sum() > 10  # events spanning midnight occur
    expected = np.add.reduceat((starts | day_begins_raining).astype(int),
                               np.arange(0, rain.size, steps_per_day))
    assert np.array_equal(out[1], expected)
    assert np.array_equal(out[0] > 0, out[1] > 0)


def test_q_counts_is_time_in_1mm_bins():
    out, _ = full_run(365, 60.0, seed=8)
    q = out[6]
    expected, _ = np.histogram(np.clip(q, 0, 399.5), bins=np.arange(401))
    assert np.array_equal(out[5], expected)


@pytest.mark.parametrize("q_init", [64.9, 66.0])
def test_run_drops_the_spell_under_way_at_the_start(q_init):
    params = dict(PARAMS, q_init=q_init)
    res = NB["run"](SIMULATE_FAST, params, n_days=30, dt_seconds=60.0, seed=9)
    out = SIMULATE_FAST(30, 1 / 60, rng=np.random.default_rng(9), n_save=0, **params)
    accum, wet, dry = np.array(out[2]), np.array(out[3]), np.array(out[4])
    if q_init > PARAMS["q_c"]:
        accum, wet = accum[1:], wet[1:]
    else:
        dry = dry[1:]
    assert np.array_equal(res["accum"], accum)
    assert res["wet_h"].size == wet.size and res["dry_h"].size == dry.size
    np.testing.assert_allclose(res["wet_h"], wet / 60, rtol=1e-12)
    np.testing.assert_allclose(res["dry_h"], dry / 60, rtol=1e-12)


@pytest.mark.parametrize("mean, cutoff", [(0.2, 64.22), (0.35, 64.22), (40 / 60, 200.0)])
def test_first_passage_pdf_is_normalized_with_its_mean_and_cutoff(mean, cutoff):
    # MN19 eq. (9): unit area, mean x-bar, and 2 var / mean = x_L
    x = np.logspace(-9, 5, 400_001)
    p = NB["first_passage_pdf"](x, mean, cutoff)
    lnx = np.log(x)
    m0 = np.trapezoid(p * x, lnx)
    m1 = np.trapezoid(p * x**2, lnx)
    m2 = np.trapezoid(p * x**3, lnx)
    assert m0 == pytest.approx(1, rel=1e-6)
    assert m1 == pytest.approx(mean, rel=1e-6)
    assert 2 * (m2 - m1**2) / m1 == pytest.approx(cutoff, rel=1e-6)


def test_cutoff_from_moments_on_inverse_gaussian_sample():
    # numpy's Wald distribution is the inverse Gaussian with mean mu and shape lam,
    # whose variance is mu^3 / lam, so its cutoff is 2 mu^2 / lam (MN19 eq. C1)
    mu, cutoff = 1.0, 20.0
    x = np.random.default_rng(2).wald(mu, 2 * mu**2 / cutoff, 4_000_000)
    assert NB["cutoff_from_moments"](x) == pytest.approx(cutoff, rel=0.05)


def test_gamma_pdf_is_normalized():
    # MN19 eq. (2): unit area and mean (1 - tau) P_L
    x = np.logspace(-12, 4, 400_001)
    p = NB["gamma_pdf"](x, 0.5, 29.7)
    assert np.trapezoid(p * x, np.log(x)) == pytest.approx(1, rel=1e-5)
    assert np.trapezoid(p * x**2, np.log(x)) == pytest.approx(0.5 * 29.7, rel=1e-6)


def gamma_sample(tau, x_L, quantum, n=2_000_000, seed=0):
    x = np.random.default_rng(seed).gamma(1 - tau, x_L, n)
    return np.rint(x / quantum) * quantum


def test_fit_moments_recovers_gamma_parameters():
    tau, x_L = NB["fit_moments"](gamma_sample(0.5, 30.0, 1e-4))
    assert tau == pytest.approx(0.5, abs=0.01)
    assert x_L == pytest.approx(30.0, rel=0.01)


def test_fit_appendix_a_recovers_gamma_parameters():
    tau, x_L = NB["fit_appendix_a"](gamma_sample(0.5, 30.0, 1e-3), 1e-3)
    assert tau == pytest.approx(0.5, abs=0.02)
    assert x_L == pytest.approx(30.0, rel=0.03)


def test_log_binned_pdf_is_exact_on_single_quanta():
    # For bins one quantum wide, the PDF is the frequency of that multiple per unit x
    quantum = 0.15
    k = np.random.default_rng(1).geometric(0.02, 100_000)
    centers, pdf = NB["log_binned_pdf"](k * quantum, quantum)
    # At 10 bins per decade the edges begin 0.5, 1.5, 2.5, 3.5, 5.5, 6.5, 7.5, 10.5
    # quanta, so 1, 2, 3, 6 and 7 quanta have bins of their own
    for j in [1, 2, 3, 6, 7]:
        i = np.flatnonzero(np.isclose(centers, j * quantum))
        assert i.size == 1
        assert pdf[i[0]] == pytest.approx((k == j).mean() / quantum, rel=1e-12)
    assert centers.size > 20  # wider bins were kept too


def test_overshoot_corrected_mean_accumulation():
    # At dt = 6 s the mean accumulation exceeds b by about kappa (D_P + D_E) sqrt(dt)
    res = NB["run"](SIMULATE_FAST, PARAMS, n_days=3650, dt_seconds=6.0, seed=10)
    predicted = NB["with_overshoot"](res)["accum_mean"]
    assert predicted - PARAMS["b"] > 0.4  # the correction is most of the mean
    assert res["accum"].mean() == pytest.approx(predicted, rel=0.04)
