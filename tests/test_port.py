"""The numba port in scripts/stoch_rain.py against the notebook's own StochRain class."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import stoch_rain  # noqa: E402


def notebook_namespace():
    """Execute the notebook cell defining StochRain, read from the notebook itself."""
    nb = json.loads((ROOT / "scratch_neelin-stoch.ipynb").read_text())
    sources = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    (src,) = [s for s in sources if "class StochRain" in s]
    ns = {"np": np}
    exec(src, ns)
    return ns


NOTEBOOK_DEFAULTS = dict(q_c=65.0, b=0.2, q_min=1.0, evap=0.1, conv=0.2, rain_rate=9.0,
                         d_wet=17.0, d_dry=3.0, q_init=64.9)


def test_port_reproduces_notebook_bitwise():
    ns = notebook_namespace()
    np.random.seed(20231)
    n_iter = 300_000
    nb = ns["StochRain"]()
    nb.run(n_iter)

    si = stoch_rain.notebook_units(**NOTEBOOK_DEFAULTS, dt_hours=ns["DT_MV19"])
    for key, attr in [("q_start", "q_thresh_start"), ("q_end", "q_thresh_end"),
                      ("q_min", "q_min"), ("q_init", "q_init"), ("amp_wet", "noise_amp_wet"),
                      ("amp_dry", "noise_amp_dry"), ("dt", "dt"), ("evap", "evap_clim"),
                      ("conv", "moist_flux_conv_clim"), ("rain_rate", "rain_rate")]:
        assert si[key] == getattr(nb, attr), key

    res = stoch_rain.integrate(si, n_iter - 1, sqrt_noise=False, wet_source=True,
                               floor_restore=False, noise=nb._moist_conv_stoch_term[1:],
                               store_q=True)
    assert len(nb.rain_event_accums) > 1000  # the comparison is not vacuous
    assert np.array_equal(res["q_series"], nb.q[1:])
    assert np.array_equal(res["accum_m"], np.array(nb.rain_event_accums))
    assert np.array_equal(res["event_steps"], np.array(nb.rain_event_lengths))
    assert np.array_equal(res["dry_steps"], np.array(nb.dry_event_lengths))
    assert res["n_rain_steps"] == int(np.nansum(nb.is_raining[1:]))

    _, day_rain, _ = nb.daily_avg()
    assert res["day_rain_m"].size == day_rain.size
    # The notebook sums each day with np.sum (pairwise), the port sequentially
    rel = np.abs(res["day_rain_m"] - day_rain) / np.maximum(day_rain, 1e-300)
    assert rel.max() < 1e-13, rel.max()


@pytest.mark.parametrize("wet_source, expected_drop_mm", [(True, 0.145), (False, 0.15)])
def test_wet_regime_source_switch(wet_source, expected_drop_mm):
    # No noise, raining from the start: one 1-min step removes R_0 dt = 0.15 mm,
    # less (E + C) dt = 0.005 mm where the scheme keeps E + C while raining
    si = stoch_rain.notebook_units(q_c=65.0, b=0.2, q_min=1.0, evap=0.1, conv=0.2,
                                   rain_rate=9.0, d_wet=0.0, d_dry=0.0,
                                   dt_hours=1 / 60, q_init=66.0)
    res = stoch_rain.integrate(si, 1, sqrt_noise=True, wet_source=wet_source,
                               floor_restore=True, seed=0, store_q=True)
    assert res["q_series"][0] / 1e-3 == pytest.approx(66.0 - expected_drop_mm, abs=1e-12)


@pytest.mark.parametrize("floor_restore, expected_mm", [(False, 1.0), (True, 1.05)])
def test_floor_switch(floor_restore, expected_mm):
    # No noise, drying at 10 mm/h: one 1-min step would take q from 1.05 to
    # 0.883 mm, below the 1 mm floor
    si = stoch_rain.notebook_units(q_c=65.0, b=0.2, q_min=1.0, evap=0.0, conv=-10.0,
                                   rain_rate=9.0, d_wet=0.0, d_dry=0.0,
                                   dt_hours=1 / 60, q_init=1.05)
    res = stoch_rain.integrate(si, 1, sqrt_noise=True, wet_source=True,
                               floor_restore=floor_restore, seed=0, store_q=True)
    assert res["q_series"][0] / 1e-3 == pytest.approx(expected_mm, abs=1e-12)
    assert res["n_floor"] == 1


@pytest.mark.parametrize("sqrt_noise, wet_source, floor_restore",
                         [(False, True, False), (True, False, True)])
def test_moisture_budget_closes(sqrt_noise, wet_source, floor_restore):
    # q_end - q_init = source - rain + noise + floor, to rounding
    si = stoch_rain.notebook_units(**NOTEBOOK_DEFAULTS, dt_hours=1 / 60)
    res = stoch_rain.integrate(si, 200_000, sqrt_noise=sqrt_noise, wet_source=wet_source,
                               floor_restore=floor_restore, seed=3)
    b = res["budget_m"]
    for term in ["source", "rain", "noise"]:
        assert abs(b[term]) > 0, term
    if not sqrt_noise:
        assert b["floor"] > 0  # the notebook's noise pushes q through the floor often
    resid = b["q_change"] - (b["source"] - b["rain"] + b["noise"] + b["floor"])
    assert abs(resid) < 1e-9 * b["rain"], resid


def hourly_increment_variance(dt_hours, sqrt_noise, d_dry=3.0, hours=2000):
    """Variance of q's change over one hour, in mm^2, for pure diffusion."""
    si = stoch_rain.notebook_units(q_c=1e9, b=0.2, q_min=-1e12, evap=0.0, conv=0.0,
                                   rain_rate=9.0, d_wet=17.0, d_dry=d_dry,
                                   dt_hours=dt_hours, q_init=1e6)
    steps_per_hour = round(1 / dt_hours)
    res = stoch_rain.integrate(si, hours * steps_per_hour, sqrt_noise=sqrt_noise,
                               wet_source=True, floor_restore=False, seed=1, store_q=True)
    q_hourly = res["q_series"][steps_per_hour - 1::steps_per_hour] / 1e-3
    return np.var(np.diff(q_hourly))


@pytest.mark.parametrize("dt_hours", [1 / 60, 1 / 600])
def test_sqrt_dt_noise_gives_D_squared_per_hour(dt_hours):
    # Euler-Maruyama: q changes by D_E^2 = 9 mm^2 per hour in variance, at any dt
    var = hourly_increment_variance(dt_hours, sqrt_noise=True)
    assert abs(var / 9.0 - 1) < 0.1, var


def test_notebook_noise_variance_scales_with_dt():
    # The notebook's noise term is multiplied by dt, so the variance per hour
    # is D_E^2 times dt in seconds: 540 mm^2 at dt = 60 s, 54 mm^2 at 6 s
    var_60 = hourly_increment_variance(1 / 60, sqrt_noise=False)
    var_6 = hourly_increment_variance(1 / 600, sqrt_noise=False)
    assert abs(var_60 / 540.0 - 1) < 0.1, var_60
    assert abs(var_6 / 54.0 - 1) < 0.1, var_6
