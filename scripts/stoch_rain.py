"""Single-column stochastic moisture model of Martinez-Villalobos and Neelin (2019).

A numba port of the ``StochRain`` class in ``scratch_neelin-stoch.ipynb``
(written 2023 for the India extreme-rainfall project), for the on-off
precipitation model of Martinez-Villalobos and Neelin (2019, J. Atmos. Sci. 76,
3611-3631, doi:10.1175/JAS-D-18-0343.1; MN19 below), their eqs. (5)-(7):

    dq/dt = E + C + D_E eta          dry regime
    dq/dt = E + C - R_0 + D_P eta    wet regime

with precipitation switching on when q rises above q_c and off when it falls
below q_c - b, and eta Gaussian white noise.

Three switches select between the notebook's choices and the paper's:

- ``sqrt_noise``: the notebook multiplies the noise term by dt; the
  Euler-Maruyama scheme MN19 use (their section 3b) multiplies it by sqrt(dt).
- ``wet_source``: the notebook keeps E + C in the wet regime; MN19 set
  E + C = 0 there (section 3a, below their eq. 6).
- ``floor_restore``: the notebook clamps q to 1 mm when it falls below; MN19
  restore q to its previous value instead (section 3a).

With all three at the notebook's setting, ``integrate`` reproduces the
notebook class bit for bit given the same noise (tests/test_port.py).
Internally the units are the notebook's: q in m of water, time in s.
"""

import time

import numba
import numpy as np

MM = 1e-3  # m per mm
HOUR = 3600.0  # s per hour

# Slots of the state vector carried from one chunk of steps to the next
Q, RAINING, CUR_ACCUM, CUR_LEN, CUR_DRY, DAY_STEP, DAY_RAIN, DAY_EVENTS = range(8)
Q_MAX, N_RAIN_STEPS, N_FLOOR, N_STEPS = range(8, 12)
# Running sums of the terms of the moisture budget, in m:
# q_end - q_init = source - rain + noise + floor
SRC_SUM, RAIN_SUM, NOISE_SUM, FLOOR_SUM = range(12, 16)
N_STATE = 16

# CWV histogram: log10(q / mm) bins sampled once per simulated minute
HIST_LOG10_EDGES = np.arange(0.0, 4.5001, 0.025)

PAPER_FIG2 = dict(  # MN19 Fig. 2b caption, on-off precipitation
    q_c=65.0,  # mm
    b=0.2,  # mm
    q_min=1.0,  # mm
    evap=0.1,  # mm/h
    conv=0.2,  # mm/h
    rain_rate=9.0,  # R_0, mm/h
    d_wet=17.0,  # D_P, mm h^-1/2
    d_dry=3.0,  # D_E, mm h^-1/2
)


@numba.njit(cache=True)
def _integrate_chunk(xi, state, prm, out_accum, out_len, out_dry, out_day_rain,
                     out_day_events, q_hist, q_out):
    """Advance the model one step per element of ``xi``, updating ``state``.

    Returns the number of completed rain events, dry spells and days written
    to the output buffers.
    """
    q_start, q_end, q_min = prm[0], prm[1], prm[2]
    amp_wet, amp_dry, dt = prm[3], prm[4], prm[5]
    evap, conv, rain_rate = prm[6], prm[7], prm[8]
    sqrt_noise, wet_source, floor_restore = prm[9] > 0.5, prm[10] > 0.5, prm[11] > 0.5
    steps_per_day, sample_every = int(prm[12]), int(prm[13])
    hist_lo, hist_dlog = prm[14], prm[15]
    sqrt_dt = np.sqrt(dt)
    n_bins = q_hist.size
    store_q = q_out.size > 0

    q = state[Q]
    raining_prev = state[RAINING] > 0.5
    cur_accum, cur_len, cur_dry = state[CUR_ACCUM], int(state[CUR_LEN]), int(state[CUR_DRY])
    day_step, day_rain, day_events = int(state[DAY_STEP]), state[DAY_RAIN], int(state[DAY_EVENTS])
    q_max, n_rain_steps, n_floor = state[Q_MAX], int(state[N_RAIN_STEPS]), int(state[N_FLOOR])
    n_steps = int(state[N_STEPS])
    src_sum, rain_sum = state[SRC_SUM], state[RAIN_SUM]
    noise_sum, floor_sum = state[NOISE_SUM], state[FLOOR_SUM]
    noise_mult = sqrt_dt if sqrt_noise else dt
    n_ev = 0
    n_dry = 0
    n_days = 0

    for i in range(xi.size):
        # Regime from the previous step's q, with hysteresis, as in the notebook
        thresh = q_end if raining_prev else q_start
        raining = q > thresh
        if raining:
            rain_n = rain_rate * dt
            if not raining_prev:
                out_dry[n_dry] = cur_dry
                n_dry += 1
                cur_accum = rain_n
                cur_len = 1
                day_events += 1
            else:
                cur_len += 1
                cur_accum += rain_n
            n_rain_steps += 1
            amp = amp_wet
            rr = rain_rate
        else:
            rain_n = 0.0
            if raining_prev:
                out_accum[n_ev] = cur_accum
                out_len[n_ev] = cur_len
                n_ev += 1
                cur_dry = 1
            else:
                cur_dry += 1
            amp = amp_dry
            rr = 0.0

        # The notebook's expression, kept in its order so the port is bitwise
        if sqrt_noise:
            if raining and not wet_source:
                q_new = q - rr * dt + xi[i] * amp * sqrt_dt
            else:
                q_new = q + (-rr + evap + conv) * dt + xi[i] * amp * sqrt_dt
        else:
            if raining and not wet_source:
                q_new = q + (xi[i] * amp - rr) * dt
            else:
                q_new = q + (xi[i] * amp - rr + evap + conv) * dt
        rain_sum += rr * dt
        noise_sum += xi[i] * amp * noise_mult
        if not (raining and not wet_source):
            src_sum += (evap + conv) * dt
        if q_new < q_min:
            n_floor += 1
            q_floored = q if floor_restore else q_min
            floor_sum += q_floored - q_new
            q_new = q_floored
        q = q_new
        raining_prev = raining

        if q > q_max:
            q_max = q
        if store_q:
            q_out[i] = q
        n_steps += 1
        if n_steps % sample_every == 0:
            k = int((np.log10(q / 1e-3) - hist_lo) / hist_dlog)
            if k < 0:
                k = 0
            if k >= n_bins:
                k = n_bins - 1
            q_hist[k] += 1

        day_rain += rain_n
        day_step += 1
        if day_step == steps_per_day:
            out_day_rain[n_days] = day_rain
            out_day_events[n_days] = day_events
            n_days += 1
            day_step = 0
            day_rain = 0.0
            day_events = 0

    state[Q] = q
    state[RAINING] = 1.0 if raining_prev else 0.0
    state[CUR_ACCUM], state[CUR_LEN], state[CUR_DRY] = cur_accum, cur_len, cur_dry
    state[DAY_STEP], state[DAY_RAIN], state[DAY_EVENTS] = day_step, day_rain, day_events
    state[Q_MAX], state[N_RAIN_STEPS], state[N_FLOOR] = q_max, n_rain_steps, n_floor
    state[N_STEPS] = n_steps
    state[SRC_SUM], state[RAIN_SUM] = src_sum, rain_sum
    state[NOISE_SUM], state[FLOOR_SUM] = noise_sum, floor_sum
    return n_ev, n_dry, n_days


def notebook_units(q_c, b, q_min, evap, conv, rain_rate, d_wet, d_dry, dt_hours, q_init):
    """Convert mm, mm/h, mm h^-1/2 and hours to m and s exactly as StochRain does."""
    return dict(
        q_start=q_c * 1e-3,
        q_end=(q_c - b) * 1e-3,
        q_min=q_min * 1e-3,
        q_init=q_init * 1e-3,
        amp_wet=d_wet * 1e-3 * 3600.0**-0.5,
        amp_dry=d_dry * 1e-3 * 3600.0**-0.5,
        dt=dt_hours * 3600.0,
        evap=evap * 1e-3 / 3600.0,
        conv=conv * 1e-3 / 3600.0,
        rain_rate=rain_rate * 1e-3 / 3600.0,
    )


def integrate(si, n_steps, *, sqrt_noise, wet_source, floor_restore, seed=None,
              noise=None, chunk=2**22, store_q=False, progress=False):
    """Integrate for ``n_steps`` steps after the initial state, step 0.

    ``si`` holds parameters in the notebook's SI units (``notebook_units``).
    The noise is either drawn from ``numpy.random.default_rng(seed)`` in chunks
    or, for testing against the notebook, passed as ``noise`` (length n_steps,
    the notebook's noise array without its unused element 0).

    Rain events and dry spells still in progress at the end are dropped, as in
    the notebook, as is a final partial day.
    """
    dt = si["dt"]
    steps_per_day = round(86400.0 / dt)
    assert abs(steps_per_day * dt - 86400.0) < 1e-6 * dt, "dt must divide a day"
    sample_every = round(60.0 / dt)
    assert abs(sample_every * dt - 60.0) < 1e-6 * dt, "dt must divide a minute"
    prm = np.array([
        si["q_start"], si["q_end"], si["q_min"], si["amp_wet"], si["amp_dry"], dt,
        si["evap"], si["conv"], si["rain_rate"],
        float(sqrt_noise), float(wet_source), float(floor_restore),
        steps_per_day, sample_every, HIST_LOG10_EDGES[0],
        HIST_LOG10_EDGES[1] - HIST_LOG10_EDGES[0],
    ])

    # Step 0, as in StochRain._setup
    q0 = si["q_init"]
    raining0 = q0 > si["q_start"]
    state = np.zeros(N_STATE)
    state[Q] = q0
    state[RAINING] = float(raining0)
    state[DAY_STEP] = 1
    state[DAY_RAIN] = si["rain_rate"] * dt if raining0 else 0.0
    state[Q_MAX] = q0
    state[N_STEPS] = 0

    if noise is not None:
        assert noise.size == n_steps
        chunk = n_steps
    rng = np.random.default_rng(seed)
    q_hist = np.zeros(HIST_LOG10_EDGES.size - 1, dtype=np.int64)
    q_series = np.empty(n_steps) if store_q else None
    n_buf = chunk // 2 + 2
    buf_accum, buf_len, buf_dry = np.empty(n_buf), np.empty(n_buf, np.int64), np.empty(n_buf, np.int64)
    buf_day_rain = np.empty(chunk // steps_per_day + 2)
    buf_day_events = np.empty(chunk // steps_per_day + 2, np.int64)
    accum, length, dry, day_rain, day_events = [], [], [], [], []

    t0 = time.perf_counter()
    done = 0
    n_chunks = -(-n_steps // chunk)
    for k in range(n_chunks):
        n = min(chunk, n_steps - done)
        xi = noise if noise is not None else rng.standard_normal(n)
        q_out = q_series[done:done + n] if store_q else np.empty(0)
        n_ev, n_dry, n_days = _integrate_chunk(
            xi, state, prm, buf_accum, buf_len, buf_dry, buf_day_rain, buf_day_events,
            q_hist, q_out)
        accum.append(buf_accum[:n_ev].copy())
        length.append(buf_len[:n_ev].copy())
        dry.append(buf_dry[:n_dry].copy())
        day_rain.append(buf_day_rain[:n_days].copy())
        day_events.append(buf_day_events[:n_days].copy())
        done += n
        if progress and (k % max(1, n_chunks // 20) == 0 or k == n_chunks - 1):
            print(f"    chunk {k + 1} of {n_chunks}, elapsed {time.perf_counter() - t0:.0f} s",
                  flush=True)

    return dict(
        accum_m=np.concatenate(accum),
        event_steps=np.concatenate(length),
        dry_steps=np.concatenate(dry),
        day_rain_m=np.concatenate(day_rain),
        day_events=np.concatenate(day_events),
        q_hist=q_hist,
        q_max_m=state[Q_MAX],
        n_rain_steps=int(state[N_RAIN_STEPS]),
        n_floor=int(state[N_FLOOR]),
        n_steps=int(state[N_STEPS]),
        budget_m=dict(q_change=state[Q] - q0, source=state[SRC_SUM], rain=state[RAIN_SUM],
                      noise=state[NOISE_SUM], floor=state[FLOOR_SUM]),
        steps_per_day=steps_per_day,
        dt=dt,
        q_series=q_series,
        wall_s=time.perf_counter() - t0,
    )
