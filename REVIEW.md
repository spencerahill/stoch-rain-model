# Review of the 2023 notebook against Martinez-Villalobos and Neelin (2019)

This document reviews a notebook Spencer wrote in 2023 for the India extreme-rainfall project. The notebook implements the single-column stochastic moisture model of Martinez-Villalobos and Neelin (2019, *J. Atmos. Sci.* 76, 3611–3631, doi:10.1175/JAS-D-18-0343.1). "MN19" below means that paper. The review asks four things: whether the code implements the paper's model correctly, whether it has bugs, what other problems it has, and whether it still runs. Review carried out 2026-10-22.

## Files

- `scratch_neelin-stoch.ipynb`: the 2023 notebook, copied byte for byte from `projects/old/finished/india-extreme-rain-var/notebooks/`, with its 2023 outputs. Unmodified.
- `martinez-villalobos-neelin-2019-jas.pdf`: the paper, from Zotero. Untracked by git (AMS copyright).
- `scripts/stoch_rain.py`: a numba port of the notebook's `StochRain` class. Three switches select the notebook's choice or the paper's at each point where they differ (listed below).
- `tests/test_port.py`: 10 tests. The central one runs the notebook's own `StochRain` class, read out of the notebook file, and requires the port to reproduce its column-water-vapor series, event lists and daily totals given the same random numbers. The port reproduces all three bit for bit. Nine planted bugs were each caught by at least one test.
- `scripts/check_against_mv19.py`: 100-year integrations under the notebook's scheme and the paper's, compared with MN19's analytic results. Output in `outputs/check_against_mv19_100yr.log` and `figs/check_against_mv19_100yr.pdf`.
- `scripts/dt_convergence.py`: the paper's scheme at four time steps. Output in `outputs/dt_convergence_10yr.log`.
- `scripts/el_nino_ratio.py`: the notebook's "El Niño" experiment redone with and without the noise fix. Output in `outputs/el_nino_ratio_100yr.log` and `figs/el_nino_ratio_100yr.pdf`.

Run any of them with `~/.pixi/bin/pixi run --manifest-path <this dir>/pixi.toml -e dev python -u scripts/<name>.py`, and the tests with `pixi run -e dev test`.

## The model

Column water vapor *q* evolves as (MN19 eqs. 5–7)

- dry regime: dq/dt = E + C̄ + D_E η
- wet regime: dq/dt = E + C̄ − R₀ + D_P η

where η is Gaussian white noise. Precipitation, at the constant rate R₀, switches on when *q* rises above q_c and off when it falls below q_c − b. MN19 set E + C̄ = 0 in the wet regime for analytical simplicity. The notebook takes its parameter values from MN19's Fig. 2b caption: q_c = 65 mm, b = 0.2 mm, E = 0.1 mm/h, C̄ = 0.2 mm/h, R₀ = 9 mm/h, D_P = 17 mm h^-1/2, D_E = 3 mm h^-1/2, with a floor at q = 1 mm. Those values all match the caption.

For this model MN19 derive the accumulation distribution analytically (their eqs. 9–10). The mean accumulation per event equals b. The accumulation cutoff scale is s_L = 2 D_P²/R₀, which is 64.2 mm for these parameters.

## Findings

### 1. The noise term is multiplied by dt where it should be multiplied by √dt (a bug)

In `StochRain._iter_cwv` the update is

```python
self.q[n] = self.q[n - 1] + (
    self._moist_conv_stoch_term[n] * noise_amp - rain_rate + self.evap_clim + self.moist_flux_conv_clim
) * self.dt
```

so the Gaussian draw is multiplied by the noise amplitude times dt. The Euler–Maruyama scheme MN19 use (their section 3b) multiplies it by the amplitude times √dt. The docstring gives the amplitudes in mm h^-1/2, the paper's units. With those units the product of amplitude and dt has units of mm h^1/2, which is not a length, so this is a bug.

The size of the error depends on the unit of dt. The notebook works in seconds with dt = 60 s, so each step adds noise with a standard deviation of D × 60 s. Euler–Maruyama would add D × √(60 s). The notebook's scaling is therefore equivalent to running MN19's model with both noise amplitudes multiplied by √60 = 7.75: D_P becomes 131.7 mm h^-1/2 and D_E becomes 23.2 mm h^-1/2. Two tests in `tests/test_port.py` check this directly, using pure diffusion with D_E = 3 mm h^-1/2:

- **Euler–Maruyama:** the variance of q's change over one hour must be within 10% of D_E² = 9 mm² at both dt = 60 s and dt = 6 s.
- **The notebook's scheme:** the same variance must be within 10% of D_E² times dt in seconds, which is 540 mm² at 60 s and 54 mm² at 6 s.

Both tests pass.

The 100-year integrations confirm that the notebook behaves as MN19's model would with the inflated amplitudes. The accumulation cutoff scale, estimated from the moments as 2σ_s²/s̄ (MN19 eq. 10), is 3938 mm. The first-passage theory evaluated at the inflated amplitudes predicts 4124 mm. With the paper's amplitudes it is 64.2 mm.

The fix is one line:

```python
self.q[n] = self.q[n - 1] + (
    - rain_rate + self.evap_clim + self.moist_flux_conv_clim
) * self.dt + self._moist_conv_stoch_term[n] * noise_amp * np.sqrt(self.dt)
```

### 2. What the bug did to the 2023 results

All numbers in this section come from `outputs/check_against_mv19_100yr.log`. Each compares the notebook's scheme with the same scheme after the √dt fix, both at dt = 1 min with the Fig. 2b parameters, over 100 years.

- **Column water vapor:** the maximum is 8583 mm, and *q* exceeds 100 mm in 34% of minutes. After the fix the maximum is 255 mm, and *q* exceeds 100 mm in 0.4% of minutes. The 2023 notebook printed a maximum of 8.14 m (in its own SI units) for its 120-season run.
- **How often it rains:** it rains 36% of the time, and 98% of days are wet. After the fix the figures are 3.4% of the time and 56% of days.
- **Mean precipitation:** 77.4 mm/day, against 7.3 mm/day after the fix. The moisture budget shows where the extra rain comes from. E + C̄ supplies 7.2 mm/day. The floor at q = 1 mm supplies 69.8 mm/day, because the inflated noise pushes *q* below 1 mm about 1.2 million times in 100 years and every clamp back to 1 mm adds water. The budget closes to within 2 × 10^-9 mm/day.
- **All-day rain:** 21% of wet days have rain in all 1440 minutes, giving the cap of 216 mm that appears as the spike at the top of the daily-precipitation PDF. After the fix this fraction is 5 × 10^-5.
- **Daily-precipitation distribution:** MN19 fit their eq. (2), p ∝ P^-τ_P exp(−P/P_L), by regressing the log of the binned PDF on log P and P (their appendix A). Applied to the notebook's daily totals, that regression finds no cutoff (the fitted exponential term has the wrong sign). After the fix it gives a cutoff P_L = 34 mm.
- **The El Niño experiment (notebook cells 18–22):** this experiment multiplies D_P by 1.5 and plots the ratio of the daily-precipitation PDFs. Under the notebook's scheme, El Niño lowers the probability of wet days with 30–130 mm to about 0.74 times the control. That matches the 2023 cell-21 figure. With the noise fixed, the ratio rises above 1 from about 80 mm upward. The probability that a wet day exceeds 100 mm goes from 0.0092 to 0.0221, a factor of 2.4 (`outputs/el_nino_ratio_100yr.log`, `figs/el_nino_ratio_100yr.pdf`). So for heavy days, the 2023 conclusion has the opposite sign from the corrected model's. A rise at the heavy end is what MN19's framework predicts: P_L scales with s_L, which scales with D_P².

### 3. The time step: 1 min where MN19's Fig. 2 used 0.6 s

The constant `DT_MV19 = 1 / 60.` (hours) is named for the paper. However, MN19's Fig. 2, the source of the parameter values, was integrated at 0.6 s. MN19 write that this step "is needed to generate the smallest accumulation and daily precipitation values" for the on–off model. They do use 1 min elsewhere: in the integrations behind their Figs. 4 and 7, where b = 1 mm, and for the hourly Miami comparison of Fig. 6d, where b = 0.2 mm. So 1 min is defensible. With b = 0.2 mm, it changes the event-scale statistics by factors of up to ten, and the moment statistics of daily totals by under 10%, as the numbers below show.

The reason is the size of one step's noise relative to b. With the √dt fix, one 1-min step in the wet regime has a noise standard deviation of 17 × √(1/60) = 2.2 mm. The gap between switching rain on and switching it off is b = 0.2 mm. So within a single step, *q* can cross both thresholds and keep going.

`scripts/dt_convergence.py` integrates the paper's scheme for 10 years at four time steps (`outputs/dt_convergence_10yr.log`). The event-scale statistics change with dt as follows:

| statistic | 60 s | 6 s | 0.6 s | 0.06 s | continuum (MN19) |
|---|---|---|---|---|---|
| mean accumulation per event (mm) | 1.93 | 0.69 | 0.34 | 0.25 | 0.2 |
| mean wet-spell duration (min) | 12.9 | 4.6 | 2.3 | 1.7 | 1.33 |
| mean dry-spell duration (min) | 370 | 145 | 71 | 48 | 40 |
| rain events per wet day | 6.5 | 17.1 | 33.5 | 48.8 | |

Each tenfold cut in dt shrinks the departure from the continuum value by a factor of 3.0 to 3.8. A factor of √10 = 3.16 is what you'd expect if the error comes from detecting threshold crossings only at discrete times. That makes these discretization errors. Even the paper's 0.6 s leaves the mean accumulation at 0.34 mm, against b = 0.2 mm. MN19's Fig. 2b would not show this, because for the on–off model it plots their analytic accumulation formula.

The quantities relevant to daily extremes vary over narrower ranges:

- **Accumulation cutoff:** the moment estimate stays between 59 and 73 mm at all four time steps, against the continuum 64.2 mm.
- **Mean wet-day total:** stays between 11.5 and 12.6 mm.
- **Daily-precipitation shape:** in the 100-year runs, 1 min and 0.6 s give method-of-moments P̂_L = 29.6 and 30.0 mm and τ̂_P = 0.57 and 0.61 (MN19 eq. D6).

The appendix-A regression is more sensitive: it gives τ_P = 0.76 at 1 min and 0.55 at 0.6 s. At 1 min a day's rain comes in multiples of one step's 0.15 mm, so the lowest bins are lumpy and pull on the fit. With bin edges placed halfway between those multiples, as in `stoch_rain_model.ipynb` (its section 9), the regression gives τ_P = 0.61 at 1 min and 0.50 at 0.6 s.

So 1 min is usable for daily totals with these parameters. Accumulation exponents, event counts and spell durations need 0.6 s or less.

### 4. Two smaller departures from the paper

These two are compared together, at 1 min with the noise fixed: the notebook's choices in one 100-year integration and the paper's in another.

- **Wet-regime source:** the notebook keeps E + C̄ in the wet regime, where MN19 set it to zero. This reduces the drying rate while raining from 9 to 8.7 mm/h. The daily statistics change little: the method-of-moments P̂_L is 29.34 mm with the notebook's choices and 29.55 mm with the paper's, and τ̂_P is 0.55 and 0.57. Whether those differences exceed sampling noise was not tested.
- **The 1-mm floor:** the notebook clamps *q* to 1 mm, where MN19 restore the previous step's value. After the √dt fix, the floor supplies about 0.1 mm/day under either rule, against 7.3 mm/day of rain.

### 5. Problems in the analysis cells

- **Histogram normalization:** the histograms are of log x with `density=True`. That gives a density per unit ln x, which differs from MN19's PDF per mm by a factor of x. Plotted on log-log axes, every power-law slope is shifted by +1, so the accumulation exponent of 1.5 appears as 0.5. Ratios between runs are unaffected. Comparisons of exponents with the paper are affected. Each histogram also uses numpy's default of 10 bins.
- **Bins in cell 21 (a latent bug):** cell 21 divides El Niño's daily-precipitation histogram by the control's. Each was computed with numpy's default range, from that run's own minimum to its maximum, and the quotient is plotted at the control's bin centers. In the 2023 runs this was harmless by coincidence: both runs' wet-day totals span exactly one 1-min step of rain (0.15 mm) to all-day rain (216 mm), so the bin edges agree. Once the noise is fixed, the maxima differ between runs and the ratio pairs mismatched bins. Pass explicit common bins.
- **No random seed:** the runs are not reproducible. The 2023 run's maximum water vapor was 8.14 m, and the rerun of 2026-10-22 gave 6.52 m.
- **Dry-spell units:** dry-spell lengths are counted in time steps and plotted unlabeled.
- **Unused code:** the bins in cell 7 and the variables `num_steps`, `num_points` and `HOURS_IN_YEAR` are never used.

### 6. It still runs

The unmodified notebook executes top to bottom under Python 3.12, numpy 2.5.4, matplotlib 3.11.2 and puffins 0.3.0 (`pixi.toml`), and its cells print no errors or warnings. It takes 245 s wall time and peaks at 4.1 GB of memory, because the two 120-season integrations keep every minute's state in arrays. One cosmetic change: minor tick labels on panel (e) of cell 17 now overlap panel (d). The numba port integrates 100 years at 1 min in 2 to 3 s and keeps only running statistics.

## Comparison with MN19 Fig. 2b

The paper's scheme at the paper's time step comes close to their Fig. 2b fits. Fitting MN19 eq. (2) to daily precipitation by their appendix-A regression gives τ_P = 0.55 and P_L = 28.9 mm, against their 0.50 and 29.7 mm. The accumulation fit gives an exponent of 1.49 and a cutoff of 66 mm, against the analytic 1.5 and 64.2 mm. This is evidence that the port, with the paper's switches set, is MN19's model. With bins aligned to whole multiples of one step's precipitation, the 100-year integration at 0.6 s in `stoch_rain_model.ipynb` gives τ_P = 0.50 and P_L = 27.7 mm.

## Decisions (Spencer, 2026-10-22)

- The 2023 notebook stays unmodified. Every bug and departure from MN19 listed above is corrected in a new, self-contained notebook, `stoch_rain_model.ipynb`, written for a PhD student reading MN19.
- No India analysis used this notebook's output: Spencer explored the model with it and went no further, so nothing downstream needs revisiting.
