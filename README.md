# stoch-rain-model

Python code for the on–off stochastic column-moisture model of Martinez-Villalobos and Neelin (2019, *J. Atmos. Sci.*, 76, 3611–3631, [doi:10.1175/JAS-D-18-0343.1](https://doi.org/10.1175/JAS-D-18-0343.1); MN19 below). In that model column water vapor follows a random walk, and precipitation at a fixed rate switches on when the water vapor rises above a threshold and off when it falls a small distance below it.

## Files

- `stoch_rain_model.ipynb`: a self-contained notebook, written 2026-10-22 for a PhD student reading MN19. The notebook integrates the model as MN19 specify it and compares the simulated accumulation, spell-duration and daily-precipitation distributions with the paper's analytical results. The notebook also shows which statistics depend on the time step, and repeats an experiment that multiplies the wet-regime noise amplitude by 1.5. The notebook needs numpy, matplotlib and numba, and runs in 2 to 4 minutes. The saved copy includes its outputs.
- `scratch_neelin-stoch.ipynb`: Spencer's 2023 notebook from the India extreme-rainfall project, unmodified. It multiplies the noise term by dt where MN19's Euler–Maruyama scheme multiplies it by √dt. With its 60-s time step, that is MN19's model with both noise amplitudes multiplied by √60 = 7.75. `stoch_rain_model.ipynb` corrects this and the 2023 notebook's other departures from MN19.
- `REVIEW.md`: the 2026-10-22 review of the 2023 notebook against MN19. Its scripts are in `scripts/`, their logs and summaries in `outputs/`, and their figures in `figs/`.
- `tests/test_port.py`: checks `scripts/stoch_rain.py`, the review's numba port of the 2023 model, bit for bit against the 2023 notebook's own class.
- `tests/test_notebook.py`: checks the model function in `stoch_rain_model.ipynb` against that port with MN19's scheme selected. It also checks the noise scaling, the wet-regime tendency, the lower boundary, the moisture budget, the daily event counts, and the notebook's transcriptions of MN19's eqs. (2), (9) and (D6) and of their appendix A fit.
- `martinez-villalobos-neelin-2019-jas.pdf`: the paper, kept out of git for AMS copyright.
- `LICENSE`: the MIT license, which covers the code and notebooks in this repository.

## Running

The environment is defined in `pixi.toml`:

```
pixi run -e dev test                          # all tests
pixi run jupyter lab stoch_rain_model.ipynb   # the notebook
```
