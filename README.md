# MonteCarlo — Project 3 (Variational Quantum Monte Carlo)

This repository contains code and result generators for Project 3 of Computational Physics (2025/26): Variational Quantum Monte Carlo for the 1D harmonic oscillator, hydrogen atom, and hydrogen molecule.

## Contents
- `ho_vmc.py`: Section 2 / 2.1 / 2.2 utilities for the 1D harmonic oscillator (wavefunctions, local energy, sampling, Monte Carlo energy).
- `ho_validation.py`: Section 2.1/2.2 diagnostics (finite-difference error sweep, eigenstate energy checks).
- `hydrogen_vmc.py`: Section 3 / 3.1 hydrogen atom ansatz, optimisation of rho, sampling, and energy estimation.
- `h2_vmc.py`: Section 4 / 4.1 H₂ ansatz, 6D sampler, local energy, and parameter optimisation.
- `generate_results.py`: Runs all tasks end-to-end and writes figures/data to `results/`.
- `results/`: Generated plots/data for the report.
- `REPORT_PLAN.md`: Suggested 6-page report outline and asset checklist.

## How to reproduce results
```bash
python generate_results.py
```
This produces:
- Section 2.1: `results/ho/fd_error.png` (+ data)
- Section 2.2: `results/ho/sampling_vs_pdf.png`, `energy_checks.txt`
- Section 3.1: `results/hydrogen/xy_density.png`, `optimisation.txt`
- Section 4.1: `results/h2/energies.txt` (includes acceptance + ESS), `morse_fit.png`, density plots at selected separations
- Extensions: `results/extensions/hydrogen_two_param.txt` (2-parameter ansatz, correlated MH), `ho_variance.txt` (variance of local energy), `ho_uniform_vs_importance.txt` (sampling comparison)
- Additional extensions: `results/extensions/adaptive_timestep.txt` (step tuning), `hydrogen_gaussian.txt` (Gaussian trial), `he_ion.txt` (He+), `helium_soft_1d.txt` (soft-Coulomb 1D helium).

## Optional extensions
- Increase sample counts or iterations in `generate_results.py` for higher precision.
- Add more bond lengths to the H₂ sweep for a denser Morse fit.
- Produce additional diagnostics (autocorrelation time, acceptance heatmaps) if needed for the report.
- Explore additional ansätze (e.g., Gaussian/polynomial for HO) or adaptive step-size MC if more time is available.
