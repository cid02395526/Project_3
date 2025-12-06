Report plan for Project 3 (Variational Quantum Monte Carlo)
==========================================================

Structure (6 pages max)
-----------------------
1) Abstract
   - One-paragraph summary of goals, methods (VMC + Metropolis), key results (H, H2 energies, bond length/dissociation).

2) Introduction (Section 1 of brief)
   - Motivation for VMC; curse of dimensionality; outline of systems (HO, H, H2).

3) Methods
   - Section 2: dimensionless HO Hamiltonian, eigenstates; Monte Carlo estimator of <H>; sampling strategy.
   - Section 2.1: finite-difference stencils (order-2/4), error scaling; chosen h.
   - Section 2.2: Metropolis sampler details (proposal width, burn-in, thinning); validation metrics (acceptance, histogram vs |psi|^2).
   - Section 3 / 3.1: hydrogen ansatz psi=exp(-rho r); analytic local energy; stochastic gradient for rho; sampling settings.
   - Section 4 / 4.1: H2 ansatz (eq. 18) with Jastrow; 6D Laplacian; REINFORCE-style gradient; Morse fit model.

4) Results
   - HO (Section 2.1): Figure `results/ho/fd_error.png`; discuss h that minimises error and order-4 benefit.
   - HO (Section 2.2): Figure `results/ho/sampling_vs_pdf.png`; Table `results/ho/energy_checks.txt` showing En=n+1/2.
   - Hydrogen (Section 3.1): Optimisation curve `results/hydrogen/optimisation.txt`; energy (-0.4995 ± 0.0001); density projection `results/hydrogen/xy_density.png`.
   - H2 (Section 4.1):
       * Energy table `results/h2/energies.txt` across separations.
       * Morse fit plot `results/h2/morse_fit.png`; report fitted D≈0.143, r0≈1.401, a≈1.32; compare to experimental D≈0.17, r0≈1.4.
       * Density maps `results/h2/density_sep_1.0.png`, `density_sep_1.4.png`, `density_sep_2.0.png`.
       * Mention acceptance + ESS diagnostics from `results/h2/energies.txt`.
   - Extensions:
       * Two-parameter hydrogen ansatz with correlated MH: table `results/extensions/hydrogen_two_param.txt` (note cusp constraint beta≈alpha-1 to avoid divergence; best near alpha=1, beta=0, <H>=-0.5).
       * Local-energy variance for HO: `results/extensions/ho_variance.txt`.
       * Uniform vs importance sampling for ⟨x²⟩ in HO: `results/extensions/ho_uniform_vs_importance.txt` (importance sampling vastly lower variance).
       * Adaptive timestep tuning: `results/extensions/adaptive_timestep.txt` (step tuned to target acceptance).
       * Alternate trial wavefunctions: `results/extensions/hydrogen_gaussian.txt` (Gaussian radial trial), `results/extensions/he_ion.txt` (He+ with Z=2).
       * Simple correlated-proposal examples already covered by the two-parameter hydrogen scan (correlated MH).
       * Soft-Coulomb 1D helium toy model: `results/extensions/helium_soft_1d.txt` (energy/variance/acceptance).

5) Discussion
   - Validation: error behaviour vs h; sampling quality; acceptance rates.
   - Hydrogen: proximity to exact -0.5; sensitivity to rho, step size.
   - H2: physical trends with bond length; quality of Morse fit; reasons for any deviations (finite samples, simple ansatz, numeric Laplacian).
   - Extensions: correlated proposals vs independent MH; effect of cusp constraint on 2-parameter ansatz; variance as secondary metric; uniform vs importance sampling variance reduction; adaptive step-size tuning; alternate trials (Gaussian vs Slater); He+ and soft-Coulomb helium toy model.
   - Possible further extensions (excited states targeting, more elaborate soft-Coulomb or multi-parameter ansätze).

6) Conclusion
   - Concise recap of methods and key numerical findings; mention future improvements.

Figures & tables checklist
--------------------------
- Section 2.1: `results/ho/fd_error.png`.
- Section 2.2: `results/ho/sampling_vs_pdf.png`; `results/ho/energy_checks.txt` table.
- Section 3.1: optimisation history (extract small table from `optimisation.txt`), energy, `xy_density.png`.
- Section 4.1: `results/h2/energies.txt` table, `morse_fit.png`, and density plots at select separations.

Reproducibility
---------------
- Mention `python generate_results.py` regenerates all outputs (with seeds set).
- Note sample sizes/iterations used; suggest increasing for higher precision if space permits.
