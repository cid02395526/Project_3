"""
Generate results and plots for Project 3 tasks (Sections 2–4.1).

Outputs are written under results/, split by subsection. This script runs:
- Section 2.1: finite-difference error sweep for the 1D HO local energy.
- Section 2.2: sampling verification and energy checks for HO eigenstates.
- Section 3.1: hydrogen atom optimisation, energy estimate, and xy density.
- Section 4.1: H2 optimisation across bond lengths, Morse fit, and densities.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Tuple

import matplotlib

# Use a local config dir to avoid permission warnings.
os.environ.setdefault("MPLCONFIGDIR", str(Path("results/_mplconfig").absolute()))
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.optimize import curve_fit  # noqa: E402
from scipy.ndimage import gaussian_filter  # noqa: E402

from ho_validation import finite_difference_error  # noqa: E402
from ho_vmc import (  # noqa: E402
    estimate_energy as ho_estimate_energy,
    metropolis_sample as ho_metropolis_sample,
    probability_density as ho_prob_density,
)
from hydrogen_vmc import (  # noqa: E402
    estimate_energy as h_estimate_energy,
    optimise_rho,
    probability_density as h_prob_density,
)
from h2_vmc import (  # noqa: E402
    estimate_energy as h2_estimate_energy,
    local_energy as h2_local_energy,
    optimise_parameters,
    place_nuclei,
    unflatten_coords,
)
from extension_vmc import (  # noqa: E402
    estimate_energy_two_param,
    probability_density_two_param,
    estimate_energy_gaussian_hydrogen,
    tune_step_size,
    estimate_energy_he_ion,
    estimate_soft_helium_1d,
)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def effective_sample_size(x: np.ndarray, max_lag: int = 1000) -> float:
    """
    Approximate ESS using autocorrelations until the first negative pair or max_lag.
    """
    x = np.asarray(x)
    x = x - np.mean(x)
    n = len(x)
    if n < 3:
        return float(n)
    var = np.var(x)
    ess_inv = 1.0
    for lag in range(1, min(max_lag, n - 1)):
        c = np.dot(x[:-lag], x[lag:]) / (n - lag)
        rho = c / var if var > 0 else 0.0
        if rho < 0:
            break
        ess_inv += 2 * rho
    return float(n / ess_inv)


def smooth_hist2d(x: np.ndarray, y: np.ndarray, bins: int, span: float, sigma: float = 1.0):
    """Compute and smooth a 2D histogram for nicer visuals."""
    counts, xedges, yedges = np.histogram2d(x, y, bins=bins, range=[[-span, span], [-span, span]])
    counts_s = gaussian_filter(counts, sigma=sigma)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    return counts_s, extent


def section_2_results(base: Path) -> None:
    ho_dir = base / "ho"
    ensure_dir(ho_dir)
    # Section 2.1: finite-difference error sweep
    hs, errors = finite_difference_error(n=0)
    fig, ax = plt.subplots(figsize=(8, 6), dpi=200)
    for order, err in errors.items():
        ax.loglog(hs, err, marker="o", label=f"order {order}")
    ax.set_xlabel("Step size h")
    ax.set_ylabel("Mean |E_L - E_true|")
    ax.set_title("Section 2.1: finite-difference error (n=0)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(ho_dir / "fd_error.png", dpi=200)
    plt.close(fig)
    np.savez(ho_dir / "fd_error_data.npz", hs=hs, **{f"order_{k}": v for k, v in errors.items()})

    # Section 2.2: sampling verification and energy checks
    rng = np.random.default_rng(123)

    def log_pdf(x: float) -> float:
        p = float(ho_prob_density(0, x))
        return -np.inf if p <= 0.0 else np.log(p)

    samp = ho_metropolis_sample(
        log_pdf=log_pdf,
        initial=0.0,
        num_samples=40_000,
        step_size=0.8,
        burn_in=2_000,
        thin=2,
        rng=rng,
    )
    xs = samp.samples
    fig, ax = plt.subplots(figsize=(8, 6), dpi=200)
    ax.hist(xs, bins=120, density=True, alpha=0.6, label="Samples")
    grid = np.linspace(-4, 4, 400)
    pdf_grid = ho_prob_density(0, grid)
    pdf_grid /= np.trapz(pdf_grid, grid)
    ax.plot(grid, pdf_grid, "k-", lw=2, label="|psi_0|^2 (analytic)")
    ax.set_xlabel("x")
    ax.set_ylabel("Probability density")
    ax.set_title("Section 2.2: sampling vs analytic density (n=0)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(ho_dir / "sampling_vs_pdf.png", dpi=200)
    plt.close(fig)
    np.savez(ho_dir / "sampling_data.npz", samples=xs)

    energies = []
    for n in range(4):
        mean_e, err_e, res = ho_estimate_energy(n, num_samples=12_000, step_size=0.8, rng=rng)
        energies.append((n, mean_e, err_e, res.acceptance_rate))
    with open(ho_dir / "energy_checks.txt", "w") as f:
        f.write("n\tmean_E\tstderr\taccept\n")
        for n, m, s, a in energies:
            f.write(f"{n}\t{m:.6f}\t{s:.6f}\t{a:.3f}\n")


def section_3_results(base: Path) -> tuple[float, float, float]:
    h_dir = base / "hydrogen"
    ensure_dir(h_dir)
    rng = np.random.default_rng(456)
    # Optimise rho (Section 3.1)
    rho_opt, hist = optimise_rho(rho0=0.8, lr=0.1, iters=20, num_samples=10_000, step_size=0.7, rng=rng)
    mean_e, err_e, res = h_estimate_energy(rho_opt, num_samples=50_000, step_size=0.7, rng=rng)
    # Save optimisation history
    with open(h_dir / "optimisation.txt", "w") as f:
        f.write("iter\trho\tenergy\n")
        for i, r, e in hist:
            f.write(f"{i}\t{r:.6f}\t{e:.6f}\n")
    # Projected density on x-y plane
    samples = res.samples
    x = samples[:, 0]
    y = samples[:, 1]
    fig, ax = plt.subplots(figsize=(8, 6), dpi=200)
    counts, extent = smooth_hist2d(x, y, bins=400, span=3, sigma=1.0)
    im = ax.imshow(counts.T, origin="lower", extent=extent, cmap="magma", aspect="equal", interpolation="gaussian")
    fig.colorbar(im, ax=ax, label="Counts")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"Hydrogen density projection (rho={rho_opt:.3f})")
    fig.tight_layout()
    fig.savefig(h_dir / "xy_density.png", dpi=200)
    plt.close(fig)
    np.savez(h_dir / "energy.npz", rho_opt=rho_opt, mean_e=mean_e, err_e=err_e, accept=res.acceptance_rate)
    return rho_opt, mean_e, err_e


def morse_potential(r: np.ndarray, D: float, a: float, r0: float, e_single: float) -> np.ndarray:
    return D * (1.0 - np.exp(-a * (r - r0))) ** 2 - D + 2.0 * e_single


def section_4_results(base: Path, e_single: float) -> None:
    h2_dir = base / "h2"
    ensure_dir(h2_dir)
    rng = np.random.default_rng(789)
    separations = np.array([0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.4])
    params = (1.0, 0.5, 0.5)
    energies = []
    params_hist = []

    for sep in separations:
        params, hist = optimise_parameters(
            separation=sep,
            params0=params,
            lr=0.02,
            iters=10,
            num_samples=8_000,
            step_size=0.65,
            rng=rng,
        )
        mean_e, err_e, res = h2_estimate_energy(
            separation=sep,
            params=params,
            num_samples=20_000,
            step_size=0.65,
            h=5e-3,
            rng=rng,
        )
        # Compute local energies for ESS diagnostic
        r1_s, r2_s = unflatten_coords(res.samples)
        loc_e = h2_local_energy(r1_s, r2_s, *place_nuclei(sep), *params, h=5e-3)
        loc_e = loc_e[np.isfinite(loc_e)]
        ess_e = effective_sample_size(loc_e)
        energies.append((sep, mean_e, err_e, res.acceptance_rate, ess_e, params))
        params_hist.append((sep, hist))
        # density projection for selected separations
        if sep in (1.0, 1.4, 2.0):
            r1, r2 = unflatten_coords(res.samples)
            xs = np.concatenate([r1[:, 0], r2[:, 0]])
            ys = np.concatenate([r1[:, 1], r2[:, 1]])
            fig, ax = plt.subplots(figsize=(8, 6), dpi=200)
            counts, extent = smooth_hist2d(xs, ys, bins=400, span=3, sigma=1.0)
            im = ax.imshow(counts.T, origin="lower", extent=extent, cmap="viridis", aspect="equal", interpolation="gaussian")
            fig.colorbar(im, ax=ax, label="Counts")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_title(f"H2 density projection (sep={sep}, params={tuple(round(p,3) for p in params)})")
            fig.tight_layout()
            fig.savefig(h2_dir / f"density_sep_{sep:.1f}.png", dpi=200)
            plt.close(fig)

    # Save energies table
    with open(h2_dir / "energies.txt", "w") as f:
        f.write("sep\tmean_E\tstderr\taccept\tess_energy\trho1\trho2\trho3\n")
        for sep, m, s, a, ess, ps in energies:
            r1, r2, r3 = ps
            f.write(f"{sep:.2f}\t{m:.6f}\t{s:.6f}\t{a:.3f}\t{ess:.1f}\t{r1:.4f}\t{r2:.4f}\t{r3:.4f}\n")

    # Morse fit
    seps = np.array([e[0] for e in energies])
    vals = np.array([e[1] for e in energies])
    popt, pcov = curve_fit(lambda r, D, a, r0: morse_potential(r, D, a, r0, e_single), seps, vals, p0=(0.2, 1.0, 1.4))
    D_fit, a_fit, r0_fit = popt
    fit_curve = morse_potential(seps, D_fit, a_fit, r0_fit, e_single)
    fig, ax = plt.subplots(figsize=(8, 6), dpi=200)
    ax.errorbar(seps, vals, yerr=[e[2] for e in energies], fmt="o", label="VMC energies")
    r_dense = np.linspace(seps.min(), seps.max(), 200)
    ax.plot(r_dense, morse_potential(r_dense, D_fit, a_fit, r0_fit, e_single), label="Morse fit")
    ax.set_xlabel("Separation |q1 - q2|")
    ax.set_ylabel("Energy")
    ax.set_title("H2 ground-state energy vs separation (Section 4.1)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(h2_dir / "morse_fit.png", dpi=200)
    plt.close(fig)
    np.savez(
        h2_dir / "morse_fit.npz",
        separations=seps,
        energies=vals,
        stderr=np.array([e[2] for e in energies]),
        params=np.array([e[5] for e in energies]),
        D=D_fit,
        a=a_fit,
        r0=r0_fit,
    )


def section_extensions(base: Path) -> None:
    ext_dir = base / "extensions"
    ensure_dir(ext_dir)
    rng = np.random.default_rng(321)

    # Hydrogen two-parameter ansatz scan
    alphas = np.linspace(0.8, 1.2, 5)
    betas = np.linspace(-0.05, 0.05, 5)  # small offset around cusp beta ≈ alpha - 1
    records = []
    for a in alphas:
        for b_off in betas:
            b = (a - 1.0) + b_off
            if abs(a - b - 1.0) > 1e-3:
                continue  # enforce cusp condition to avoid divergent local energy
            mean_e, err_e, var_e, res = estimate_energy_two_param(
                alpha=a,
                beta=b,
                num_samples=20_000,
                rho=0.9,
                sigma=0.6,
                rng=rng,
            )
            records.append((a, b, mean_e, err_e, var_e, res.acceptance_rate))
    records.sort(key=lambda x: x[2])
    best = records[0]
    with open(ext_dir / "hydrogen_two_param.txt", "w") as f:
        f.write("alpha\tbeta\tmean_E\tstderr\tvar_E\taccept\n")
        for a, b, m, s, v, acc in records:
            f.write(f"{a:.3f}\t{b:.3f}\t{m:.6f}\t{s:.6f}\t{v:.6f}\t{acc:.3f}\n")
        f.write(f"\nBest: alpha={best[0]:.3f}, beta={best[1]:.3f}, <H>={best[2]:.6f}, var={best[4]:.6f}, accept={best[5]:.3f}\n")

    # HO: variance of local energy for n=0 importance sampling
    from ho_vmc import local_energy as ho_local_energy, metropolis_sample as ho_metropolis_sample, probability_density as ho_prob_density

    def log_pdf(x: float) -> float:
        p = float(ho_prob_density(0, x))
        return -np.inf if p <= 0.0 else np.log(p)

    samp = ho_metropolis_sample(
        log_pdf=log_pdf,
        initial=0.0,
        num_samples=60_000,
        step_size=0.8,
        burn_in=2_000,
        thin=2,
        rng=rng,
    )
    el = ho_local_energy(0, samp.samples)
    el = el[np.isfinite(el)]
    mean_e = float(np.mean(el))
    var_e = float(np.var(el, ddof=1))
    with open(ext_dir / "ho_variance.txt", "w") as f:
        f.write(f"mean_E={mean_e:.6f}\nvar_E={var_e:.6f}\naccept={samp.acceptance_rate:.3f}\n")

    # HO: compare uniform vs importance sampling for <x^2>
    grid_L = 5.0
    uni = rng.uniform(-grid_L, grid_L, size=400_000)
    uni_est = float(np.mean(uni * uni))
    uni_stderr = float(np.std(uni * uni, ddof=1) / np.sqrt(len(uni)))
    imp_xs = samp.samples
    imp_est = float(np.mean(imp_xs * imp_xs))
    imp_stderr = float(np.std(imp_xs * imp_xs, ddof=1) / np.sqrt(len(imp_xs)))
    with open(ext_dir / "ho_uniform_vs_importance.txt", "w") as f:
        f.write("method\testimate\tstderr\n")
        f.write(f"uniform[-{grid_L},{grid_L}]\t{uni_est:.6f}\t{uni_stderr:.6f}\n")
        f.write(f"importance|psi0|^2\t{imp_est:.6f}\t{imp_stderr:.6f}\n")

    # Adaptive timestep tuning example on HO ground state
    tuned_step, accept = tune_step_size(
        log_pdf=lambda x: -np.inf if ho_prob_density(0, x) <= 0 else np.log(ho_prob_density(0, x)),
        initial=0.0,
        target_accept=0.5,
        init_step=0.8,
        iters=50,
        rng=rng,
    )
    with open(ext_dir / "adaptive_timestep.txt", "w") as f:
        f.write(f"tuned_step={tuned_step:.4f}\naccept={accept:.3f}\n")

    # Hydrogen Gaussian trial (Slater-like alternative)
    gauss_alpha = np.linspace(0.3, 1.2, 5)
    gauss_records = []
    for a in gauss_alpha:
        m, s, v, res = estimate_energy_gaussian_hydrogen(alpha=a, num_samples=20_000, rng=rng)
        gauss_records.append((a, m, s, v, res.acceptance_rate))
    gauss_records.sort(key=lambda x: x[1])
    with open(ext_dir / "hydrogen_gaussian.txt", "w") as f:
        f.write("alpha\tmean_E\tstderr\tvar_E\taccept\n")
        for a, m, s, v, acc in gauss_records:
            f.write(f"{a:.3f}\t{m:.6f}\t{s:.6f}\t{v:.6f}\t{acc:.3f}\n")

    # Hydrogen ion (He+) ground state with Z=2
    he1_m, he1_s, he1_v, he1_acc = estimate_energy_he_ion(
        Z=2.0,
        rho=2.0,
        num_samples=30_000,
        step_size=0.7,
        rng=rng,
    )
    with open(ext_dir / "he_ion.txt", "w") as f:
        f.write(f"Z=2 mean_E={he1_m:.6f} stderr={he1_s:.6f} var_E={he1_v:.6f} accept={he1_acc:.3f}\n")

    # Simple 1D soft-Coulomb helium model
    he_soft = estimate_soft_helium_1d(
        a_soft=0.5,
        alpha=1.0,
        num_samples=30_000,
        step_size=0.5,
        rng=rng,
    )
    with open(ext_dir / "helium_soft_1d.txt", "w") as f:
        f.write("soft_he_1d\n")
        for k, v in he_soft.items():
            f.write(f"{k}={v}\n")


def main() -> None:
    base = Path("results")
    ensure_dir(base)
    section_2_results(base)
    rho_opt, e_single, err_single = section_3_results(base)
    section_4_results(base, e_single=e_single)
    section_extensions(base)
    print("Results generated under results/")


if __name__ == "__main__":
    main()
