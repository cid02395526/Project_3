"""
Variational Monte Carlo for the hydrogen atom.

Sections covered from the brief:
- Section 3: define the 3D Hamiltonian and exponential ansatz psi(r; rho).
- Section 3.1: sample epsilon(R; rho), compute local energy, and minimise the
  variational parameter rho.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Tuple

import numpy as np


def psi_hydrogen(r: np.ndarray, rho: float) -> np.ndarray:
    """Ansatz wavefunction psi(r; rho) = exp(-rho * |r|)."""
    r = np.asarray(r)
    norm = np.linalg.norm(r, axis=-1)
    return np.exp(-rho * norm)


def probability_density(r: np.ndarray, rho: float) -> np.ndarray:
    """Probability density proportional to |psi|^2."""
    psi = psi_hydrogen(r, rho)
    return np.abs(psi) ** 2


def local_energy(
    r: np.ndarray,
    rho: float,
    r_cut: float = 1e-8,
) -> np.ndarray:
    """
    Local energy for the hydrogen ansatz psi(r; rho).

    H = -1/2 nabla^2 - 1/|r|.
    Analytic expression: E_L = -0.5*rho^2 + (rho - 1)/r.
    """
    r = np.asarray(r)
    norm_r = np.linalg.norm(r, axis=-1)
    norm_r = np.maximum(norm_r, r_cut)
    el = -0.5 * rho * rho + (rho - 1.0) / norm_r
    return el


@dataclass
class MetropolisResult:
    samples: np.ndarray
    acceptance_rate: float


def metropolis_sample(
    log_pdf: Callable[[np.ndarray], float],
    initial: np.ndarray,
    num_samples: int,
    step_size: float = 1.0,
    burn_in: int = 2_000,
    thin: int = 2,
    rng: np.random.Generator | None = None,
) -> MetropolisResult:
    """
    Metropolis-Hastings sampler for 3D vectors using Gaussian proposals.
    """
    rng = rng or np.random.default_rng()
    x = np.array(initial, dtype=float)
    log_p = float(log_pdf(x))
    kept = []
    accepted = 0
    total = 0
    target_count = num_samples * thin + burn_in

    while len(kept) < num_samples:
        proposal = x + rng.normal(scale=step_size, size=3)
        log_p_new = float(log_pdf(proposal))
        if np.isfinite(log_p_new):
            accept = np.log(rng.random()) < min(0.0, log_p_new - log_p)
        else:
            accept = False
        total += 1
        if accept:
            x = proposal
            log_p = log_p_new
            accepted += 1
        if total > burn_in and (total - burn_in) % thin == 0:
            kept.append(x.copy())
        if total >= target_count + burn_in:
            break
    acceptance_rate = accepted / max(total, 1)
    return MetropolisResult(samples=np.stack(kept, axis=0), acceptance_rate=acceptance_rate)


def estimate_energy(
    rho: float,
    num_samples: int = 20_000,
    step_size: float = 0.8,
    rng: np.random.Generator | None = None,
) -> Tuple[float, float, MetropolisResult]:
    """
    Estimate <H> for the hydrogen ansatz at a fixed rho.
    """
    rng = rng or np.random.default_rng()

    def _log_pdf(r_vec: np.ndarray) -> float:
        p = float(probability_density(r_vec, rho))
        return -np.inf if p <= 0.0 else np.log(p)

    initial = np.array([0.5, 0.0, 0.0])
    result = metropolis_sample(
        log_pdf=_log_pdf,
        initial=initial,
        num_samples=num_samples,
        step_size=step_size,
        burn_in=2_000,
        thin=2,
        rng=rng,
    )
    energies = local_energy(
        result.samples,
        rho,
    )
    finite = np.isfinite(energies)
    energies = energies[finite]
    mean_energy = float(np.mean(energies))
    stderr = float(np.std(energies, ddof=1) / np.sqrt(len(energies)))
    return mean_energy, stderr, result


def optimise_rho(
    rho0: float = 1.0,
    lr: float = 0.1,
    iters: int = 15,
    num_samples: int = 8_000,
    step_size: float = 0.7,
    rng: np.random.Generator | None = None,
) -> Tuple[float, list[tuple[int, float, float]]]:
    """
    Simple stochastic gradient descent on rho using the analytic gradient (eq. 16).

    Returns best rho and a history list of (iter, rho, energy).
    """
    rng = rng or np.random.default_rng()
    rho = float(rho0)
    history: list[tuple[int, float, float]] = []

    for i in range(iters):
        mean_e, _, res = estimate_energy(
            rho,
            num_samples=num_samples,
            step_size=step_size,
            rng=rng,
        )
        samples = res.samples
        psi_vals = psi_hydrogen(samples, rho)
        grad_log_psi = -np.linalg.norm(samples, axis=1)
        energies = local_energy(samples, rho)
        finite = np.isfinite(energies)
        energies = energies[finite]
        grad_log_psi = grad_log_psi[finite]
        el_centered = energies - np.mean(energies)
        grad = 2.0 * np.mean(el_centered * grad_log_psi)
        rho -= lr * grad
        history.append((i, rho, mean_e))
    return rho, history


def _demo():
    """Quick demo for Section 3.1: optimise rho and report energy."""
    rng = np.random.default_rng(123)
    rho_opt, hist = optimise_rho(rho0=0.8, lr=0.1, iters=15, rng=rng)
    mean_e, err_e, res = estimate_energy(rho_opt, num_samples=40_000, step_size=0.7, rng=rng)
    print("Hydrogen atom VMC (Sections 3 & 3.1)")
    print(f"  Optimised rho ≈ {rho_opt:.4f}")
    print(f"  Energy <H> ≈ {mean_e:.5f} ± {err_e:.5f} (target -0.5)")
    print(f"  Acceptance rate: {res.acceptance_rate:.3f}")
    print("  Optimisation history (iter, rho, energy):")
    for i, r, e in hist:
        print(f"    {i:02d}: rho={r:.4f}, <H>≈{e:.5f}")


if __name__ == "__main__":
    _demo()
