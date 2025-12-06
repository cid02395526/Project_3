"""
Variational Monte Carlo utilities for the 1D harmonic oscillator.

Sections covered from the brief:
- Section 2: set up the dimensionless Hamiltonian and eigenstates.
- Section 2.1: compute the local energy via finite-difference Laplacians.
- Section 2.2: Monte Carlo sampling of epsilon(x) and energy estimation.

Everything is in natural units with Hamiltonian H = -1/2 d^2/dx^2 + 1/2 x^2.
"""

from dataclasses import dataclass
from typing import Callable, Iterable, Tuple

import numpy as np
from numpy.polynomial import hermite


def hermite_polynomial(n: int, x: np.ndarray) -> np.ndarray:
    """Physicists' Hermite polynomial H_n evaluated at x."""
    # numpy.polynomial.hermite.hermval uses the physicists' definition.
    coeffs = [0.0] * n + [1.0]
    return hermite.hermval(x, coeffs)


def harmonic_oscillator_wavefunction(n: int, x: np.ndarray) -> np.ndarray:
    """Unnormalised harmonic oscillator eigenfunction psi_n(x)."""
    return hermite_polynomial(n, x) * np.exp(-0.5 * x * x)


def probability_density(n: int, x: np.ndarray) -> np.ndarray:
    """Probability density proportional to |psi_n(x)|^2."""
    psi = harmonic_oscillator_wavefunction(n, x)
    return np.abs(psi) ** 2


def second_derivative(
    f: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray,
    h: float = 1e-3,
    order: int = 2,
) -> np.ndarray:
    """
    Finite-difference approximation to f''(x).

    Parameters
    ----------
    f:
        Function to differentiate.
    x:
        Points at which to evaluate the second derivative.
    h:
        Step size.
    order:
        Supported orders: 2 (3-point stencil) or 4 (5-point stencil).
    """
    if order == 2:
        return (f(x + h) - 2.0 * f(x) + f(x - h)) / (h * h)
    if order == 4:
        return (
            -f(x + 2 * h)
            + 16.0 * f(x + h)
            - 30.0 * f(x)
            + 16.0 * f(x - h)
            - f(x - 2 * h)
        ) / (12.0 * h * h)
    raise ValueError(f"Unsupported order {order}; use 2 or 4.")


def local_energy(
    n: int,
    x: np.ndarray,
    h: float = 1e-3,
    derivative_order: int = 4,
) -> np.ndarray:
    """
    Local energy for the 1D harmonic oscillator eigenstate psi_n at positions x.

    Returns
    -------
    np.ndarray
        Local energy values. Points where psi ~ 0 return np.nan to avoid
        infinities from division.
    """
    psi = harmonic_oscillator_wavefunction(n, x)
    small = np.abs(psi) < 1e-14
    kinetic = -0.5 * second_derivative(
        lambda t: harmonic_oscillator_wavefunction(n, t),
        x,
        h=h,
        order=derivative_order,
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        el = kinetic / psi + 0.5 * x * x
    el = np.where(small, np.nan, el)
    return el


@dataclass
class MetropolisResult:
    samples: np.ndarray
    acceptance_rate: float


def metropolis_sample(
    log_pdf: Callable[[float], float],
    initial: float,
    num_samples: int,
    step_size: float = 1.0,
    burn_in: int = 1000,
    thin: int = 1,
    rng: np.random.Generator | None = None,
) -> MetropolisResult:
    """
    Metropolis-Hastings sampler for 1D target distributions.

    Parameters
    ----------
    log_pdf:
        Function returning log(probability density) up to a constant.
    initial:
        Starting point for the chain.
    num_samples:
        Number of samples to return after burn-in and thinning.
    step_size:
        Standard deviation of the Gaussian proposal distribution.
    burn_in:
        Number of initial steps to discard.
    thin:
        Keep one sample every `thin` steps.
    """
    rng = rng or np.random.default_rng()
    x = float(initial)
    log_p = float(log_pdf(x))
    kept = []
    accepted = 0
    total = 0
    target_count = num_samples * thin + burn_in

    while len(kept) < num_samples:
        proposal = x + rng.normal(scale=step_size)
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
            kept.append(x)
        if total >= target_count + burn_in:
            break
    acceptance_rate = accepted / max(total, 1)
    return MetropolisResult(samples=np.array(kept), acceptance_rate=acceptance_rate)


def estimate_energy(
    n: int,
    num_samples: int = 10_000,
    step_size: float = 1.0,
    h: float = 1e-3,
    derivative_order: int = 4,
    rng: np.random.Generator | None = None,
) -> Tuple[float, float, MetropolisResult]:
    """
    Estimate <H> for the n-th harmonic oscillator eigenstate via Monte Carlo.

    Returns mean energy, its standard error, and the sampling diagnostics.
    """
    rng = rng or np.random.default_rng()

    def _log_pdf(x: float) -> float:
        p = float(probability_density(n, x))
        return -np.inf if p <= 0.0 else np.log(p)

    initial = 0.0
    result = metropolis_sample(
        log_pdf=_log_pdf,
        initial=initial,
        num_samples=num_samples,
        step_size=step_size,
        burn_in=1_000,
        thin=2,
        rng=rng,
    )
    energies = local_energy(
        n,
        result.samples,
        h=h,
        derivative_order=derivative_order,
    )
    finite = np.isfinite(energies)
    energies = energies[finite]
    mean_energy = float(np.mean(energies))
    stderr = float(np.std(energies, ddof=1) / np.sqrt(len(energies)))
    return mean_energy, stderr, result


def _demo():
    """Quick demonstration when run as a script."""
    mean_e, err_e, res = estimate_energy(n=0, num_samples=5_000, step_size=0.8)
    print("Harmonic oscillator ground state (n=0)")
    print(f"  <H> ≈ {mean_e:.5f} ± {err_e:.5f}")
    print(f"  Acceptance rate: {res.acceptance_rate:.3f}")


if __name__ == "__main__":
    _demo()
