"""
Extensions and variants for Project 3.

Implemented extensions from the optional list:
- Two-parameter hydrogen ansatz psi(r; alpha, beta) = (1 + beta r) exp(-alpha r).
- Correlated Metropolis-Hastings (AR(1)-style proposals).
- Numeric 3D Laplacian for general scalar fields.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Tuple

import numpy as np


def laplacian_3d_numeric(
    f: Callable[[np.ndarray], np.ndarray],
    r: np.ndarray,
    h: float = 1e-3,
) -> np.ndarray:
    """
    Finite-difference Laplacian for a scalar field f(r) in 3D (order-2 stencil).
    """
    r = np.asarray(r)
    if r.ndim == 1:
        r = r[None, :]
    base = f(r)
    lap = -6.0 * base
    for axis in range(3):
        shift = np.zeros(3)
        shift[axis] = h
        lap += f(r + shift) + f(r - shift)
    return lap / (h * h)


def psi_hydrogen_two_param(r: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    """Two-parameter hydrogen ansatz: (1 + beta r) e^{-alpha r}."""
    r = np.asarray(r)
    norm = np.linalg.norm(r, axis=-1)
    return (1.0 + beta * norm) * np.exp(-alpha * norm)


def probability_density_two_param(r: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    psi = psi_hydrogen_two_param(r, alpha, beta)
    return np.abs(psi) ** 2


def local_energy_two_param(
    r: np.ndarray,
    alpha: float,
    beta: float,
    r_cut: float = 1e-8,
) -> np.ndarray:
    """
    Local energy for the two-parameter hydrogen ansatz using analytic Laplacian.
    H = -1/2 nabla^2 - 1/r.
    """
    r = np.asarray(r)
    norm_r = np.linalg.norm(r, axis=-1)
    norm_r = np.maximum(norm_r, r_cut)

    a = alpha
    b = beta
    num = (
        0.5 * a * norm_r * (-a * (b * norm_r + 1.0) + 2.0 * b)
        + a * (b * norm_r + 1.0)
        - b * norm_r
        - b
        - 1.0
    )
    den = norm_r * (b * norm_r + 1.0)
    return num / den


@dataclass
class MetropolisResult:
    samples: np.ndarray
    acceptance_rate: float


def correlated_metropolis(
    log_pdf: Callable[[np.ndarray], float],
    initial: np.ndarray,
    num_samples: int,
    rho: float = 0.9,
    sigma: float = 0.5,
    burn_in: int = 2_000,
    thin: int = 2,
    rng: np.random.Generator | None = None,
) -> MetropolisResult:
    """
    Metropolis-Hastings with correlated (AR(1)-style) Gaussian proposals.

    proposal = rho * current + sqrt(1 - rho^2) * Normal(0, sigma^2 I)
    """
    rng = rng or np.random.default_rng()
    x = np.array(initial, dtype=float)
    log_p = float(log_pdf(x))
    kept = []
    accepted = 0
    total = 0
    target_count = num_samples * thin + burn_in
    scale = np.sqrt(1.0 - rho * rho) * sigma

    while len(kept) < num_samples:
        proposal = rho * x + rng.normal(scale=scale, size=x.shape)
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
    acceptance_rate = accepted / max(total, 1)
    return MetropolisResult(samples=np.stack(kept, axis=0), acceptance_rate=acceptance_rate)


def estimate_energy_two_param(
    alpha: float,
    beta: float,
    num_samples: int = 30_000,
    rho: float = 0.9,
    sigma: float = 0.6,
    rng: np.random.Generator | None = None,
) -> Tuple[float, float, float, MetropolisResult]:
    """
    Estimate <H> and Var(E_L) for the two-parameter hydrogen ansatz using correlated MH.
    """
    rng = rng or np.random.default_rng()

    def _log_pdf(r_vec: np.ndarray) -> float:
        p = float(probability_density_two_param(r_vec, alpha, beta))
        return -np.inf if p <= 0.0 else np.log(p)

    initial = np.array([0.5, 0.0, 0.0])
    res = correlated_metropolis(
        log_pdf=_log_pdf,
        initial=initial,
        num_samples=num_samples,
        rho=rho,
        sigma=sigma,
        burn_in=3_000,
        thin=2,
        rng=rng,
    )
    energies = local_energy_two_param(res.samples, alpha, beta)
    finite = np.isfinite(energies)
    energies = energies[finite]
    mean_e = float(np.mean(energies))
    var_e = float(np.var(energies, ddof=1))
    stderr = float(np.std(energies, ddof=1) / np.sqrt(len(energies)))
    return mean_e, stderr, var_e, res


def estimate_energy_gaussian_hydrogen(
    alpha: float,
    num_samples: int = 20_000,
    step_size: float = 0.7,
    rng: np.random.Generator | None = None,
) -> Tuple[float, float, float, MetropolisResult]:
    """
    Gaussian radial trial psi = exp(-alpha r^2) for hydrogen (one electron).
    Local energy analytic: E_L = -1/2*(-6a+4a^2 r^2) - 1/r.
    """
    rng = rng or np.random.default_rng()

    def psi(r):
        r = np.asarray(r)
        n = np.linalg.norm(r, axis=-1)
        return np.exp(-alpha * n * n)

    def local_energy(r):
        r = np.asarray(r)
        n = np.linalg.norm(r, axis=-1)
        n = np.maximum(n, 1e-8)
        return -0.5 * (-6 * alpha + 4 * alpha * alpha * n * n) - 1.0 / n

    def _log_pdf(r_vec: np.ndarray) -> float:
        p = float(np.abs(psi(r_vec)) ** 2)
        return -np.inf if p <= 0.0 else np.log(p)

    initial = np.array([0.5, 0.0, 0.0])
    res = correlated_metropolis(
        log_pdf=_log_pdf,
        initial=initial,
        num_samples=num_samples,
        rho=0.9,
        sigma=step_size,
        burn_in=2_000,
        thin=2,
        rng=rng,
    )
    el = local_energy(res.samples)
    el = el[np.isfinite(el)]
    mean_e = float(np.mean(el))
    var_e = float(np.var(el, ddof=1))
    stderr = float(np.std(el, ddof=1) / np.sqrt(len(el)))
    return mean_e, stderr, var_e, res


def tune_step_size(
    log_pdf: Callable[[float], float],
    initial: float,
    target_accept: float = 0.5,
    init_step: float = 0.8,
    iters: int = 50,
    rng: np.random.Generator | None = None,
) -> Tuple[float, float]:
    """
    Simple adaptive tuning for a 1D Metropolis sampler to hit a target acceptance.
    """
    rng = rng or np.random.default_rng()
    step = init_step
    x = initial
    log_p = log_pdf(x)
    accept = 0.0
    for i in range(iters):
        prop = x + rng.normal(scale=step)
        lp_new = log_pdf(prop)
        if np.isfinite(lp_new) and np.log(rng.random()) < min(0.0, lp_new - log_p):
            x = prop
            log_p = lp_new
            acc = 1.0
        else:
            acc = 0.0
        accept = accept * 0.9 + 0.1 * acc
        # adjust step gently
        if accept > target_accept:
            step *= 1.02
        else:
            step *= 0.98
    return step, accept


def estimate_energy_he_ion(
    Z: float = 2.0,
    rho: float = 2.0,
    num_samples: int = 20_000,
    step_size: float = 0.7,
    rng: np.random.Generator | None = None,
) -> Tuple[float, float, float, float]:
    """
    Hydrogenic ion (He+) with potential -Z/r and Slater ansatz exp(-rho r).
    """
    rng = rng or np.random.default_rng()

    def psi(r):
        r = np.asarray(r)
        n = np.linalg.norm(r, axis=-1)
        return np.exp(-rho * n)

    def local_energy(r):
        r = np.asarray(r)
        n = np.linalg.norm(r, axis=-1)
        n = np.maximum(n, 1e-8)
        return -0.5 * rho * rho + (rho - Z) / n

    def _log_pdf(r_vec: np.ndarray) -> float:
        p = float(np.abs(psi(r_vec)) ** 2)
        return -np.inf if p <= 0.0 else np.log(p)

    initial = np.array([0.5, 0.0, 0.0])
    res = correlated_metropolis(
        log_pdf=_log_pdf,
        initial=initial,
        num_samples=num_samples,
        rho=0.9,
        sigma=step_size,
        burn_in=2_000,
        thin=2,
        rng=rng,
    )
    el = local_energy(res.samples)
    el = el[np.isfinite(el)]
    mean_e = float(np.mean(el))
    var_e = float(np.var(el, ddof=1))
    stderr = float(np.std(el, ddof=1) / np.sqrt(len(el)))
    return mean_e, stderr, var_e, res.acceptance_rate


def estimate_soft_helium_1d(
    a_soft: float = 0.5,
    alpha: float = 1.0,
    num_samples: int = 30_000,
    step_size: float = 0.5,
    rng: np.random.Generator | None = None,
) -> dict:
    """
    Simple 1D soft-Coulomb helium model with two electrons.
    V = -2/sqrt(x1^2+a^2) -2/sqrt(x2^2+a^2) + 1/sqrt((x1-x2)^2+a^2).
    Trial psi = exp(-alpha(|x1|+|x2|)).
    """
    rng = rng or np.random.default_rng()

    def psi(x1x2):
        x1, x2 = x1x2[..., 0], x1x2[..., 1]
        return np.exp(-alpha * (np.abs(x1) + np.abs(x2)))

    def local_energy(x1x2):
        x1, x2 = x1x2[..., 0], x1x2[..., 1]
        # kinetic in 1D via second derivative of exp(-alpha|x|)
        kin = -0.5 * (alpha * alpha) * 2.0  # two electrons, each gives alpha^2
        pot = (
            -2.0 / np.sqrt(x1 * x1 + a_soft * a_soft)
            -2.0 / np.sqrt(x2 * x2 + a_soft * a_soft)
            + 1.0 / np.sqrt((x1 - x2) ** 2 + a_soft * a_soft)
        )
        return kin + pot

    def _log_pdf(x1x2: np.ndarray) -> float:
        p = float(np.abs(psi(x1x2)) ** 2)
        return -np.inf if p <= 0 else np.log(p)

    # 2D correlated MH
    res = correlated_metropolis(
        log_pdf=_log_pdf,
        initial=np.array([0.2, -0.2]),
        num_samples=num_samples,
        rho=0.9,
        sigma=step_size,
        burn_in=3_000,
        thin=2,
        rng=rng,
    )
    el = local_energy(res.samples)
    el = el[np.isfinite(el)]
    return {
        "alpha": alpha,
        "a_soft": a_soft,
        "mean_E": float(np.mean(el)),
        "stderr": float(np.std(el, ddof=1) / np.sqrt(len(el))),
        "var_E": float(np.var(el, ddof=1)),
        "accept": res.acceptance_rate,
    }
