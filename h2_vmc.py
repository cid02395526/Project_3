"""
Variational Monte Carlo for the hydrogen molecule (H2).

Sections covered from the brief:
- Section 4: two-electron, two-nucleus Hamiltonian and molecular ansatz.
- Section 4.1: sample epsilon(R; theta), compute local energy numerically,
  and optimise the parameters across bond lengths.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Tuple

import numpy as np


def place_nuclei(separation: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Positions of the two nuclei along z with separation |q1 - q2| = separation.
    """
    half = 0.5 * separation
    q1 = np.array([0.0, 0.0, -half])
    q2 = np.array([0.0, 0.0, half])
    return q1, q2


def h2_ansatz(
    r1: np.ndarray,
    r2: np.ndarray,
    q1: np.ndarray,
    q2: np.ndarray,
    rho1: float,
    rho2: float,
    rho3: float,
) -> np.ndarray:
    """
    Molecular ansatz (eq. 18): psi(r1,r2; rho) = (exp(-rho1(|r1-q1|+|r2-q2|))
    + exp(-rho1(|r1-q2|+|r2-q1|))) * exp(-rho2 / (1 + rho3*|r1-r2|)).
    """
    r1 = np.asarray(r1)
    r2 = np.asarray(r2)
    q1 = np.asarray(q1)
    q2 = np.asarray(q2)
    d1 = np.linalg.norm(r1 - q1, axis=-1)
    d2 = np.linalg.norm(r2 - q2, axis=-1)
    d1_swap = np.linalg.norm(r1 - q2, axis=-1)
    d2_swap = np.linalg.norm(r2 - q1, axis=-1)
    r12 = np.linalg.norm(r1 - r2, axis=-1)
    term_a = np.exp(-rho1 * (d1 + d2))
    term_b = np.exp(-rho1 * (d1_swap + d2_swap))
    jastrow = np.exp(-rho2 / (1.0 + rho3 * r12))
    return (term_a + term_b) * jastrow


def probability_density(
    r1: np.ndarray,
    r2: np.ndarray,
    q1: np.ndarray,
    q2: np.ndarray,
    rho1: float,
    rho2: float,
    rho3: float,
) -> np.ndarray:
    """Probability density proportional to |psi|^2."""
    psi = h2_ansatz(r1, r2, q1, q2, rho1, rho2, rho3)
    return np.abs(psi) ** 2


def flatten_coords(r1: np.ndarray, r2: np.ndarray) -> np.ndarray:
    """Flatten (r1, r2) into a 6-vector per sample."""
    r1 = np.asarray(r1)
    r2 = np.asarray(r2)
    if r1.ndim == 1:
        r1 = r1[None, :]
        r2 = r2[None, :]
    return np.concatenate([r1, r2], axis=-1)


def unflatten_coords(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Inverse of flatten_coords."""
    x = np.asarray(x)
    if x.ndim == 1:
        x = x[None, :]
    r1 = x[..., :3]
    r2 = x[..., 3:]
    return r1, r2


def laplacian_numeric(
    f: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray,
    h: float = 5e-3,
) -> np.ndarray:
    """
    Finite-difference Laplacian in 6D using a simple central stencil.
    """
    x = np.asarray(x)
    if x.ndim == 1:
        x = x[None, :]
    base = f(x)
    lap = -2.0 * 6.0 * base  # each dimension contributes -2*f
    for dim in range(6):
        shift = np.zeros(6)
        shift[dim] = h
        lap += f(x + shift) + f(x - shift)
    return lap / (h * h)


def local_energy(
    r1: np.ndarray,
    r2: np.ndarray,
    q1: np.ndarray,
    q2: np.ndarray,
    rho1: float,
    rho2: float,
    rho3: float,
    h: float = 5e-3,
    r_cut: float = 1e-8,
) -> np.ndarray:
    """
    Local energy for the H2 ansatz.

    Hamiltonian (dimensionless):
    H = -1/2 sum_i nabla_i^2 - sum_i sum_j 1/|ri - qj| + 1/|r1 - r2| + 1/|q1 - q2|.
    """
    r1 = np.asarray(r1)
    r2 = np.asarray(r2)
    flat = flatten_coords(r1, r2)

    def psi_flat(z: np.ndarray) -> np.ndarray:
        a, b = unflatten_coords(z)
        return h2_ansatz(a, b, q1, q2, rho1, rho2, rho3)

    psi = psi_flat(flat)
    small = np.abs(psi) < 1e-14
    kinetic = -0.5 * laplacian_numeric(psi_flat, flat, h=h)

    d1q1 = np.linalg.norm(r1 - q1, axis=-1)
    d1q2 = np.linalg.norm(r1 - q2, axis=-1)
    d2q1 = np.linalg.norm(r2 - q1, axis=-1)
    d2q2 = np.linalg.norm(r2 - q2, axis=-1)
    r12 = np.linalg.norm(r1 - r2, axis=-1)
    nuc_dist = np.linalg.norm(q1 - q2)

    d1q1 = np.maximum(d1q1, r_cut)
    d1q2 = np.maximum(d1q2, r_cut)
    d2q1 = np.maximum(d2q1, r_cut)
    d2q2 = np.maximum(d2q2, r_cut)
    r12 = np.maximum(r12, r_cut)

    potential = (
        -1.0 / d1q1
        - 1.0 / d1q2
        - 1.0 / d2q1
        - 1.0 / d2q2
        + 1.0 / r12
        + 1.0 / nuc_dist
    )

    with np.errstate(divide="ignore", invalid="ignore"):
        el = kinetic / psi + potential
    el = np.where(small, np.nan, el)
    return el


@dataclass
class MetropolisResult:
    samples: np.ndarray  # shape (N, 6) flattened
    acceptance_rate: float


def metropolis_sample(
    log_pdf: Callable[[np.ndarray], float],
    initial: np.ndarray,
    num_samples: int,
    step_size: float = 0.7,
    burn_in: int = 3_000,
    thin: int = 2,
    rng: np.random.Generator | None = None,
) -> MetropolisResult:
    """
    Metropolis-Hastings sampler for 6D coordinates with Gaussian proposals.
    """
    rng = rng or np.random.default_rng()
    x = np.array(initial, dtype=float)
    log_p = float(log_pdf(x))
    kept = []
    accepted = 0
    total = 0
    target_count = num_samples * thin + burn_in

    while len(kept) < num_samples:
        proposal = x + rng.normal(scale=step_size, size=6)
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
    separation: float,
    params: tuple[float, float, float],
    num_samples: int = 15_000,
    step_size: float = 0.65,
    h: float = 5e-3,
    rng: np.random.Generator | None = None,
) -> tuple[float, float, MetropolisResult]:
    """
    Estimate <H> for given bond length and parameters.
    """
    rho1, rho2, rho3 = params
    rng = rng or np.random.default_rng()
    q1, q2 = place_nuclei(separation)

    def _log_pdf(flat: np.ndarray) -> float:
        r1, r2 = unflatten_coords(flat)
        p = float(probability_density(r1, r2, q1, q2, rho1, rho2, rho3))
        return -np.inf if p <= 0.0 else np.log(p)

    initial = np.concatenate([q1 + [0.2, 0.0, 0.0], q2 + [-0.2, 0.0, 0.0]])
    result = metropolis_sample(
        log_pdf=_log_pdf,
        initial=initial,
        num_samples=num_samples,
        step_size=step_size,
        burn_in=3_000,
        thin=2,
        rng=rng,
    )
    r1, r2 = unflatten_coords(result.samples)
    energies = local_energy(
        r1,
        r2,
        q1,
        q2,
        rho1,
        rho2,
        rho3,
        h=h,
    )
    finite = np.isfinite(energies)
    energies = energies[finite]
    mean_e = float(np.mean(energies))
    stderr = float(np.std(energies, ddof=1) / np.sqrt(len(energies)))
    return mean_e, stderr, result


def optimise_parameters(
    separation: float,
    params0: tuple[float, float, float] = (1.0, 0.5, 0.5),
    lr: float = 0.05,
    iters: int = 20,
    num_samples: int = 8_000,
    step_size: float = 0.65,
    rng: np.random.Generator | None = None,
) -> tuple[tuple[float, float, float], list[tuple[int, tuple[float, float, float], float]]]:
    """
    Optimise (rho1, rho2, rho3) using a log-psi gradient estimator (REINFORCE).
    """
    rng = rng or np.random.default_rng()
    rho1, rho2, rho3 = params0
    history: list[tuple[int, tuple[float, float, float], float]] = []
    q1, q2 = place_nuclei(separation)

    def log_psi_terms(r1: np.ndarray, r2: np.ndarray):
        d1 = np.linalg.norm(r1 - q1, axis=-1)
        d2 = np.linalg.norm(r2 - q2, axis=-1)
        d1_swap = np.linalg.norm(r1 - q2, axis=-1)
        d2_swap = np.linalg.norm(r2 - q1, axis=-1)
        r12 = np.linalg.norm(r1 - r2, axis=-1)
        term_a = np.exp(-rho1 * (d1 + d2))
        term_b = np.exp(-rho1 * (d1_swap + d2_swap))
        pref = term_a + term_b
        pref = np.maximum(pref, 1e-12)
        log_pref = np.log(pref)
        d_log_pref_rho1 = -( (d1 + d2) * term_a + (d1_swap + d2_swap) * term_b ) / pref
        log_jastrow = -rho2 / (1.0 + rho3 * r12)
        d_log_jastrow_rho2 = -1.0 / (1.0 + rho3 * r12)
        d_log_jastrow_rho3 = rho2 * r12 / (1.0 + rho3 * r12) ** 2
        log_psi = log_pref + log_jastrow
        return log_psi, d_log_pref_rho1 + 0.0, d_log_jastrow_rho2, d_log_jastrow_rho3

    for i in range(iters):
        def _log_pdf(flat: np.ndarray) -> float:
            r1, r2 = unflatten_coords(flat)
            p = float(probability_density(r1, r2, q1, q2, rho1, rho2, rho3))
            return -np.inf if p <= 0.0 else np.log(p)

        initial = np.concatenate([q1 + [0.2, 0.0, 0.0], q2 + [-0.2, 0.0, 0.0]])
        res = metropolis_sample(
            log_pdf=_log_pdf,
            initial=initial,
            num_samples=num_samples,
            step_size=step_size,
            burn_in=3_000,
            thin=2,
            rng=rng,
        )
        r1, r2 = unflatten_coords(res.samples)
        energies = local_energy(
            r1,
            r2,
            q1,
            q2,
            rho1,
            rho2,
            rho3,
            h=1e-2,
        )
        finite = np.isfinite(energies)
        energies = energies[finite]
        r1 = r1[finite]
        r2 = r2[finite]
        log_psi, d_rho1, d_rho2, d_rho3 = log_psi_terms(r1, r2)
        el_centered = energies - np.mean(energies)
        grad_rho1 = 2.0 * np.mean(el_centered * d_rho1)
        grad_rho2 = 2.0 * np.mean(el_centered * d_rho2)
        grad_rho3 = 2.0 * np.mean(el_centered * d_rho3)
        rho1 -= lr * grad_rho1
        rho2 -= lr * grad_rho2
        rho3 -= lr * grad_rho3
        mean_e = float(np.mean(energies))
        history.append((i, (rho1, rho2, rho3), mean_e))
    return (rho1, rho2, rho3), history


def _demo():
    """Quick demo for one bond length (Section 4.1)."""
    rng = np.random.default_rng(123)
    separation = 1.4
    params_opt, hist = optimise_parameters(
        separation=separation,
        params0=(1.0, 0.5, 0.5),
        lr=0.02,
        iters=8,
        num_samples=6_000,
        step_size=0.65,
        rng=rng,
    )
    mean_e, err_e, res = estimate_energy(
        separation=separation,
        params=params_opt,
        num_samples=12_000,
        step_size=0.7,
        h=1e-2,
        rng=rng,
    )
    print("H2 VMC demo (Sections 4 & 4.1)")
    print(f"  Separation: {separation}")
    print(f"  Optimised params (rho1, rho2, rho3): {tuple(f'{p:.3f}' for p in params_opt)}")
    print(f"  Energy <H> ≈ {mean_e:.4f} ± {err_e:.4f}")
    print(f"  Acceptance rate: {res.acceptance_rate:.3f}")
    print("  Optimisation history (iter, params, energy):")
    for i, ps, e in hist:
        r1, r2, r3 = ps
        print(f"    {i:02d}: ({r1:.3f}, {r2:.3f}, {r3:.3f}) -> {e:.4f}")


if __name__ == "__main__":
    _demo()
