"""
Validation utilities for the 1D harmonic oscillator tasks.

Sections covered from the brief:
- Section 2.1: scan step size / order to study finite-difference error in the
  local energy.
- Section 2.2: verify Monte Carlo energies for eigenstates using epsilon(x).

Run this file directly to print a small report.
"""

from __future__ import annotations

import numpy as np

from ho_vmc import estimate_energy, local_energy


def finite_difference_error(
    n: int = 0,
    hs: np.ndarray | None = None,
    x_grid: np.ndarray | None = None,
    orders: tuple[int, ...] = (2, 4),
) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    """
    Compute mean absolute error of the local energy for various step sizes.

    Returns step sizes and a dict mapping derivative order -> errors.
    """
    if hs is None:
        hs = np.logspace(-4, -0.5, 12)
    if x_grid is None:
        x_grid = np.linspace(-4.0, 4.0, 400)
    true_energy = n + 0.5
    errors: dict[int, np.ndarray] = {}
    for order in orders:
        errs = []
        for h in hs:
            el = local_energy(
                n,
                x_grid,
                h=float(h),
                derivative_order=order,
            )
            finite = np.isfinite(el)
            if not np.any(finite):
                errs.append(np.nan)
                continue
            mean_abs_err = float(np.mean(np.abs(el[finite] - true_energy)))
            errs.append(mean_abs_err)
        errors[order] = np.array(errs)
    return hs, errors


def print_fd_report(n: int = 0) -> None:
    """Print finite-difference error table for a given state."""
    hs, errors = finite_difference_error(n=n)
    header = "h\torder-2 error\torder-4 error"
    print(f"Finite-difference local energy error for n={n}")
    print(header)
    for i, h in enumerate(hs):
        o2 = errors.get(2, [np.nan] * len(hs))[i]
        o4 = errors.get(4, [np.nan] * len(hs))[i]
        print(f"{h:.1e}\t{o2:.3e}\t\t{o4:.3e}")


def print_energy_checks(max_n: int = 3) -> None:
    """Monte Carlo energy estimates for n=0..max_n."""
    print("\nMonte Carlo energy estimates")
    print("n\t<H>\t\tstderr\taccept")
    for n in range(max_n + 1):
        mean_e, err_e, res = estimate_energy(
            n=n,
            num_samples=5_000,
            step_size=0.8,
        )
        print(f"{n}\t{mean_e:.6f}\t{err_e:.6f}\t{res.acceptance_rate:.3f}")


def main() -> None:
    np.set_printoptions(precision=4, suppress=True)
    print_fd_report(n=0)
    print_energy_checks(max_n=3)


if __name__ == "__main__":
    main()
