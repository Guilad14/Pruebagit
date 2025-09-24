"""Utility for generating Bode magnitude and phase diagrams for linear time-invariant systems.

This module allows specifying the numerator and denominator coefficients of a transfer
function and creates the associated Bode plots using matplotlib.  It can be executed as a
script, reading coefficients from the command line, or imported to use the ``bode_diagram``
function directly from Python code.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class TransferFunction:
    """Representation of a single-input-single-output transfer function.

    The transfer function is represented by real numerator and denominator coefficients in
    descending powers of ``s``.  The data class stores the coefficients as numpy arrays and
    provides a convenience method for evaluating the frequency response.
    """

    numerator: np.ndarray
    denominator: np.ndarray

    @classmethod
    def from_coefficients(
        cls, numerator: Iterable[float], denominator: Iterable[float]
    ) -> "TransferFunction":
        """Create a transfer function from coefficient iterables.

        The coefficients are converted to numpy arrays and validated to ensure the
        denominator is not the zero polynomial.
        """

        num = np.atleast_1d(np.asarray(list(numerator), dtype=float))
        den = np.atleast_1d(np.asarray(list(denominator), dtype=float))
        if not np.any(den):
            raise ValueError("The denominator polynomial must not be zero.")
        return cls(num, den)

    def frequency_response(self, omega: np.ndarray) -> np.ndarray:
        """Evaluate the complex frequency response ``H(jω)`` for frequencies ``ω``.

        Parameters
        ----------
        omega:
            Array of angular frequencies (rad/s) at which to evaluate the response.
        """

        s = 1j * omega
        num_vals = np.polyval(self.numerator, s)
        den_vals = np.polyval(self.denominator, s)
        return num_vals / den_vals


def bode_diagram(
    numerator: Sequence[float],
    denominator: Sequence[float],
    *,
    omega: Sequence[float] | None = None,
    rad_per_sec: bool = True,
    show: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Plot the Bode magnitude and phase for the given transfer function.

    Parameters
    ----------
    numerator, denominator:
        Coefficients of the numerator and denominator polynomials (descending order).
    omega:
        Optional list/array of angular frequencies to evaluate.  When omitted a logspace
        sweep is created based on the poles/zeros of the transfer function.
    rad_per_sec:
        If ``True`` the horizontal axis is labelled in rad/s, otherwise in Hz.
    show:
        When ``True`` the matplotlib figure is shown immediately.

    Returns
    -------
    ω, magnitude, phase: tuple of numpy arrays
        The angular frequency vector, magnitude in dB, and phase in degrees.
    """

    tf = TransferFunction.from_coefficients(numerator, denominator)

    if omega is None:
        # Choose frequency range based on pole/zero magnitudes; fall back to default range.
        roots = np.concatenate((np.roots(tf.numerator), np.roots(tf.denominator)))
        finite_roots = roots[np.isfinite(roots) & (roots != 0)]
        if finite_roots.size:
            min_freq = 0.1 * np.min(np.abs(finite_roots))
            max_freq = 10 * np.max(np.abs(finite_roots))
            min_freq = max(min_freq, 1e-2)
            max_freq = max(max_freq, min_freq * 10)
        else:
            min_freq, max_freq = 1e-2, 1e2
        omega = np.logspace(np.log10(min_freq), np.log10(max_freq), 500)
    else:
        omega = np.asarray(omega, dtype=float)

    response = tf.frequency_response(omega)
    magnitude_db = 20 * np.log10(np.abs(response))
    phase_deg = np.unwrap(np.angle(response)) * 180 / np.pi

    fig, (ax_mag, ax_phase) = plt.subplots(2, 1, sharex=True)
    fig.suptitle("Diagrama de Bode")

    freq = omega if rad_per_sec else omega / (2 * np.pi)
    x_label = "Frecuencia (rad/s)" if rad_per_sec else "Frecuencia (Hz)"

    ax_mag.semilogx(freq, magnitude_db)
    ax_mag.set_ylabel("Magnitud (dB)")
    ax_mag.grid(True, which="both")

    ax_phase.semilogx(freq, phase_deg)
    ax_phase.set_ylabel("Fase (°)")
    ax_phase.set_xlabel(x_label)
    ax_phase.grid(True, which="both")

    if show:
        plt.show()

    return omega, magnitude_db, phase_deg


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Genera diagramas de Bode para funciones de transferencia."
    )
    parser.add_argument(
        "numerator",
        nargs="+",
        type=float,
        help="Coeficientes del numerador en potencias descendentes de s.",
    )
    parser.add_argument(
        "-d",
        "--denominator",
        nargs="+",
        type=float,
        required=True,
        help="Coeficientes del denominador en potencias descendentes de s.",
    )
    parser.add_argument(
        "--hz",
        action="store_true",
        help="Muestra la frecuencia en Hertz en lugar de radianes por segundo.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="No mostrar la ventana de matplotlib (útil para scripts).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    bode_diagram(
        args.numerator,
        args.denominator,
        rad_per_sec=not args.hz,
        show=not args.no_show,
    )


if __name__ == "__main__":
    main()
