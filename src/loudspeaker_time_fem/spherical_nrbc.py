from __future__ import annotations

import numpy as np
from scipy.special import spherical_jn, spherical_yn


def outgoing_modal_impedance(
    frequency_Hz: float, radius_m: float, sound_speed_m_s: float, max_order: int
) -> np.ndarray:
    """Exact spherical DtN impedance B_l=-d_r(p_l)/p_l for exp(+i*omega*t)."""
    if frequency_Hz <= 0 or radius_m <= 0 or sound_speed_m_s <= 0:
        raise ValueError("frequency, radius, and sound speed must be positive")
    orders = np.arange(int(max_order) + 1)
    k = 2.0 * np.pi * float(frequency_Hz) / float(sound_speed_m_s)
    argument = k * float(radius_m)
    hankel = spherical_jn(orders, argument) - 1j * spherical_yn(orders, argument)
    derivative = spherical_jn(orders, argument, derivative=True) - 1j * spherical_yn(
        orders, argument, derivative=True
    )
    return -k * derivative / hankel
