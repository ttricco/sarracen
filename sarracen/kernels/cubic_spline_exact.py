import numpy as np
from numba import njit


@njit
def pint(r0, d1, d2, hi1):
    if r0 == 0:
        return 0
    elif r0 > 0:
        result = 1
        ar0 = r0
    else:
        result = -1
        ar0 = -r0

    q0 = ar0 * hi1
    phi1 = np.arctan(abs(d1) / ar0)
    phi2 = np.arctan(abs(d2) / ar0)

    if d1 * d2 >= 0:
        result = result * (full_2d_mod(phi1, q0) + full_2d_mod(phi2, q0))
    elif abs(d1) < abs(d2):
        result = result * (full_2d_mod(phi2, q0) - full_2d_mod(phi1, q0))
    else:
        result = result * (full_2d_mod(phi1, q0) - full_2d_mod(phi2, q0))

    return result


@njit
def full_2d_mod(phi, q0):
    if q0 <= 1.0:
        cphi = np.cos(phi)
        q = q0 / cphi

        if q <= 1.0:
            return F1_2d(phi, q0)
        elif q <= 2.0:
            phi1 = np.arccos(q0)
            return F2_2d(phi, q0) - F2_2d(phi1, q0) + F1_2d(phi1, q0)
        else:
            phi1 = np.arccos(q0)
            phi2 = np.arccos(0.5 * q0)
            return F3_2d(phi) - F3_2d(phi2) + F2_2d(phi2, q0) - F2_2d(phi1, q0) + F1_2d(
                phi1, q0)
    elif q0 <= 2.0:
        cphi = np.cos(phi)
        q = q0 / cphi

        if q <= 2.0:
            return F2_2d(phi, q0)
        else:
            phi2 = np.arccos(0.5 * q0)
            return F3_2d(phi) - F3_2d(phi2) + F2_2d(phi2, q0)
    else:
        return F3_2d(phi)


@njit
def F1_2d(phi, q0):
    cphi2 = np.cos(phi) ** 2

    logs = np.log(np.tan(phi / 2. + np.pi / 4.))

    I2 = np.tan(phi)
    I4 = 1. / 3. * np.tan(phi) * (2. + 1. / cphi2)

    I5 = 1. / 16. * (0.5 * (11. * np.sin(phi) + 3. * np.sin(3. * phi)) / cphi2 / cphi2 + 6. * logs)

    return 5. / 7. * q0 ** 2 / np.pi * (I2 - 3. / 4. * q0 ** 2 * I4 + 0.3 * q0 ** 3 * I5)


@njit
def F2_2d(phi, q0):
    cphi2 = np.cos(phi) ** 2

    q02 = q0 * q0
    q03 = q02 * q0

    logs = np.log(np.tan(phi / 2. + np.pi / 4.))

    I0 = phi
    I2 = np.tan(phi)
    I4 = 1. / 3. * np.tan(phi) * (2. + 1. / cphi2)

    I3 = 1. / 2. * (np.tan(phi) / np.cos(phi) + logs)
    I5 = 1. / 16. * (0.5 * (11. * np.sin(phi) + 3. * np.sin(3. * phi)) / cphi2 / cphi2 + 6. * logs)

    return 5. / 7. * q02 / np.pi * (
                2. * I2 - 2. * q0 * I3 + 3. / 4. * q02 * I4 - 1. / 10. * q03 * I5 - 1. / 10. / q02 * I0)


@njit
def F3_2d(phi):
    I0 = phi
    return 0.5 / np.pi * I0
