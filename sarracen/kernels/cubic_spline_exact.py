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


@njit
def wallint(r0, xp, yp, xc, yc, pixwidthx, pixwidthy, hi):
    result = 0.0
    dx = xc - xp
    dy = yc - yp
    h = hi

    R_0 = 0.5 * pixwidthy + dy
    d1 = 0.5 * pixwidthx - dx
    d2 = 0.5 * pixwidthx + dx
    result = result + pint3D(r0, R_0, d1, d2, h)

    R_0 = 0.5 * pixwidthy - dy
    d1 = 0.5 * pixwidthx + dx
    d2 = 0.5 * pixwidthx - dx
    result = result + pint3D(r0, R_0, d1, d2, h)

    R_0 = 0.5 * pixwidthx + dx
    d1 = 0.5 * pixwidthy + dy
    d2 = 0.5 * pixwidthy - dy
    result = result + pint3D(r0, R_0, d1, d2, h)

    R_0 = 0.5 * pixwidthx - dx
    d1 = 0.5 * pixwidthy - dy
    d2 = 0.5 * pixwidthy + dy
    result = result + pint3D(r0, R_0, d1, d2, h)

    return result


@njit
def pint3D(r0, R_0, d1, d2, hi):
    if abs(r0) == 0:
        return 0

    if r0 > 0.:
        result = 1
        ar0 = r0
    else:
        result = -1
        ar0 = -r0

    if R_0 > 0.:
        aR_0 = R_0
    else:
        result = -result
        aR_0 = -R_0

    int1 = full_integral_3D(d1, ar0, aR_0, hi)
    int2 = full_integral_3D(d2, ar0, aR_0, hi)

    if int1 < 0:
        int1 = 0
    if int2 < 0:
        int2 = 0

    if d1 * d2 >= 0:
        result = result * (int1 + int2)
        if int1 + int2 < 0:
            print('Error: int1 + int2 < 0')
    elif abs(d1) < abs(d2):
        result = result * (int2 - int1)
        if int2 - int1 < 0:
            print('Error: int2 - int1 < 0: ', int1, int2, '(', d1, d2, ')')
    else:
        result = result * (int1 - int2)
        if int1 - int2 < 0:
            print('Error: int1 - int2 < 0: ', int1, int2, '(', d1, d2, ')')

    return result


@njit
def full_integral_3D(d, r0, R_0, h):
    r0h = r0 / h
    tanphi = abs(d) / R_0
    phi = np.arctan(tanphi)

    if abs(r0h) == 0 or abs(R_0 / h) == 0 or abs(phi) == 0:
        return 0

    h2 = h * h
    r03 = r0 * r0 * r0
    r0h2 = r0h * r0h
    r0h3 = r0h2 * r0h
    r0h_2 = 1. / r0h2
    r0h_3 = 1. / r0h3

    B1 = None
    B2 = None
    if r0 > 2.0 * h:
        B3 = 0.25 * h2 * h
    elif r0 > h:
        B3 = 0.25 * r03 * (-4. / 3. + (r0h) - 0.3 * r0h2 + 1. / 30. * r0h3 - 1. / 15. * r0h_3 + 8. / 5. * r0h_2)
        B2 = 0.25 * r03 * (-4. / 3. + (r0h) - 0.3 * r0h2 + 1. / 30. * r0h3 - 1. / 15. * r0h_3)
    else:
        B3 = 0.25 * r03 * (-2. / 3. + 0.3 * r0h2 - 0.1 * r0h3 + 7. / 5. * r0h_2)
        B2 = 0.25 * r03 * (-2. / 3. + 0.3 * r0h2 - 0.1 * r0h3 - 1. / 5. * r0h_2)
        B1 = 0.25 * r03 * (-2. / 3. + 0.3 * r0h2 - 0.1 * r0h3)

    a = R_0 / r0
    a2 = a * a

    linedist2 = (r0 * r0 + R_0 * R_0)
    cosphi = np.cos(phi)
    R_ = R_0 / cosphi
    r2 = (r0 * r0 + R_ * R_)

    D2 = 0.0
    D3 = 0.0

    if linedist2 < h2:
        cosp = R_0 / np.sqrt(h2 - r0 * r0)
        I0, I1, I_2, I_3, I_4, I_5 = get_I_terms(cosp, a2, a)

        D2 = -1. / 6. * I_2 + 0.25 * (r0h) * I_3 - 0.15 * r0h2 * I_4 + 1. / 30. * r0h3 * I_5 - 1. / 60. * r0h_3 * I1 + (
                B1 - B2) / r03 * I0
    if linedist2 < 4. * h2:
        cosp = R_0 / np.sqrt(4.0 * h2 - r0 * r0)
        I0, I1, I_2, I_3, I_4, I_5 = get_I_terms(cosp, a2, a)

        if B2 is None:
            print("Ouch!")

        D3 = 1. / 3. * I_2 - 0.25 * (
            r0h) * I_3 + 3. / 40. * r0h2 * I_4 - 1. / 120. * r0h3 * I_5 + 4. / 15. * r0h_3 * I1 + (
                     B2 - B3) / r03 * I0 + D2

    I0, I1, I_2, I_3, I_4, I_5 = get_I_terms(cosphi, a2, a, phi=phi, tanphi=tanphi)

    if r2 <= h2:
        return r0h3 / np.pi * (1. / 6. * I_2 - 3. / 40. * r0h2 * I_4 + 1. / 40. * r0h3 * I_5 + B1 / r03 * I0)
    elif r2 <= 4. * h2:
        return r0h3 / np.pi * (0.25 * (4. / 3. * I_2 - (
                r0 / h) * I_3 + 0.3 * r0h2 * I_4 - 1. / 30. * r0h3 * I_5 + 1. / 15. * r0h_3 * I1) + B2 / r03 * I0 + D2)
    else:
        return r0h3 / np.pi * (-0.25 * r0h_3 * I1 + B3 / r03 * I0 + D3)


@njit
def get_I_terms(cosp, a2, a, phi=None, tanphi=None):
    cosp2 = cosp * cosp
    if phi:
        p = phi
        tanp = tanphi
    else:
        p = np.arccos(cosp)
        tanp = np.sqrt(1. - cosp2) / cosp

    mu2_1 = 1. / (1. + cosp2 / a2)
    I0 = p
    I_2 = p + a2 * tanp
    I_4 = p + 2. * a2 * tanp + 1. / 3. * a2 * a2 * tanp * (2. + 1. / cosp2)

    u2 = (1. - cosp2) * mu2_1
    u = np.sqrt(u2)
    logs = np.log((1. + u) / (1. - u))
    I1 = np.arctan2(u, a)

    fac = 1. / (1. - u2)
    I_1 = 0.5 * a * logs + I1
    I_3 = I_1 + a * 0.25 * (1. + a2) * (2. * u * fac + logs)
    I_5 = I_3 + a * (1. + a2) * (1. + a2) / 16. * ((10. * u - 6. * u * u2) * fac * fac + 3. * logs)

    return I0, I1, I_2, I_3, I_4, I_5
