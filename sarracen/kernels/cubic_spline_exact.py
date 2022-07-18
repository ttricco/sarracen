import math

from numba import njit


@njit
def line_int(r0, d1, d2, h):
    """ Calculate an exact 2D line integral over the cubic spline kernel.

    Used in exact calculation of a pixel surface integral in 2D.

    Parameters
    ----------
    r0: float
        Distance between the contributing particle and the line.
    d1, d2: float
        Distance from the endpoint of `r0` to each endpoint of the line.
    h: float
        Smoothing length of the contributing particle.

    Returns
    -------
    float: The exact value of this line integral.
    """
    if r0 == 0:
        return 0
    elif r0 > 0:
        result = 1
        ar0 = r0
    else:
        result = -1
        ar0 = -r0

    q0 = ar0 / h

    # Determine the angle between q0 and the endpoints of the line, relative to the contributing particle.
    phi1 = math.atan(abs(d1) / ar0)
    phi2 = math.atan(abs(d2) / ar0)

    if d1 * d2 >= 0:
        # Both line endpoints are on opposite sides of r0.
        result = result * (_full_2d_mod(phi1, q0) + _full_2d_mod(phi2, q0))
    elif abs(d1) < abs(d2):
        # Both line endpoints are on the same side of r0, with d2 having a larger magnitude.
        result = result * (_full_2d_mod(phi2, q0) - _full_2d_mod(phi1, q0))
    else:
        # Both line endpoints are on the same side of r0, with d1 having a larger magnitude.
        result = result * (_full_2d_mod(phi1, q0) - _full_2d_mod(phi2, q0))

    return result


@njit
def _full_2d_mod(phi, q0):
    """ Calculate an exact 2D line integral over the cubic spline kernel.

    Assumes that one endpoint of the line is at the end of `q0`. Used in pint.

    Parameters
    ----------
    phi: float
        Angle between `q0` and the endpoint of the line, relative to the contributing particle.
    q0: float
        The distance between the contributing particle and the line, scaled by the smoothing length of the particle.

    Returns
    -------
    float: The exact value of this line integral.
    """
    if q0 <= 1.0:
        # At least part of the line lies within 0 < q <= 1
        q = q0 / math.cos(phi)

        if q <= 1.0:
            # The line lies entirely within 0 < q <= 1.
            return _f1_2d(phi, q0)
        elif q <= 2.0:
            # The line lies partly in 0 < q <= 1 and partly in 1 < q <= 2.

            # Angle between q0 and the line region endpoint within 0 < q <= 1, relative to the contributing particle.
            phi1 = math.acos(q0)
            return _f2_2d(phi, q0) - _f2_2d(phi1, q0) + _f1_2d(phi1, q0)
        else:
            # The line spans all three possible regions, 0 < q <= 1, 1 < q <= 2, and q > 2.

            # Angle between q0 and the line region endpoint within 0 < q <= 1, relative to the contributing particle.
            phi1 = math.acos(q0)
            # Angle between q0 and the line region endpoint within 1 < q <= 2, relative to the contributing particle.
            phi2 = math.acos(0.5 * q0)
            return _f3_2d(phi) - _f3_2d(phi2) + _f2_2d(phi2, q0) - _f2_2d(phi1, q0) + _f1_2d(phi1, q0)
    elif q0 <= 2.0:
        # No part of the line lies within 0 < q <= 1, but it does lie within 1 < q <= 2.
        q = q0 / math.cos(phi)

        if q <= 2.0:
            # The line lies entirely within 1 < q <= 2.
            return _f2_2d(phi, q0)
        else:
            # The line lies partly in 1 < q <= 2 and q > 2.

            # Angle between q0 and the line region endpoint within 1 < q <= 2, relative to the contributing particle.
            phi2 = math.acos(0.5 * q0)
            return _f3_2d(phi) - _f3_2d(phi2) + _f2_2d(phi2, q0)
    else:
        # The line lies entirely within q > 2.
        return _f3_2d(phi)


@njit
def _f1_2d(phi, q0):
    """ Calculate an exact 2D line integral over the cubic spline kernel.

    Assumes that one endpoint of the line is at the end of `q0`. Only valid for 0 < q <= 1.
    Used in _full_2d_mod.

    Parameters
    ----------
    phi: float
        Angle between `q0` and the endpoint of the line segment, relative to the contributing particle.
    q0: float
        The distance between the contributing particle and the line, scaled by the smoothing length of the particle.

    Returns
    -------
    float: The exact value of this line integral segment.
    """
    cphi2 = math.cos(phi) ** 2
    logs = math.log(math.tan(phi / 2. + math.pi / 4.))

    i2 = math.tan(phi)
    i4 = 1. / 3. * math.tan(phi) * (2. + 1. / cphi2)
    i5 = 1. / 16. * (0.5 * (11. * math.sin(phi) + 3. * math.sin(3. * phi)) / cphi2 / cphi2 + 6. * logs)

    return 5. / 7. * q0 ** 2 / math.pi * (i2 - 3. / 4. * q0 ** 2 * i4 + 0.3 * q0 ** 3 * i5)


@njit
def _f2_2d(phi, q0):
    """ Calculate an exact 2D line integral over the cubic spline kernel.

    Assumes that one endpoint of the line is at the end of `q0`. Only valid for 1 < q <= 2.
    Used in _full_2d_mod.

    Parameters
    ----------
    phi: float
        Angle between `q0` and the endpoint of the line segment, relative to the contributing particle.
    q0: float
        The distance between the contributing particle and the line, scaled by the smoothing length of the particle.

    Returns
    -------
    float: The exact value of this line integral segment.
    """
    cphi2 = math.cos(phi) ** 2

    q02 = q0 * q0
    q03 = q02 * q0

    logs = math.log(math.tan(phi / 2. + math.pi / 4.))

    i0 = phi
    i2 = math.tan(phi)
    i3 = 1. / 2. * (math.tan(phi) / math.cos(phi) + logs)
    i4 = 1. / 3. * math.tan(phi) * (2. + 1. / cphi2)
    i5 = 1. / 16. * (0.5 * (11. * math.sin(phi) + 3. * math.sin(3. * phi)) / cphi2 / cphi2 + 6. * logs)

    return 5. / 7. * q02 / math.pi * (
            2. * i2 - 2. * q0 * i3 + 3. / 4. * q02 * i4 - 1. / 10. * q03 * i5 - 1. / 10. / q02 * i0)


@njit
def _f3_2d(phi):
    """ Calculate an exact 2D line integral over the cubic spline kernel.

    Assumes that one endpoint of the line is at the end of `q0`. Only valid for q > 2.
    Used in _full_2d_mod.

    Parameters
    ----------
    phi: float
        Angle pointing towards the endpoint of the line segment, relative to the contributing particle.

    Returns
    -------
    float: The exact value of this line integral segment.
    """
    return 0.5 / math.pi * phi


@njit
def surface_int(r0, x1, y1, x2, y2, wx, wy, h):
    """ Calculate an exact 3D surface integral over the cubic spline kernel.

    Used to exactly calculating the contribution of a particle to a pixel's volume in 3D space.

    Parameters
    ----------
    r0: float
        Distance between the contributing particle and the target surface.
    x1, y1, x2, y2: float
        Upper and lower bounds of the target surface, relative to r0.
    wx, wy: float
        The size of a single pixel.
    h: float
        The smoothing length of the contributing particle.

    Returns
    -------
    float: The exact value of this surface integral.
    """
    result = 0.0
    dx = x2 - x1
    dy = y2 - y1

    # Calculate the exact value of this surface by summing the comprising line integrals.

    # Bottom boundary
    r1 = 0.5 * wy + dy
    d1 = 0.5 * wx - dx
    d2 = 0.5 * wx + dx
    result = result + _line_int3d(r0, r1, d1, d2, h)

    # Top boundary
    r1 = 0.5 * wy - dy
    d1 = 0.5 * wx + dx
    d2 = 0.5 * wx - dx
    result = result + _line_int3d(r0, r1, d1, d2, h)

    # Right boundary
    r1 = 0.5 * wx + dx
    d1 = 0.5 * wy + dy
    d2 = 0.5 * wy - dy
    result = result + _line_int3d(r0, r1, d1, d2, h)

    # Left boundary
    r1 = 0.5 * wx - dx
    d1 = 0.5 * wy - dy
    d2 = 0.5 * wy + dy
    result = result + _line_int3d(r0, r1, d1, d2, h)

    return result


@njit
def _line_int3d(r0, r1, d1, d2, h):
    """ Calculate an exact 3D line integral over the cubic spline kernel.

    Used in wallint.

    Parameters
    ----------
    r0: float
        Distance between the contributing particle and the target surface.
    r1: float
        Distance between the endpoint of `r0` and the line.
    d1, d2: float
        Distance from the endpoint of `r1` to each endpoint of the line.
    h: float
        Smoothing length of the contributing particle.

    Returns
    -------
    float: The exact value of this line integral.
    """
    if abs(r0) == 0:
        return 0
    if r0 > 0:
        result = 1
        ar0 = r0
    else:
        result = -1
        ar0 = -r0

    if r1 > 0.:
        ar1 = r1
    else:
        result = -result
        ar1 = -r1

    # Split this line integral into two separate line integrals, where one end point is at the endpoint of r1,
    # and the other end point is d1 or d2 respectively.
    int1 = _full_integral_3d(d1, ar0, ar1, h)
    int2 = _full_integral_3d(d2, ar0, ar1, h)

    if int1 < 0:
        int1 = 0
    if int2 < 0:
        int2 = 0

    if d1 * d2 >= 0:
        # Both line endpoints are on opposite sides of r1.
        result = result * (int1 + int2)
        if int1 + int2 < 0:
            print('Error: int1 + int2 < 0')
    elif abs(d1) < abs(d2):
        # Both line endpoints are on the same side of r1, with d2 having a larger magnitude.
        result = result * (int2 - int1)
        if int2 - int1 < 0:
            print('Error: int2 - int1 < 0: ', int1, int2, '(', d1, d2, ')')
    else:
        # Both line endpoints are on the same side of r1, with d1 having a larger magnitude.
        result = result * (int1 - int2)
        if int1 - int2 < 0:
            print('Error: int1 - int2 < 0: ', int1, int2, '(', d1, d2, ')')

    return result


@njit
def _full_integral_3d(d, r0, r1, h):
    """ Calculate an exact 3D line integral over the cubic spline kernel.

    Assumes that one endpoint of the line is at the end of `r1`. Used in _pint3d.

    Parameters
    ----------
    d: float
        Distance of the line.
    r0: float
        Distance between the target surface and the contributing particle.
    r1: float
        Distance between the end of `r0` and the line.
    h: float
        The smoothing length of the contributing particle.

    Returns
    -------
    float: The exact value of this line integral.
    """
    r0h = r0 / h
    # Angle between the end of the line and the end of r1, relative to the start of r1.
    phi = math.atan(abs(d) / r1)

    if abs(r0h) == 0 or abs(r1 / h) == 0 or abs(phi) == 0:
        return 0

    h2 = h * h
    r03 = r0 * r0 * r0
    r0h2 = r0h * r0h
    r0h3 = r0h2 * r0h
    r0h_2 = 1. / r0h2
    r0h_3 = 1. / r0h3

    b1 = None
    b2 = None
    if r0 > 2.0 * h:
        # The entire surface lies outside r > 2h.
        b3 = 0.25 * h2 * h
    elif r0 > h:
        # A part of the surface lies in the region h < r <= 2h.
        b3 = 0.25 * r03 * (-4. / 3. + r0h - 0.3 * r0h2 + 1. / 30. * r0h3 - 1. / 15. * r0h_3 + 8. / 5. * r0h_2)
        b2 = 0.25 * r03 * (-4. / 3. + r0h - 0.3 * r0h2 + 1. / 30. * r0h3 - 1. / 15. * r0h_3)
    else:
        # A part of the surface lies in the region 0 < r <= h.
        b3 = 0.25 * r03 * (-2. / 3. + 0.3 * r0h2 - 0.1 * r0h3 + 7. / 5. * r0h_2)
        b2 = 0.25 * r03 * (-2. / 3. + 0.3 * r0h2 - 0.1 * r0h3 - 1. / 5. * r0h_2)
        b1 = 0.25 * r03 * (-2. / 3. + 0.3 * r0h2 - 0.1 * r0h3)

    a = r1 / r0
    a2 = a * a

    # Squared distance between the contributing particle and the end of r1.
    linedist2 = r0 * r0 + r1 * r1
    # Distance between the end of r1 and the end of the line.
    r_ = r1 / math.cos(phi)
    # Squared distance between the contributing particle and the end of the line.
    r2 = (r0 * r0 + r_ * r_)

    d2 = 0.0
    d3 = 0.0

    if linedist2 < h2:
        # A portion of the line lies within 0 < r < h.
        i = get_I_terms(r1 / math.sqrt(h2 - r0 * r0), a2, a)

        d2 = -1. / 6. * i[2] + 0.25 * r0h * i[3] - 0.15 * r0h2 * i[4] + 1. / 30. * r0h3 * i[5] - 1. / 60. * r0h_3\
             * i[1] + (b1 - b2) / r03 * i[0]
    if linedist2 < 4. * h2:
        # A portion of the line lies within 0 < r < 2h.
        i = get_I_terms(r1 / math.sqrt(4.0 * h2 - r0 * r0), a2, a)

        d3 = 1. / 3. * i[2] - 0.25 * r0h * i[3] + 3. / 40. * r0h2 * i[4] - 1. / 120. * r0h3 * i[5] + 4. / 15. * r0h_3\
             * i[1] + (b2 - b3) / r03 * i[0] + d2

    i = get_I_terms(math.cos(phi), a2, a)

    if r2 <= h2:
        # The entire line lies within 0 < r <= h.
        return r0h3 / math.pi * (1. / 6. * i[2] - 3. / 40. * r0h2 * i[4] + 1. / 40. * r0h3 * i[5] + b1 / r03 * i[0])
    elif r2 <= 4. * h2:
        # The entire line lies within 0 < r <= 2h.
        return r0h3 / math.pi * (0.25 * (4. / 3. * i[2] - (r0 / h) * i[3] + 0.3 * r0h2 * i[4] - 1. / 30. * r0h3 * i[5] +
                                         1. / 15. * r0h_3 * i[1]) + b2 / r03 * i[0] + d2)
    else:
        # The line lies in all possible regions, 0 < r <= h, 0 < r <= 2h, and r > 2h.
        return r0h3 / math.pi * (-0.25 * r0h_3 * i[1] + b3 / r03 * i[0] + d3)


@njit
def get_I_terms(cosp, a2, a):
    """Calculate I constants for calculations in _full_integral_3d"""
    cosp2 = cosp * cosp
    p = math.acos(cosp)
    tamath = math.sqrt(1. - cosp2) / cosp

    mu2_1 = 1. / (1. + cosp2 / a2)
    I0 = p
    I_2 = p + a2 * tamath
    I_4 = p + 2. * a2 * tamath + 1. / 3. * a2 * a2 * tamath * (2. + 1. / cosp2)

    u2 = (1. - cosp2) * mu2_1
    u = math.sqrt(u2)
    logs = math.log((1. + u) / (1. - u))
    I1 = math.atan2(u, a)

    fac = 1. / (1. - u2)
    I_1 = 0.5 * a * logs + I1
    I_3 = I_1 + a * 0.25 * (1. + a2) * (2. * u * fac + logs)
    I_5 = I_3 + a * (1. + a2) * (1. + a2) / 16. * ((10. * u - 6. * u * u2) * fac * fac + 3. * logs)

    return I0, I1, I_2, I_3, I_4, I_5
