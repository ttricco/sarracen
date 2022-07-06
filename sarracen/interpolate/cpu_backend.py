from typing import Tuple

from numba import njit, prange
from numba.core.registry import CPUDispatcher
from numpy import ndarray
import numpy as np

from sarracen.interpolate.base_backend import BaseBackend


class CPUBackend(BaseBackend):

    @staticmethod
    def interpolate_2d_render(target: ndarray, x: ndarray, y: ndarray, mass: ndarray, rho: ndarray, h: ndarray,
                              weight_function: CPUDispatcher, kernel_radius: float, x_pixels: int, y_pixels: int,
                              x_min: float, x_max: float, y_min: float, y_max: float) -> ndarray:
        return CPUBackend._fast_2d(target, 0, x, y, np.zeros(len(target)), mass, rho, h, weight_function, kernel_radius,
                                   x_pixels, y_pixels, x_min, x_max, y_min, y_max, 2)

    @staticmethod
    def interpolate_2d_render_vec(target_x: ndarray, target_y: ndarray, x: ndarray, y: ndarray, mass: ndarray,
                                  rho: ndarray, h: ndarray, weight_function: CPUDispatcher, kernel_radius: float,
                                  x_pixels: int, y_pixels: int, x_min: float, x_max: float, y_min: float,
                                  y_max: float) -> Tuple[ndarray, ndarray]:
        return (CPUBackend._fast_2d(target_x, 0, x, y, np.zeros(len(target_x)), mass, rho, h, weight_function,
                                    kernel_radius, x_pixels, y_pixels, x_min, x_max, y_min, y_max, 2),
                CPUBackend._fast_2d(target_y, 0, x, y, np.zeros(len(target_y)), mass, rho, h, weight_function,
                                    kernel_radius, x_pixels, y_pixels, x_min, x_max, y_min, y_max, 2))

    @staticmethod
    def interpolate_2d_cross(target: ndarray, x: ndarray, y: ndarray, mass: ndarray, rho: ndarray, h: ndarray,
                             weight_function: CPUDispatcher, kernel_radius: float, pixels: int, x1: float, x2: float,
                             y1: float, y2: float) -> ndarray:
        return CPUBackend._fast_2d_cross_cpu(target, x, y, mass, rho, h, weight_function, kernel_radius, pixels, x1, x2,
                                             y1, y2)

    @staticmethod
    def interpolate_3d_projection(target: ndarray, x: ndarray, y: ndarray, mass: ndarray, rho: ndarray, h: ndarray,
                                  weight_function: CPUDispatcher, kernel_radius: float, x_pixels: int, y_pixels: int,
                                  x_min: float, x_max: float, y_min: float, y_max: float) -> ndarray:
        return CPUBackend._fast_2d(target, 0, x, y, np.zeros(len(target)), mass, rho, h, weight_function, kernel_radius,
                                   x_pixels, y_pixels, x_min, x_max, y_min, y_max, 2)

    @staticmethod
    def interpolate_3d_projection_vec(target_x: ndarray, target_y: ndarray, x: ndarray, y: ndarray, mass: ndarray,
                                      rho: ndarray, h: ndarray, weight_function: CPUDispatcher, kernel_radius: float,
                                      x_pixels: int, y_pixels: int, x_min: float, x_max: float, y_min: float,
                                      y_max: float) -> Tuple[ndarray, ndarray]:
        return (CPUBackend._fast_2d(target_x, 0, x, y, np.zeros(len(target_x)), mass, rho, h, weight_function,
                                    kernel_radius, x_pixels, y_pixels, x_min, x_max, y_min, y_max, 2),
                CPUBackend._fast_2d(target_y, 0, x, y, np.zeros(len(target_y)), mass, rho, h, weight_function,
                                    kernel_radius, x_pixels, y_pixels, x_min, x_max, y_min, y_max, 2))

    @staticmethod
    def interpolate_3d_cross(target: ndarray, z_slice: float, x: ndarray, y: ndarray, z: ndarray, mass: ndarray,
                             rho: ndarray, h: ndarray, weight_function: CPUDispatcher, kernel_radius: float,
                             x_pixels: int, y_pixels: int, x_min: float, x_max: float, y_min: float,
                             y_max: float) -> ndarray:
        return CPUBackend._fast_2d(target, z_slice, x, y, z, mass, rho, h, weight_function, kernel_radius, x_pixels,
                                   y_pixels, x_min, x_max, y_min, y_max, 3)

    @staticmethod
    def interpolate_3d_cross_vec(target_x: ndarray, target_y: ndarray, z_slice: float, x: ndarray, y: ndarray,
                                 z: ndarray, mass: ndarray, rho: ndarray, h: ndarray, weight_function: CPUDispatcher,
                                 kernel_radius: float, x_pixels: int, y_pixels: int, x_min: float, x_max: float,
                                 y_min: float, y_max: float) -> Tuple[ndarray, ndarray]:
        return (CPUBackend._fast_2d(target_x, z_slice, x, y, z, mass, rho, h, weight_function, kernel_radius, x_pixels,
                                    y_pixels, x_min, x_max, y_min, y_max, 3),
                CPUBackend._fast_2d(target_y, z_slice, x, y, z, mass, rho, h, weight_function, kernel_radius, x_pixels,
                                    y_pixels, x_min, x_max, y_min, y_max, 3))

    # Underlying CPU numba-compiled code for interpolation to a 2D grid. Used in interpolation of 2D data,
    # and column integration / cross-sections of 3D data.
    @staticmethod
    @njit(parallel=True, fastmath=True)
    def _fast_2d(target, z_slice, x_data, y_data, z_data, mass_data, rho_data, h_data, weight_function,
                 kernel_radius, x_pixels, y_pixels, x_min, x_max, y_min, y_max, n_dims):
        image = np.zeros((y_pixels, x_pixels))
        pixwidthx = (x_max - x_min) / x_pixels
        pixwidthy = (y_max - y_min) / y_pixels
        if not n_dims == 2:
            dz = np.float64(z_slice) - z_data
        else:
            dz = np.zeros(target.size)

        term = (target * mass_data / (rho_data * h_data ** n_dims))

        # iterate through the indexes of non-filtered particles
        for i in prange(term.size):
            if np.abs(dz[i]) >= kernel_radius * h_data[i]:
                continue

            # determine maximum and minimum pixels that this particle contributes to
            ipixmin = int(np.rint((x_data[i] - kernel_radius * h_data[i] - x_min) / pixwidthx))
            jpixmin = int(np.rint((y_data[i] - kernel_radius * h_data[i] - y_min) / pixwidthy))
            ipixmax = int(np.rint((x_data[i] + kernel_radius * h_data[i] - x_min) / pixwidthx))
            jpixmax = int(np.rint((y_data[i] + kernel_radius * h_data[i] - y_min) / pixwidthy))

            if ipixmax < 0 or ipixmin > x_pixels or jpixmax < 0 or jpixmin > y_pixels:
                continue
            if ipixmin < 0:
                ipixmin = 0
            if ipixmax > x_pixels:
                ipixmax = x_pixels
            if jpixmin < 0:
                jpixmin = 0
            if jpixmax > y_pixels:
                jpixmax = y_pixels

            # precalculate differences in the x-direction (optimization)
            dx2i = ((x_min + (np.arange(ipixmin, ipixmax) + 0.5) * pixwidthx - x_data[i]) ** 2) \
                   * (1 / (h_data[i] ** 2)) + ((dz[i] ** 2) * (1 / h_data[i] ** 2))

            # determine differences in the y-direction
            ypix = y_min + (np.arange(jpixmin, jpixmax) + 0.5) * pixwidthy
            dy = ypix - y_data[i]
            dy2 = dy * dy * (1 / (h_data[i] ** 2))

            # calculate contributions at pixels i, j due to particle at x, y
            q2 = dx2i + dy2.reshape(len(dy2), 1)

            for jpix in prange(jpixmax - jpixmin):
                for ipix in prange(ipixmax - ipixmin):
                    if np.sqrt(q2[jpix][ipix]) > kernel_radius:
                        continue
                    wab = weight_function(np.sqrt(q2[jpix][ipix]), n_dims)
                    image[jpix + jpixmin, ipix + ipixmin] += term[i] * wab

        return image

    # Underlying CPU numba-compiled code for 2D->1D cross-sections.
    @staticmethod
    @njit(parallel=True, fastmath=True)
    def _fast_2d_cross_cpu(target, x_data, y_data, mass_data, rho_data, h_data, weight_function, kernel_radius, pixels,
                           x1, x2, y1, y2):
        # determine the slope of the cross-section line
        gradient = 0
        if not x2 - x1 == 0:
            gradient = (y2 - y1) / (x2 - x1)
        yint = y2 - gradient * x2

        # determine the fraction of the line that one pixel represents
        xlength = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        pixwidth = xlength / pixels
        xpixwidth = (x2 - x1) / pixels

        term = target * mass_data / (rho_data * h_data ** 2)

        # the intersections between the line and a particle's 'smoothing circle' are
        # found by solving a quadratic equation with the below values of a, b, and c.
        # if the determinant is negative, the particle does not contribute to the
        # cross-section, and can be removed.
        aa = 1 + gradient ** 2
        bb = 2 * gradient * (yint - y_data) - 2 * x_data
        cc = x_data ** 2 + y_data ** 2 - 2 * yint * y_data + yint ** 2 - (kernel_radius * h_data) ** 2
        det = bb ** 2 - 4 * aa * cc

        # create a filter for particles that do not contribute to the cross-section
        filter_det = det >= 0
        det = np.sqrt(det)
        cc = None

        output = np.zeros(pixels)

        # the starting and ending x coordinates of the lines intersections with a particle's smoothing circle
        xstart = ((-bb[filter_det] - det[filter_det]) / (2 * aa)).clip(a_min=x1, a_max=x2)
        xend = ((-bb[filter_det] + det[filter_det]) / (2 * aa)).clip(a_min=x1, a_max=x2)
        bb, det = None, None

        # the start and end distances which lie within a particle's smoothing circle.
        rstart = np.sqrt((xstart - x1) ** 2 + ((gradient * xstart + yint) - y1) ** 2)
        rend = np.sqrt((xend - x1) ** 2 + (((gradient * xend + yint) - y1) ** 2))
        xstart, xend = None, None

        # the maximum and minimum pixels that each particle contributes to.
        ipixmin = np.rint(rstart / pixwidth).clip(a_min=0, a_max=pixels)
        ipixmax = np.rint(rend / pixwidth).clip(a_min=0, a_max=pixels)
        rstart, rend = None, None

        # iterate through the indices of all non-filtered particles
        for i in prange(len(x_data[filter_det])):
            # determine contributions to all affected pixels for this particle
            xpix = x1 + (np.arange(int(ipixmin[i]), int(ipixmax[i])) + 0.5) * xpixwidth
            ypix = gradient * xpix + yint
            dy = ypix - y_data[filter_det][i]
            dx = xpix - x_data[filter_det][i]

            q2 = (dx * dx + dy * dy) * (1 / (h_data[filter_det][i] * h_data[filter_det][i]))
            wab = weight_function(np.sqrt(q2), 2)

            # add contributions to output total
            for ipix in prange(int(ipixmax[i]) - int(ipixmin[i])):
                output[ipix + int(ipixmin[i])] += term[filter_det][i] * wab[ipix]

        return output