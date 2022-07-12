import math
from typing import Tuple

import numpy as np
from numba import cuda
from numba.core.registry import CPUDispatcher
from numpy import ndarray

from sarracen.interpolate.base_backend import BaseBackend


class GPUBackend(BaseBackend):

    @staticmethod
    def interpolate_2d_render(target: ndarray, x: ndarray, y: ndarray, mass: ndarray, rho: ndarray, h: ndarray,
                              weight_function: CPUDispatcher, kernel_radius: float, x_pixels: int, y_pixels: int,
                              x_min: float, x_max: float, y_min: float, y_max: float, exact) -> ndarray:
        return GPUBackend._fast_2d(target, 0, x, y, np.zeros(len(target)), mass, rho, h, weight_function, kernel_radius,
                                   x_pixels, y_pixels, x_min, x_max, y_min, y_max, 2)

    @staticmethod
    def interpolate_2d_render_vec(target_x: ndarray, target_y: ndarray, x: ndarray, y: ndarray, mass: ndarray,
                                  rho: ndarray, h: ndarray, weight_function: CPUDispatcher, kernel_radius: float,
                                  x_pixels: int, y_pixels: int, x_min: float, x_max: float, y_min: float,
                                  y_max: float) -> Tuple[ndarray, ndarray]:
        return (GPUBackend._fast_2d(target_x, 0, x, y, np.zeros(len(target_x)), mass, rho, h, weight_function,
                                    kernel_radius, x_pixels, y_pixels, x_min, x_max, y_min, y_max, 2),
                GPUBackend._fast_2d(target_y, 0, x, y, np.zeros(len(target_y)), mass, rho, h, weight_function,
                                    kernel_radius, x_pixels, y_pixels, x_min, x_max, y_min, y_max, 2))

    @staticmethod
    def interpolate_2d_cross(target: ndarray, x: ndarray, y: ndarray, mass: ndarray, rho: ndarray, h: ndarray,
                             weight_function: CPUDispatcher, kernel_radius: float, pixels: int, x1: float, x2: float,
                             y1: float, y2: float) -> ndarray:
        return GPUBackend._fast_2d_cross(target, x, y, mass, rho, h, weight_function, kernel_radius, pixels, x1, x2,
                                             y1, y2)

    @staticmethod
    def interpolate_3d_projection(target: ndarray, x: ndarray, y: ndarray, mass: ndarray, rho: ndarray, h: ndarray,
                                  weight_function: CPUDispatcher, kernel_radius: float, x_pixels: int, y_pixels: int,
                                  x_min: float, x_max: float, y_min: float, y_max: float) -> ndarray:
        return GPUBackend._fast_2d(target, 0, x, y, np.zeros(len(target)), mass, rho, h, weight_function, kernel_radius,
                                   x_pixels, y_pixels, x_min, x_max, y_min, y_max, 2)

    @staticmethod
    def interpolate_3d_projection_vec(target_x: ndarray, target_y: ndarray, x: ndarray, y: ndarray, mass: ndarray,
                                      rho: ndarray, h: ndarray, weight_function: CPUDispatcher, kernel_radius: float,
                                      x_pixels: int, y_pixels: int, x_min: float, x_max: float, y_min: float,
                                      y_max: float) -> Tuple[ndarray, ndarray]:
        return (GPUBackend._fast_2d(target_x, 0, x, y, np.zeros(len(target_x)), mass, rho, h, weight_function,
                                    kernel_radius, x_pixels, y_pixels, x_min, x_max, y_min, y_max, 2),
                GPUBackend._fast_2d(target_y, 0, x, y, np.zeros(len(target_y)), mass, rho, h, weight_function,
                                    kernel_radius, x_pixels, y_pixels, x_min, x_max, y_min, y_max, 2))

    @staticmethod
    def interpolate_3d_cross(target: ndarray, z_slice: float, x: ndarray, y: ndarray, z: ndarray, mass: ndarray,
                             rho: ndarray, h: ndarray, weight_function: CPUDispatcher, kernel_radius: float,
                             x_pixels: int, y_pixels: int, x_min: float, x_max: float, y_min: float,
                             y_max: float) -> ndarray:
        return GPUBackend._fast_2d(target, z_slice, x, y, z, mass, rho, h, weight_function, kernel_radius, x_pixels,
                                   y_pixels, x_min, x_max, y_min, y_max, 3)

    @staticmethod
    def interpolate_3d_cross_vec(target_x: ndarray, target_y: ndarray, z_slice: float, x: ndarray, y: ndarray,
                                 z: ndarray, mass: ndarray, rho: ndarray, h: ndarray, weight_function: CPUDispatcher,
                                 kernel_radius: float, x_pixels: int, y_pixels: int, x_min: float, x_max: float,
                                 y_min: float, y_max: float) -> Tuple[ndarray, ndarray]:
        return (GPUBackend._fast_2d(target_x, z_slice, x, y, z, mass, rho, h, weight_function, kernel_radius, x_pixels,
                                    y_pixels, x_min, x_max, y_min, y_max, 3),
                GPUBackend._fast_2d(target_y, z_slice, x, y, z, mass, rho, h, weight_function, kernel_radius, x_pixels,
                                    y_pixels, x_min, x_max, y_min, y_max, 3))

    # For the GPU, the numba code is compiled using a factory function approach. This is required
    # since a CUDA numba kernel cannot easily take weight_function as an argument.
    @staticmethod
    def _fast_2d(target, z_slice, x_data, y_data, z_data, mass_data, rho_data, h_data, weight_function,
                     kernel_radius, x_pixels, y_pixels, x_min, x_max, y_min, y_max, n_dims):
        # Underlying GPU numba-compiled code for interpolation to a 2D grid. Used in interpolation of 2D data,
        # and column integration / cross-sections of 3D data.
        @cuda.jit(fastmath=True)
        def _2d_func(target, z_slice, x_data, y_data, z_data, mass_data, rho_data, h_data, kernel_radius,
                     x_pixels, y_pixels, x_min, x_max, y_min, y_max, n_dims, image):
            pixwidthx = (x_max - x_min) / x_pixels
            pixwidthy = (y_max - y_min) / y_pixels

            i = cuda.grid(1)
            if i < len(target):
                if not n_dims == 2:
                    dz = np.float64(z_slice) - z_data[i]
                else:
                    dz = 0

                term = (target[i] * mass_data[i] / (rho_data[i] * h_data[i] ** n_dims))

                if abs(dz) >= kernel_radius * h_data[i]:
                    return

                # determine maximum and minimum pixels that this particle contributes to
                ipixmin = round((x_data[i] - kernel_radius * h_data[i] - x_min) / pixwidthx)
                jpixmin = round((y_data[i] - kernel_radius * h_data[i] - y_min) / pixwidthy)
                ipixmax = round((x_data[i] + kernel_radius * h_data[i] - x_min) / pixwidthx)
                jpixmax = round((y_data[i] + kernel_radius * h_data[i] - y_min) / pixwidthy)

                if ipixmax < 0 or ipixmin > x_pixels or jpixmax < 0 or jpixmin > y_pixels:
                    return
                if ipixmin < 0:
                    ipixmin = 0
                if ipixmax > x_pixels:
                    ipixmax = x_pixels
                if jpixmin < 0:
                    jpixmin = 0
                if jpixmax > y_pixels:
                    jpixmax = y_pixels

                # calculate contributions to all nearby pixels
                for jpix in range(jpixmax - jpixmin):
                    for ipix in range(ipixmax - ipixmin):
                        # determine difference in the x-direction
                        xpix = x_min + ((ipix + ipixmin) + 0.5) * pixwidthx
                        dx = xpix - x_data[i]
                        dx2 = dx * dx * (1 / (h_data[i] ** 2))

                        # determine difference in the y-direction
                        ypix = y_min + ((jpix + jpixmin) + 0.5) * pixwidthy
                        dy = ypix - y_data[i]
                        dy2 = dy * dy * (1 / (h_data[i] ** 2))

                        dz2 = ((dz ** 2) * (1 / h_data[i] ** 2))

                        # calculate contributions at pixels i, j due to particle at x, y
                        q = math.sqrt(dx2 + dy2 + dz2)

                        # add contribution to image
                        if q < kernel_radius:
                            # atomic add protects the summation against race conditions.
                            wab = weight_function(q, n_dims)
                            cuda.atomic.add(image, (jpix + jpixmin, ipix + ipixmin), term * wab)

        threadsperblock = 32
        blockspergrid = (target.size + (threadsperblock - 1)) // threadsperblock

        # transfer relevant data to the GPU
        d_target = cuda.to_device(target)
        d_x = cuda.to_device(x_data)
        d_y = cuda.to_device(y_data)
        d_z = cuda.to_device(z_data)
        d_m = cuda.to_device(mass_data)
        d_rho = cuda.to_device(rho_data)
        d_h = cuda.to_device(h_data)
        # CUDA kernels have no return values, so the image data must be
        # allocated on the device beforehand.
        d_image = cuda.to_device(np.zeros((y_pixels, x_pixels)))

        # execute the newly compiled CUDA kernel.
        _2d_func[blockspergrid, threadsperblock](d_target, z_slice, d_x, d_y, d_z, d_m, d_rho, d_h, kernel_radius,
                                                 x_pixels,
                                                 y_pixels, x_min, x_max, y_min, y_max, n_dims, d_image)

        return d_image.copy_to_host()

    # For the GPU, the numba code is compiled using a factory function approach. This is required
    # since a CUDA numba kernel cannot easily take weight_function as an argument.
    @staticmethod
    def _fast_2d_cross(target, x_data, y_data, mass_data, rho_data, h_data, weight_function, kernel_radius, pixels,
                           x1, x2, y1, y2):
        # determine the slope of the cross-section line
        gradient = 0
        if not x2 - x1 == 0:
            gradient = (y2 - y1) / (x2 - x1)
        yint = y2 - gradient * x2

        # determine the fraction of the line that one pixel represents
        xlength = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        pixwidth = xlength / pixels
        xpixwidth = (x2 - x1) / pixels
        aa = 1 + gradient ** 2

        # Underlying GPU numba-compiled code for 2D->1D cross-sections
        @cuda.jit(fastmath=True)
        def _2d_func(target, x_data, y_data, mass_data, rho_data, h_data, kernel_radius, pixels, x1, x2, y1, y2, image):
            i = cuda.grid(1)
            if i < target.size:
                term = target[i] * mass_data[i] / (rho_data[i] * h_data[i] ** 2)

                # the intersections between the line and a particle's 'smoothing circle' are
                # found by solving a quadratic equation with the below values of a, b, and c.
                # if the determinant is negative, the particle does not contribute to the
                # cross-section, and can be removed.
                bb = 2 * gradient * (yint - y_data[i]) - 2 * x_data[i]
                cc = x_data[i] ** 2 + y_data[i] ** 2 - 2 * yint * y_data[i] + yint ** 2 - (
                            kernel_radius * h_data[i]) ** 2
                det = bb ** 2 - 4 * aa * cc

                # create a filter for particles that do not contribute to the cross-section
                if det < 0:
                    return

                det = math.sqrt(det)

                # the starting and ending x coordinates of the lines intersections with a particle's smoothing circle
                xstart = min(max(x1, (-bb - det) / (2 * aa)), x2)
                xend = min(max(x1, (-bb + det) / (2 * aa)), x2)

                # the start and end distances which lie within a particle's smoothing circle.
                rstart = math.sqrt((xstart - x1) ** 2 + ((gradient * xstart + yint) - y1) ** 2)
                rend = math.sqrt((xend - x1) ** 2 + (((gradient * xend + yint) - y1) ** 2))

                # the maximum and minimum pixels that each particle contributes to.
                ipixmin = min(max(0, round(rstart / pixwidth)), pixels)
                ipixmax = min(max(0, round(rend / pixwidth)), pixels)

                # iterate through all affected pixels
                for ipix in range(ipixmin, ipixmax):
                    # determine contributions to all affected pixels for this particle
                    xpix = x1 + (ipix + 0.5) * xpixwidth
                    ypix = gradient * xpix + yint
                    dy = ypix - y_data[i]
                    dx = xpix - x_data[i]

                    q2 = (dx * dx + dy * dy) * (1 / (h_data[i] * h_data[i]))
                    wab = weight_function(math.sqrt(q2), 2)

                    # add contributions to output total.
                    cuda.atomic.add(image, ipix, wab * term)

        threadsperblock = 32
        blockspergrid = (target.size + (threadsperblock - 1)) // threadsperblock

        # transfer relevant data to the GPU
        d_target = cuda.to_device(target)
        d_x = cuda.to_device(x_data)
        d_y = cuda.to_device(y_data)
        d_m = cuda.to_device(mass_data)
        d_rho = cuda.to_device(rho_data)
        d_h = cuda.to_device(h_data)

        # CUDA kernels have no return values, so the image data must be
        # allocated on the device beforehand.
        d_image = cuda.to_device(np.zeros(pixels))

        # execute the newly compiled GPU kernel
        _2d_func[blockspergrid, threadsperblock](d_target, d_x, d_y, d_m, d_rho, d_h, kernel_radius, pixels, x1, x2, y1,
                                                 y2, d_image)

        return d_image.copy_to_host()