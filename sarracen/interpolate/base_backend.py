from typing import Tuple

from numba.core.registry import CPUDispatcher
from numpy import ndarray, zeros


class BaseBackend:
    """Backend implementation of SPH interpolation functions."""

    @staticmethod
    def interpolate_2d_render(x: ndarray, y: ndarray, weight: ndarray, h: ndarray, weight_function: CPUDispatcher,
                              kernel_radius: float, x_pixels: int, y_pixels: int, x_min: float, x_max: float,
                              y_min: float, y_max: float, exact: bool) -> ndarray:
        """ Interpolate 2D particle data to a 2D grid of pixels."""
        return zeros((y_pixels, x_pixels))

    @staticmethod
    def interpolate_2d_render_vec(x: ndarray, y: ndarray, weight_x: ndarray, weight_y: ndarray, h: ndarray,
                                  weight_function: CPUDispatcher, kernel_radius: float, x_pixels: int, y_pixels: int,
                                  x_min: float, x_max: float, y_min: float, y_max: float,
                                  exact: bool) -> Tuple[ndarray, ndarray]:
        """ Interpolate 2D particle vector data to a pair of 2D grids of pixels. """
        return zeros((y_pixels, x_pixels)), zeros((y_pixels, x_pixels))

    @staticmethod
    def interpolate_2d_line(x: ndarray, y: ndarray, weight: ndarray, h: ndarray, weight_function: CPUDispatcher,
                             kernel_radius: float, pixels: int, x1: float, x2: float, y1: float, y2: float) -> ndarray:
        """ Interpolate 2D particle data to a 1D cross-sectional line. """
        return zeros(pixels)

    @staticmethod
    def interpolate_3d_line(x: ndarray, y: ndarray, z: ndarray, weight: ndarray, h: ndarray,
                            weight_function: CPUDispatcher, kernel_radius: float, pixels: int, x1: float, x2: float,
                            y1: float, y2: float, z1: float, z2: float) -> ndarray:
        """ Interpolate 3D particle data to a 1D cross-sectional line. """
        return zeros(pixels)

    @staticmethod
    def interpolate_3d_projection(x: ndarray, y: ndarray, z: ndarray, weight: ndarray, h: ndarray,
                                  weight_function: CPUDispatcher, kernel_radius: float, x_pixels: int, y_pixels: int,
                                  x_min: float, x_max: float, y_min: float, y_max: float, exact: bool) -> ndarray:
        """ Interpolate 3D particle data to a 2D grid of pixels, using column projection."""
        return zeros((y_pixels, x_pixels))

    @staticmethod
    def interpolate_3d_projection_vec(x: ndarray, y: ndarray, weight_x: ndarray, weight_y: ndarray, h: ndarray,
                                      weight_function: CPUDispatcher, kernel_radius: float, x_pixels: int,
                                      y_pixels: int, x_min: float, x_max: float, y_min: float, y_max: float,
                                      exact: bool) -> Tuple[ndarray, ndarray]:
        """ Interpolate 3D particle vector data to a pair of 2D grids of pixels, using column projection."""
        return zeros((y_pixels, x_pixels)), zeros((y_pixels, x_pixels))

    @staticmethod
    def interpolate_3d_cross(x: ndarray, y: ndarray, z: ndarray, z_slice: float, weight: ndarray, h: ndarray,
                             weight_function: CPUDispatcher, kernel_radius: float, x_pixels: int, y_pixels: int,
                             x_min: float, x_max: float, y_min: float, y_max: float) -> ndarray:
        """
        Interpolate 3D particle data to a pair of 2D grids of pixels, using a 3D cross-section at a specific z value.
        """
        return zeros((y_pixels, x_pixels))

    @staticmethod
    def interpolate_3d_cross_vec(x: ndarray, y: ndarray, z: ndarray, z_slice: float, weight_x: ndarray,
                                 weight_y: ndarray, h: ndarray, weight_function: CPUDispatcher, kernel_radius: float,
                                 x_pixels: int, y_pixels: int, x_min: float, x_max: float, y_min: float,
                                 y_max: float) -> Tuple[ndarray, ndarray]:
        """
        Interpolate 3D particle vector data to a pair of 2D grids of pixels, using a 3D cross-section at a
        specific z value.
        """
        return zeros((y_pixels, x_pixels)), zeros((y_pixels, x_pixels))

    @staticmethod
    def interpolate_3d_grid(x: ndarray, y: ndarray, z: ndarray, weight: ndarray, h: ndarray,
                            weight_function: CPUDispatcher, kernel_radius: float, x_pixels: int, y_pixels: int,
                            z_pixels: int, x_min: float, x_max: float, y_min: float, y_max: float, z_min: float,
                            z_max: float) -> ndarray:
        """
        Interpolate 3D particle data to a 3D grid of pixels.
        """
        return zeros((z_pixels, y_pixels, x_pixels))
