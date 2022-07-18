from typing import Tuple

from numba.core.registry import CPUDispatcher
from numpy import ndarray, zeros


class BaseBackend:
    """Backend implementation of SPH interpolation functions."""

    @staticmethod
    def interpolate_2d_render(target: ndarray, x: ndarray, y: ndarray, mass: ndarray, rho: ndarray, h: ndarray,
                              weight_function: CPUDispatcher, kernel_radius: float, x_pixels: int, y_pixels: int,
                              x_min: float, x_max: float, y_min: float, y_max: float, exact: bool) -> ndarray:
        """ Interpolate 2D particle data to a 2D grid of pixels."""
        return zeros((y_pixels, x_pixels))

    @staticmethod
    def interpolate_2d_render_vec(target_x: ndarray, target_y: ndarray, x: ndarray, y: ndarray, mass: ndarray,
                                  rho: ndarray, h: ndarray, weight_function: CPUDispatcher, kernel_radius: float,
                                  x_pixels: int, y_pixels: int, x_min: float, x_max: float, y_min: float,
                                  y_max: float, exact: bool) -> Tuple[ndarray, ndarray]:
        """ Interpolate 2D particle vector data to a pair of 2D grids of pixels. """
        return zeros((y_pixels, x_pixels)), zeros((y_pixels, x_pixels))

    @staticmethod
    def interpolate_2d_cross(target: ndarray, x: ndarray, y: ndarray, mass: ndarray, rho: ndarray, h: ndarray,
                             weight_function: CPUDispatcher, kernel_radius: float, pixels: int, x1: float,
                             x2: float, y1: float, y2: float) -> ndarray:
        """ Interpolate 2D particle data to a 1D cross-sectional line. """
        return zeros(pixels)

    @staticmethod
    def interpolate_3d_projection(target: ndarray, x: ndarray, y: ndarray, z: ndarray, mass: ndarray, rho: ndarray,
                                  h: ndarray, weight_function: CPUDispatcher, kernel_radius: float, x_pixels: int,
                                  y_pixels: int, x_min: float, x_max: float, y_min: float, y_max: float,
                                  exact: bool) -> ndarray:
        """ Interpolate 3D particle data to a 2D grid of pixels, using column projection."""
        return zeros((y_pixels, x_pixels))

    @staticmethod
    def interpolate_3d_projection_vec(target_x: ndarray, target_y: ndarray, x: ndarray, y: ndarray, mass: ndarray,
                                      rho: ndarray, h: ndarray, weight_function: CPUDispatcher, kernel_radius: float,
                                      x_pixels: int, y_pixels: int, x_min: float, x_max: float, y_min: float,
                                      y_max: float, exact: bool) -> Tuple[ndarray, ndarray]:
        """ Interpolate 3D particle vector data to a pair of 2D grids of pixels, using column projection."""
        return zeros((y_pixels, x_pixels)), zeros((y_pixels, x_pixels))


    @staticmethod
    def interpolate_3d_cross(target: ndarray, z_slice: float, x: ndarray, y: ndarray, z: ndarray, mass: ndarray,
                             rho: ndarray, h: ndarray, weight_function: CPUDispatcher, kernel_radius: float,
                             x_pixels: int, y_pixels: int, x_min: float, x_max: float, y_min: float,
                             y_max: float) -> ndarray:
        """
        Interpolate 3D particle data to a pair of 2D grids of pixels, using a 3D cross-section at a specific z value.
        """
        return zeros((y_pixels, x_pixels))


    @staticmethod
    def interpolate_3d_cross_vec(target_x: ndarray, target_y: ndarray, z_slice: float, x: ndarray, y: ndarray,
                                 z: ndarray, mass: ndarray, rho: ndarray, h: ndarray, weight_function: CPUDispatcher,
                                 kernel_radius: float, x_pixels: int, y_pixels: int, x_min: float, x_max: float,
                                 y_min: float, y_max: float) -> Tuple[ndarray, ndarray]:
        """
        Interpolate 3D particle vector data to a pair of 2D grids of pixels, using a 3D cross-section at a
        specific z value.
        """
        return zeros((y_pixels, x_pixels)), zeros((y_pixels, x_pixels))
