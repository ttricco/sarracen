.. _render:

=========
Rendering
=========

Sarracen offers several options to render SPH particle data using the smoothing kernel. In general, all rendering options involve interpolation of particles to a fixed grid. Options include

* Rendering of 2-dimensional data,
* Planar slices through 3-dimensional data,
* Column integrated views through 3-dimensional data,
* 1-dimensional lines (cuts) through 2- or 3-dimensional data,
* Streamlines and arrow plots of vector fields,
* Arbitrary rotations, and
* Exact interpolation to grids by integration of the kernel.


Integration with Matplotlib
---------------------------

Rendering is implemented using `Matplotlib <https://matplotlib.org/>`_. This means that Sarracen may be used in an interactive environment, such as Jupyter notebooks, just as one would use Matplotlib.

All rendering functions both accept and return a Matplotlib Axes object. Sarracen will render onto a pre-existing Axes object if one is passed, or will use the current Axes if not. The Axes that Sarracen used will be returned, allowing for further customization to the figure.

A simple code structure to pass a new Matplotlib Axes to Sarracen and assign the returned Axes would be as follows.

>>> fig, ax = plt.subplots(figsize=(7,5))
>>> ax = sdf.render('rho', ax=ax)

Note that Matplotlib figure sizes are in inches.



Two-dimensional Data
--------------------

:python:`sdf.render()` will render the target column of the data frame.


Three-dimensional Data
----------------------

:python:`sdf.render()` will also render the target column of the data frame. The keyword argument ``xsec=`` will define whether a cross section of the data is rendered (at the specified `z` height) or a column integrated view is calculated.

By default, the cross section is taken in the x-y plane given by the specified `z` height, and column integrated views are along the `z`-axis. Different views may be obtained by rotating the data (see below).


1D Lines through Data
---------------------

:python:`sdf.lineplot()` plots 1-dimensional lines through higher dimensional data. The line along which to interpolate data is specified by its two points, and works for both 2D and 3D data sets.


Vector Fields
-------------

Vector fields can be rendered using streamlines (:python:`sdf.streamlines()`) and arrow plots (:python:`sdf.arrowplot()`).


Rotation
--------

The render functions allow for rotation of the data (``rotation=``). This may be specified as a set of rotation angles (Euler angles) given in degrees about the `z`, `y` and `x` axes. For example, to rotate 30 degrees about the `z`-axis, use ``rotation=[30,0,0]``.

Additionally, Sarracen rotations accept SciPy `Rotation objects <https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.html>`_, which allow for a much richer expression of rotations using Euler angles, rotation matrices, rotation vectors or quaternions.

Note: If you are used to using Splash, the rotation angles as specified in Sarracen have opposite sign for rotations about the z and x axes. Our implementation matches SciPy.


Colour Maps
-----------

All colour maps implemented by Matplotlib are accessible in Sarracen. The default colour map is gist_heat (equivalent to the default used in Splash). A full list of Matplotlib colour maps can be found `here <https://matplotlib.org/stable/tutorials/colors/colormaps.html>`_. For creating custom colour maps, we recommend the `Seaborn <https://seaborn.pydata.org>`_ visualization library, which has a comprehensive set of colour map tools that work with Sarracen and Matplotlib.


Exact Interpolation
-------------------

Interpolation, by default, uses the smoothing kernel to obtain values at points, which form a grid of pixels. Sarracen also implements the exact interpolation method of Petkova, Laibe & Bonnell (2018), which may be used by setting the :python:`exact=` argument in the render functions.

If each grid cell is considered a cube (or square in 2D), this method exactly solves the volume (or surface) integral of the smoothing kernel for each grid cell. This has the advantage that the pixel representation is not for a single point, but represents the entire space that pixel occcupies.


Code Efficiency
---------------

Interpolation functions are optimized into machine code using the Numba library. The first invocation of a function will be slower as this is a just in time compilation. Subsequent function calls will be fast, with generally a 10-20x speedup.

Sarracen includes both multi-threaded CPU and GPU (using CUDA) implementations of the interpolation functions. The default is to perform interplation on the CPU. This can be changed by setting the default backend for the DataFrame, or by choosing the backend individually per function call.
