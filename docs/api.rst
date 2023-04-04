.. _api:

=============
API Reference
=============

.. currentmodule:: sarracen

File reading
------------

Sarracen can read all general file formats supported by pandas (csv, notably).

For SPH codes, Sarracen supports reading the native binary format of the `Phantom
SPH code <https://phantomsph.bitbucket.io>`_. Raise an issue on our GitHub if you
would like Sarracen to be able to read the file format for other SPH codes (or
make a pull request!).

.. autosummary::
   :toctree: api/

   read_csv
   read_phantom
   read_marisa


SarracenDataFrame
-----------------

A SarracenDataFrame is a subclass of the pandas DataFrame class. It holds SPH
particle data. SarracenDataFrames will attempt to identify columns which hold
particle positions, velocities, smoothing lengths, masses and densities so that
they can be used for interpolation and rendering. Global simulation values are
stored in ``params``, which is a standard Python dictionary.

Constructor
"""""""""""
.. autosummary::
   :toctree: api/
   :nosignatures:

   SarracenDataFrame

Extra Quantities
""""""""""""""""
.. autosummary::
   :toctree: api/
   :nosignatures:

   SarracenDataFrame.calc_density

Rendering
"""""""""

.. autosummary::
   :toctree: api/

   SarracenDataFrame.render
   SarracenDataFrame.lineplot
   SarracenDataFrame.streamlines
   SarracenDataFrame.arrowplot

Interpolation
"""""""""""""

.. autosummary::
   :toctree: api/

   SarracenDataFrame.sph_interpolate


Kernels
-------

The default smoothing kernel is the cubic spline. Additional smoothing kernels are included within Sarracen.

.. autosummary::
   :toctree: api/

   kernels.CubicSplineKernel
   kernels.QuarticSplineKernel
   kernels.QuinticSplineKernel
