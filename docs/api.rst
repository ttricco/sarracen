.. _api:

=============
API Reference
=============

.. currentmodule:: sarracen

SarracenDataFrame
-----------------

A SarracenDataFrame extends the pandas DataFrame class. It holds SPH particle data. They attempt to find particle position, velocity, smoothing length, mass and density within the data so that they can be used to interpolate and render the data.

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


File reading
------------

Sarracen can read the native binary format of the Phantom SPH code (`https://phantomsph.bitbucket.io <https://phantomsph.bitbucket.io/>`_). Contact us if you want Sarracen to be able to read the file format for other SPH codes (or make a pull request!).

.. autosummary::
   :toctree: api/

   read_phantom
   read_csv
   read_marisa


Kernels
-------

The default smoothing kernel is the cubic spline. Additional smoothing kernels are included within Sarracen.

.. autosummary::
   :toctree: api/

   kernels.CubicSplineKernel
   kernels.QuarticSplineKernel
   kernels.QuinticSplineKernel
