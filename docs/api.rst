.. _api:

=============
API Reference
=============

.. currentmodule:: sarracen

File reading
------------

Sarracen's design goal is to read data from multiple SPH codes while preserving
full functionality. All the general file formats supported by pandas (csv,
notably) work within Sarracen.

For SPH codes, Sarracen supports reading the native binary format of the
`Phantom code <https://phantomsph.bitbucket.io>`_, the `Gasoline code
<https://gasoline-code.com/>`_, and the `Shamrock code
<https://shamrock-code.github.io/>`_.

Raise an issue on our GitHub if you would like Sarracen to be able to read the
file format for other SPH codes (or make a pull request!).

.. autosummary::
   :toctree: api/

   read_csv
   read_gasoline
   read_gradsph
   read_marisa
   read_phantom
   read_phantom_ev
   read_shamrock
   read_shamrock_vtk


File writing
------------

Sarracen can write native binary Phantom dump files. SarracenDataFrames can
also be dumped to .csv using pandas functionality.

.. autosummary::
   :toctree: api/

   write_phantom


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
   SarracenDataFrame.calc_one_fluid_quantities
   SarracenDataFrame.centre_of_mass

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


sarracen.kernels
----------------

The default smoothing kernel is the cubic spline. Additional smoothing kernels
are included within the kernels module.

.. autosummary::
   :toctree: api/

   kernels.CubicSplineKernel
   kernels.QuarticSplineKernel
   kernels.QuinticSplineKernel


sarracen.disc
-------------

Accretion disc analysis routines are in the disc module.

.. autosummary::
   :toctree: api/

   disc.surface_density
   disc.azimuthal_average
   disc.angular_momentum
   disc.scale_height
   disc.honH


sarracen.ptmass
---------------

Analysis routines related to point masses and sink particles are in the ptmass
module.

.. autosummary::
   :toctree: api/

   ptmass.classify_bound_particles
