
.. role:: python(code)
   :language: python

======================
Sarracen documentation
======================

Sarracen is a Python library for analysis and visualization of smoothed particle hydrodynamics (SPH) data.

It is built upon the pandas and Matplotlib data and visualization libraries.  SPH data can be loaded into a pandas
DataFrame structure that has been extended to support SPH. Visualizations of the data use the SPH kernel, and a variety
of rendering options are available. All SPH interpolation functions are optimized using Numba into machine code with
both multi-threaded and CUDA enabled routines. Our primary intended application is for astrophysical SPH simulations.

Visit the :ref:`quick start guide <quick_start>` to learn the basics of Sarracen. For details on specific functions,
consult the :ref:`API reference <api>`. The codebase can be found on `GitHub <https://github.com/ttricco/sarracen/>`_.

.. toctree::
   :maxdepth: 1
   :caption: Contents:

   installation
   quick-start
   render
   examples
   api
   contributing


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
