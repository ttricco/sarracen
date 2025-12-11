
.. role:: python(code)
   :language: python

======================
Sarracen documentation
======================

Sarracen is a Python library for analysis and visualization of smoothed
article hydrodynamics (SPH) data.

Our goal is to leverage the rich data science toolkits available in Python for
the analysis of SPH data. Sarracen is built upon the pandas and Matplotlib data
and visualization libraries. SPH data can be loaded into a pandas DataFrame
structure that has been extended to support SPH. Sarracen should be familiar to
you if you have previous experience with Matplotlib, NumPy or pandas. Our
primary intended application is for astrophysical SPH simulations.

Visualizations of the data use the SPH kernel, and a variety of rendering
options are available. All SPH interpolation functions are optimized into
machine code using Numba with both multi-threaded and CUDA enabled routines.
Our aim is to provide common analyses tasks as part of Sarracen, for example,
calculating surface density profiles. This aids in correctness, performance and
reproducibility.

Visit the :ref:`quick start guide <quick_start>` to learn the basics of
Sarracen. For details on specific functions, consult the :ref:`API reference
<api>`. The codebase can be found on `GitHub
<https://github.com/ttricco/sarracen/>`_.

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
