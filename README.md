Sarracen
========

A Python library for smoothed particle hydrodynamics (SPH) analysis and visualization.

About
-----

Sarracen is built upon the pandas, Matplotlib and NumPy libraries. It can load SPH particle data into a pandas DataFrame object that has been extended to allow for rendering of particle data and interpolation of particles to fixed grids. Additionally, this allows for access to the rich landscape of Python scientific and statistical libraries. All SPH functions offer multi-threaded CPU and CUDA implementations. Our intended application is for astrophysical SPH data. 

Installation
------------

Clone the Sarracen repository and add it to your path so that it can be imported.

    import sys
    sys.path.append('/path/to/sarracen') # replace with your path to sarracen

    import sarracen

Documentation
-------------

Sarracen's documentation is hosted online at [https://sarracen.readthedocs.io](https://sarracen.readthedocs.io).

Bugs / Missing Features
-----------------------

Please raise any bugs [as an issue](https://github.com/ttricco/sarracen/issues). If something does not work as you might expect, please let us know. If there are features that you feel are missing, please let us know.
