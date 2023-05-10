Sarracen
========

A Python library for smoothed particle hydrodynamics (SPH) analysis and visualization.

About
-----

Sarracen is built upon the pandas, Matplotlib and NumPy libraries. It can load SPH particle data into a pandas DataFrame object that has been extended to allow for rendering of particle data and interpolation of particles to fixed grids. Additionally, this allows for access to the rich landscape of Python scientific and statistical libraries. All SPH functions offer multi-threaded CPU and CUDA implementations. Our intended application is for astrophysical SPH data. 

Installation
------------

The latest stable release and associated dependencies can be installed from PyPi:

    pip install sarracen

This is the recommended way to install Sarracen.

To install the latest development snapshot, install using this GitHub repository. Either clone the repository and add it to your path so that it can be imported, or install directly through pip:

    pip install git+https://github.com/ttricco/sarracen.git

Documentation
-------------

Sarracen's documentation is hosted online at [https://sarracen.readthedocs.io](https://sarracen.readthedocs.io).

Contributing
------------

Contributions are welcomed and appreciated. Here are some ways to get involved:
- Submitting bug reports.
- Feature requests or suggestions.
- Improving the documentation or providing examples.
- Writing code to add optimizations or new features.

Please use the [GitHub issue tracker](https://github.com/ttricco/sarracen/issues) to raise any bugs or to submit feature
requests. If something does not work as you might expect, please let us know. If there are features that you feel are
missing, please let us know.

Code submissions should be submitted as a pull request. Make sure that all existing unit tests successfully pass, and 
please add any new unit tests that are relevant. Documentation changes should also be submitted as a pull request.

If you are stuck or need help, [raising an issue](https://github.com/ttricco/sarracen/issues) is a good place to start. 
This helps us keep common issues in public view. Feel free to also [email](mailto:tstricco@mun.ca) with questions.

Please note that we adhere to a [code of conduct](CODE_OF_CONDUCT.md).