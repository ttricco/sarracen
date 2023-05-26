---
title: 'Sarracen: a Python package for analysis and visualization of smoothed particle hydrodynamics data'
tags:
  - Python
  - Jupyter Notebook
  - smoothed particle hydrodynamics
  - astronomy
  - astrophysics
  - data visualization
  - data science
authors:
  - name: Andrew Harris
    equal-contrib: true # (This is how you can denote equal contributions between multiple authors)
    affiliation: 1 # (Multiple affiliations must be quoted)
  - name: Terrence S. Tricco
    orcid: 0000-0002-6238-9096
    equal-contrib: true
    corresponding: true
    affiliation: 1
affiliations:
  - name: Memorial University of Newfoundland, Canada
    index: 1
date: 1 March 2023
bibliography: paper.bib
---

# Summary

`Sarracen` is a Python package for analyzing and visualizing smoothed particle
hydrodynamics (SPH) data. SPH is a method of fluid simulation that discretizes
fluid elements into a collection of particles [@gm:1977; @lucy:1977; @monaghan:2005;
@price:2012]. This approach works well for many astrophysical problems of interest,
and as such there are a number of SPH codes widely used for astrophysical 
simulations, e.g., @gasoline2, @phantom, @swift. `Sarracen` offers a variety of 
SPH interpolation methods to aid in analysis and visualization of SPH data. It is 
built in Python so that users can leverage the robust scientific libraries that are 
available. Much of the core of `Sarracen` is built upon `pandas` and `Matplotlib`. 
Users familiar with these packages should be able to use `Sarracen` to do complex 
analyses without difficulty. Our intended use is for astrophysical SPH data, but 
anticipate that our package may be useful in other scientific domains. 

# Statement of need

`Splash` [@splash] is the current standard bearer for visualization of astrophysical
SPH data. It is an open-source, command-line visualization tool written in Fortran.
It is comprehensive, highly efficient, and can natively read SPH data from a large 
number of SPH simulation codes. `Splash` has a large user base for these reasons. The
significant shortcoming of `Splash` is that it has limited capability for analysis of 
SPH data. Any complicated analysis requires modification of the Fortran code. 

There are publicly available Python solutions for visualization or analysis of 
astrophysical SPH data. However, these are often specific to a single code, such as 
`SWIFTsimIO` [@swiftsimio], which is dedicated to the `Swift` [@swift] cosmological 
code. `yt` [@yt] is a general purpose analysis and visualization package for 
volumetric astrophysical data. Originally designed for data from grid-based codes, 
recent versions have added support to store SPH particle data directly (instead of 
storing only a mesh interpolation). `Plonk` [@plonk] is the work most similar to 
ours, but uses custom data structures for storing particle data and is limited to 
reading HDF5 data from the `Phantom` [@phantom] SPH code.

Our goal with `Sarracen` is to provide a Python package that implements the robust
interpolation and visualization of SPH data offered by `Splash`, but which can be
used for deeper analysis and integrated into a data scientist's Python toolkit. We 
use `Matplotlib` [@matplotlib] for visualization, and an extension of the `pandas` 
[@pandas] DataFrame structure for storing particle data. Using `Sarracen` should 
be familiar for most users. Furthermore, Python has many high-quality scientific 
libraries for data manipulation and statistical analysis, such as `NumPy` [@numpy] 
and `SciPy` [@scipy]. A user will be able to easily write custom analysis scripts 
specific to their simulation or area of astronomy and astrophysics. These factors 
should aid in making analyses more reproducible, efficient, and less error prone. 
Finally, `Sarracen` can be run interactively inside of a Jupyter notebook 
environment, which enables results to be easily shared, presented and modified.


# Features

At its core, `Sarracen` supports the interpolation of SPH particle data via the SPH
smoothing kernel. The basic approach for interpolation of a quantity, $A$, is
\begin{equation}
A_a = \sum_b \frac{m_b}{\rho_b} A_b W_{ab}(h_b),
\end{equation}
where the summation is over neighbouring particles, $m$ is the mass, $\rho$ is 
density, and $W(h_b)$ is the smoothing kernel with smoothing length, $h$. 
`Sarracen` includes multiple choices for the smoothing kernel, with the cubic spline 
as default. 

For 3D data, a quantity may be interpolated to a 3D fixed grid, to a 2D grid 
representing a slice through the data, or to a 1D line that cuts through the volume. 
Column integrated line-of-sight interpolation is included. Interpolation of 2D data 
is also supported. Additionally, `Sarracen` implements the mapping method of 
@petkova:2018, which exactly computes the volume-averaged quantity within each 
cell of a fixed grid by analytically computing the integral of the kernel function 
over the volume of each cell. `Sarracen` can render interpolated grids with 
`Matplotlib` using API syntax inspired by `Seaborn` [@seaborn]. Vector quantities, 
such as velocity, can be rendered with streamlines or arrow plots. `Sarracen` uses 
`SciPy` to support view rotation, with the rotation specified by Euler angles, a 
rotation vector, rotation matrix or quaternions.

The interpolation routines in `Sarracen` use `Numba` [@numba] to implement 
multi-threaded CPU parallelization and CUDA-enabled GPU acceleration. Furthermore,
`Numba` translates these routines into optimized machine code using just-in-time
compilation. Additionally, operations on data are vectorized by using `NumPy`.


# Acknowledgements

Thank you to Rebecca Nealon for testing `Sarracen` and providing feedback and 
suggestions. This research was enabled in part by support provided by the Digital 
Research Alliance of Canada. We acknowledge the support of the Natural Sciences and 
Engineering Research Council of Canada (NSERC).


# References

