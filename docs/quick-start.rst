.. _quick_start:

=================
Quick Start Guide
=================



Loading data into a SarracenDataFrame
-------------------------------------

SarracenDataFrame extends the pandas DataFrame. Thus, Sarracen can read all the file formats that pandas supports.

Additionally, Sarracen can read the native binary format of `Phantom <https://phantomsph.bitbucket.io>`_. We would like to support file formats for other SPH codes in future. Contact us if you wish to have your code supported (or raise an issue).

Loading Phantom data is as straightforward as

>>> import sarracen
>>>
>>> sdf = sarracen.read_phantom('dumpfile')

This call can separate different particle species into their own SarracenDataFrame. By default, sink particles are separated, and a list of SarracenDataFrames is returned in such a case. For example, if you data contains SPH particles plus sink particles, a sensible call would be

>>> import sarracen
>>>
>>> sdf, sdf_sinks = sarracen.read_phantom('dumpfile')

If you encounter any bugs with file reading for your particular set up, please contact us or raise an issue.


Analysis
--------

Your analysis may be specific to your particular problem and needs. All the same, :python:`sdf.describe()` is often a good starting point to get a high-level statistical summary of your data. Additionally, since SarracenDataFrame extends the pandas DataFame, it has a very close integration with numpy and works well with scipy.


Density
-------

Since density is a function of smoothing length and mass, many SPH codes do not explicitly store the density (Phantom is no exception). In general, Sarracen respects this desire to save memory. Sarracen's render functions will accept ``rho`` as a rendering target even if density is not present.
Interpolation functions will compute density on the fly.

A convenience function exists (:python:`sdf.calc_density()`) that calculates density and adds it as a column to the data frame for times when density may be needed. See the :ref:`api` for more details.


Rendering Data
--------------

:python:`sdf.render()` is the main function for rendering data. It can be used with 2D or 3D data.

For 3D data, this will render a cross section of the data if the ``xsec=`` argument has a value, otherwise it will perform a column integrated view of the data. Sarracen also offers rotations, 1D cuts through data, and rendering of vector fields through streamlines and arrow plots. See the :ref:`api` for more details.


pandas Primer
-------------

The pandas DataFrame (and hence SarracenDataFrame) is a swiss army knife of data manipulation. Columns in the data frame can be combined together mathematically, new columns assigned with little effort, and subsets of data extracted with a straightforward API.

The below will compute the magnitude of the velocity and store the result as a new column in the data frame. Note the use of numpy.

>>> sdf['vmag'] = np.sqrt(sdf['vx']**2 + sdf['vy']**2 + sdf['vz']**2)

Extracting subsets of data can be done using boolean logic to slice into the data frame. The example below computes the average speed of particles above a certain critical density threshold. The first set of square brackets will return a `copy` of the data frame containing only the particles that have density greater than ``rho_crit``, while the second square brackets accesses the column ``vmag`` of that copy.

>>> rho_crit = 1.0e10
>>> sdf.calc_density()
>>> sdf[sdf['rho'] > rho_crit]['vmag'].mean()
