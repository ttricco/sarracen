import numpy as np
import pandas as pd

import vtk
from vtk.util.numpy_support import vtk_to_numpy

from ..sarracen_dataframe import SarracenDataFrame


def read_shamrock_vtk(filename: str, pmass: float) -> SarracenDataFrame:
    """
    Read data from a Shamrock vtk file (compatible with Paraview).

    Parameters
    ----------
    filename : str
        Name of the file to be loaded.
    pmass : float
        Mass of particles in the simulation (for now, it is assumed all
        particles have the same mass).

    Returns
    -------
    SarracenDataFrame

    """

    # read the vtk file
    reader = vtk.vtkDataSetReader()
    reader.SetFileName(filename)
    reader.Update()
    vtk_data = reader.GetOutput()
    num_arrays = vtk_data.GetPointData().GetNumberOfArrays()

    # initialize the dataframe
    df = pd.DataFrame()

    for i in range(num_arrays):
        vtk_array = vtk_data.GetPointData().GetArray(i)
        array_name = vtk_array.GetName()
        numpy_array = vtk_to_numpy(vtk_array)
        ndim = numpy_array.ndim

        if ndim == 1:
            df[array_name] = numpy_array
        else:
            df[array_name + 'x'] = numpy_array[:, 0]
            df[array_name + 'y'] = numpy_array[:, 1]
            df[array_name + 'z'] = numpy_array[:, 2]

    # add B
    if 'B/rhox' in df.columns:
        df['Bx'] = df['B/rhox'] * df['rho']
    if 'B/rhoy' in df.columns:
        df['By'] = df['B/rhoy'] * df['rho']
    if 'B/rhoz' in df.columns:
        df['Bz'] = df['B/rhoz'] * df['rho']

    # now add position columns
    points = vtk_data.GetPoints()
    numpy_points = vtk_to_numpy(points.GetData())

    df['x'] = numpy_points[:, 0]
    df['y'] = numpy_points[:, 1]
    df['z'] = numpy_points[:, 2]

    # finish by adding mass
    df['mass'] = pmass * np.ones_like(numpy_points[:, 0])
    return SarracenDataFrame(df)
