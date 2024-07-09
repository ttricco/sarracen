from typing import Union

import numpy as np
import pandas as pd

import vtk
from vtk.util.numpy_support import vtk_to_numpy

from ..sarracen_dataframe import SarracenDataFrame

def read_header(filename):
    """ function to read the header of a vtk file. For now 
    there is no relevant info written in the header of Shamrock."""
    header = None
    with open(filename, 'rb') as f:
        # Read lines until we find the header line starting with '# vtk DataFile Version'
        while True:
            line = f.readline().decode('utf-8')
            if line.startswith('# vtk DataFile Version'):
                header = line.strip() 
                break
            if not line: 
                break
    return header

def read_vtk_series(filename):
    reader = vtk.vtkDataSetReader()
    reader.SetFileName(filename)
    reader.Update()

    vtk_data = reader.GetOutput()
    num_arrays = vtk_data.GetPointData().GetNumberOfArrays()
    columns_dict = {}

    for i in range(num_arrays):
        vtk_array = vtk_data.GetPointData().GetArray(i)
        array_name = vtk_array.GetName()
        numpy_array = vtk_to_numpy(vtk_array)
        columns_dict[array_name] = numpy_array

    points = vtk_data.GetPoints()
    numpy_points = vtk_to_numpy(points.GetData())

    columns_dict['x'] = numpy_points[:, 0]
    columns_dict['y'] = numpy_points[:, 1]
    columns_dict['z'] = numpy_points[:, 2]

    columns_dict['vx'] = columns_dict['v'][:, 0]
    columns_dict['vy'] = columns_dict['v'][:, 1]
    columns_dict['vz'] = columns_dict['v'][:, 2]

    columns_dict['Bx'] = columns_dict['B/rho'][:, 0]
    columns_dict['By'] = columns_dict['B/rho'][:, 1]
    columns_dict['Bz'] = columns_dict['B/rho'][:, 2]

    columns_dict['B'] = columns_dict['B/rho']
 
    # Print or use the extracted columns as needed
    for column_name, column_data in columns_dict.items():
        print(f"{column_name}: {column_data}")

    series = pd.Series(columns_dict)
    df = pd.DataFrame(series)

    df_list = sarracen.SarracenDataFrame(df)
    return df_list


def read_vtk(filename, pmass):
    """
    Read data from a SHAMROCK vtk file ('big' simulation current format).

    Parameters
    ----------
    filename : str
        Name of the file to be loaded.
    pmass : float
        Mass of particles in the simulation (for now, it is assumed all particles)
        have the same mass).

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

        if ndim==1:
            print(array_name)
            df[array_name] = numpy_array

        else:
            df[array_name + 'x'] = numpy_array[:, 0]
            df[array_name + 'y'] = numpy_array[:, 1]
            df[array_name + 'z'] = numpy_array[:, 2]

    # add B
    df['Bx'] = df['B/rhox'] * df['rho']
    df['By'] = df['B/rhoy'] * df['rho']
    df['Bz'] = df['B/rhoz'] * df['rho']
    
    # now add position columns
    points = vtk_data.GetPoints()
    numpy_points = vtk_to_numpy(points.GetData())

    df['x'] = numpy_points[:, 0]
    df['y'] = numpy_points[:, 1]
    df['z'] = numpy_points[:, 2]

    # finish by adding mass
    df['mass'] = pmass * np.ones_like(numpy_points[:, 0])
    sarracen_df = sarracen.SarracenDataFrame(df)

    return sarracen_df
