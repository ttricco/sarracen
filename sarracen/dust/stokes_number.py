import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from sklearn.neighbors import KDTree
from ..sarracen_dataframe import SarracenDataFrame

from typing import Union, Tuple, Literal


# BRANCH CODE BELOW

#SELF IMPLIES SARRACEN DATAFRAME. NEED TO UPDATE THIS!!!!!!

#method to interpolate gas density for dust particles
def interpolate_gas_density(self):
    """Linearly interpolates gas density onto dust particles using a grid method.
    """
    if "rho_gas" not in self: # speed up by only performing interpolation if not already done

        self.calc_density()
        self["rho_gas"] = self["rho"]

        gas_df = self[self["itype"]==1]
        gas_pos = (gas_df["x"],gas_df["y"],gas_df["z"])
        gasvals = gas_df["rho"]

        dust_df = self[self["itype"]==7]
        dust_pos = (dust_df["x"],dust_df["y"],dust_df["z"])
        dustvals = griddata(gas_pos,gasvals,dust_pos,method="nearest")

        self.loc[self["itype"]==7,"rho_gas"] = dustvals

def calc_stokes(self,c_s,G=1):
    """"Calculates Stokes number of two-fluid dust particles."""
    if "St" not in self:
        self.interpolate_gas_density()
        self["r"] = (self[self.xcol]**2 + self[self.ycol]**2)**0.5
        omega_kep = (self.sinks["m"][0]*G/self["r"]**3)**0.5
        t_s = self.params["graindens"] * self.params["grainsize"]/(self["rho_gas"]+self["rho"])/c_s
        self["St"] = t_s * omega_kep
        self.loc[self["itype"]==1,"St"] = 0

#---------------------------------------------------------------------------------------------------------------
# PR CODE BELOW

def _check_dimension(data: 'SarracenDataFrame',  # noqa: F821
                     dim: Literal[2, 3]) -> None:
    """
    Verify that a given dataset describes data with a required number of
    dimensions.

    Parameters
    ----------
    data: SarracenDataFrame
        The particle dataset to interpolate over.
    dim: [2, 3]
        The number of required dimensions.

    Raises
    ------
    ValueError
        If the dataset is not `dim`-dimensional or `dim` is not 2 or 3.
    """
    if dim not in [2, 3]:
        raise ValueError("`dim` must be 2 or 3.")
    if data.get_dim() != dim:
        raise ValueError(f"Dataset is not {dim}-dimensional.")

def _default_xyz(data: 'SarracenDataFrame',  # noqa: F821
                 x: Union[str, None],
                 y: Union[str, None],
                 z: Union[str, None]) -> Tuple[str, str, str]:
    """
    Utility function to determine the x, y and z columns to use during 3-D
    interpolation.

    Parameters
    ----------
    data: SarracenDataFrame
        The particle dataset to interpolate over.
    x, y, z: str
        The x, y and z directional column labels passed to the interpolation
        function.

    Returns
    -------
    x, y, z: str
        The directional column labels to use in interpolation.
    """
    xcol = data.xcol
    ycol = data.ycol
    zcol = data.zcol

    if x is None:
        x = xcol if not y == xcol and not z == xcol else \
            ycol if not y == ycol and not z == ycol else zcol
    if y is None:
        y = ycol if not x == ycol and not z == ycol else \
            xcol if not x == xcol and not z == xcol else zcol
    if z is None:
        z = zcol if not x == zcol and not y == zcol else \
            ycol if not x == ycol and not y == ycol else xcol

    return x, y, z

def _default_vxyz(data, vx, vy, vz) -> Tuple[str, str, str]:
    """
    Utility function to determine the vx, vy, and vz columns to use.
    Parameters
    ----------
    data: SarracenDataFrame
        The particle dataset to use.
    vx, vy, vz: str
        The vx, vy, and vz directional column labels.
    Returns
    -------
    vx, vy, vz: str
        The directional column labels to use.
    """
    vxcol = data.vxcol
    vycol = data.vycol
    vzcol = data.vzcol

    if vx is None:
        vx = vxcol if not vy == vxcol and not vz == vxcol else \
             vycol if not vy == vycol and not vz == vycol else vzcol
    if vy is None:
        vy = vycol if not vx == vycol and not vz == vycol else \
             vxcol if not vx == vxcol and not vz == vxcol else vzcol
    if vz is None:
        vz = vzcol if not vx == vzcol and not vy == vzcol else \
             vycol if not vx == vycol and not vy == vycol else vxcol

    return vx, vy, vz

def _verify_columns(data: 'SarracenDataFrame',  # noqa: F821
                    columns: list) -> None:
    """
    Verify that the given columns exist in `data`.

    Parameters
    ----------
    data: SarracenDataFrame
        The particle dataset to interpolate over.
    columns: list
        The column labels.

    Raises
    ------
    KeyError
        If a label in columns does not exist in `data`.
    """
    for column_label in columns:
        if column_label not in data.columns:
            raise KeyError(f"'{column_label}' column does not exist in the "
                           f"provided dataset")
        
def _get_mass(data: 'SarracenDataFrame') -> Union[np.ndarray,  # noqa: F821
                                                  float]:
    if data.mcol is None:
        return data.params['mass']

    return data[data.mcol].to_numpy()


def _get_density(data: 'SarracenDataFrame') -> np.ndarray:  # noqa: F821
    if data.rhocol is None:
        hfact = data.params['hfact']
        mass = _get_mass(data)
        return ((hfact / data[data.hcol])**(data.get_dim()) * mass).to_numpy()

    return data[data.rhocol].to_numpy()

def stoppingtime(rho_dust, rho_gas, v_gas, v_dust,
                 rho_grain, grain_size, gamma, c_s):
    return np.sqrt(np.pi * gamma * 0.125) * rho_grain * grain_size / \
           (rho_dust + rho_gas) / np.sqrt(1 + 0.0703125 * np.pi * 
           (np.linalg.norm(v_gas - v_dust))**2 / c_s**2)

def Stokes_number(data_dust: 'SarracenDataFrame',
                  data_gas: 'SarracenDataFrame',
                  rho_grain: float,
                  grain_size: float,
                  c_s: float,
                  kernel: BaseKernel = None,
                  x_dust: str = None,
                  y_dust: str = None,
                  z_dust: str = None,
                  vx_dust: str = None,
                  vy_dust: str = None,
                  vz_dust: str = None,
                  x_gas: str = None,
                  y_gas: str = None,
                  z_gas: str = None,
                  vx_gas: str = None,
                  vy_gas: str = None,
                  vz_gas: str = None):
    dim = 3
    _check_dimension(data_dust, dim)
    _check_dimension(data_gas, dim)

    # Getting default values
    x_dust, y_dust, z_dust = _default_xyz(data_dust, x_dust, y_dust, z_dust)
    x_gas, y_gas, z_gas = _default_xyz(data_gas, x_gas, y_gas, z_gas)
    vx_dust, vy_dust, vz_dust = _default_vxyz(data_dust, vx_dust,
                                              vy_dust, vz_dust)
    vx_gas, vy_gas, vz_gas = _default_vxyz(data_gas, vx_gas, vy_gas, vz_gas)

    # Ensuring all required columns exist in their respective dataframes
    _verify_columns(data_dust, [x_dust, y_dust, z_dust, 'h',
                                vx_dust, vy_dust, vz_dust])
    _verify_columns(data_gas, [x_gas, y_gas, z_gas, 'h',
                               vx_gas, vy_gas, vz_gas])

    # Getting specific dataframe columns as lists
    rho_dust_data = _get_density(data_dust)

    h_gas_data = data_gas['h'].values
    vx_gas_data = data_gas[vx_gas].values
    vy_gas_data = data_gas[vy_gas].values
    vz_gas_data = data_gas[vz_gas].values
    rho_gas_data = _get_density(data_gas)

    # Getting gas and dust coordinates
    gas_positions = data_gas[[x_gas, y_gas, z_gas]].values
    dust_positions = data_dust[[x_dust, y_dust, z_dust]].values

    # Getting dust velocities
    dust_velocities = data_dust[[vx_dust, vy_dust, vz_dust]].values

    # Creating a tree for gas positions
    gas_tree = KDTree(gas_positions, leaf_size=10)

    # Search for dust particles that are neighbours of each gas particle.
    # "all_gas_neighbours" is an ndarray of size len(gas_positions).
    # Each element of "all_gas_neighbours" is an integer array of indices of 
    # "dust_positions" that are neighbours of the corresponding gas particle.
    all_gas_neighbours = gas_tree.query_radius(dust_positions, r=2*h_gas_data)

    # Initialize structure to hold the list of gas particles that 
    # each dust particle is a neighbour of
    all_dust_neighbours = [np.array([], dtype=np.int64) 
                           for _ in range(len(dust_positions))]

    # Loop over each gas-dust neighbour pair
    for gas_particle, gas_particle_neighbours in enumerate(all_gas_neighbours):
        for dust_particle in gas_particle_neighbours:
            all_dust_neighbours[dust_particle] = np.append(
                all_dust_neighbours[dust_particle], gas_particle)

    dust_number = len(data_dust)
    rhog_on_dust = np.zeros(dust_number)
    vx_on_dust = np.zeros(dust_number)
    vy_on_dust = np.zeros(dust_number)
    vz_on_dust = np.zeros(dust_number)

    for ind, array in enumerate(all_dust_neighbours):
        vx_neighb = 0
        vy_neighb = 0
        vz_neighb = 0    
        neighbor_rho = 0

        r_dust = dust_positions[ind]

        for j in array:
            rho_gas = rho_gas_data[j]
            r_gas = gas_positions[j]
            h_gas = h_gas_data[j]
            vx_gas = vx_gas_data[j]
            vy_gas = vy_gas_data[j]
            vz_gas = vz_gas_data[j]

            q = np.linalg.norm(r_gas - r_dust) / h_gas
            normalized_weight = kernel.w(q, dim)

            neighbor_rho += rho_gas * normalized_weight
            vx_neighb += vx_gas * normalized_weight
            vy_neighb += vy_gas * normalized_weight
            vz_neighb += vz_gas * normalized_weight

        rhog_on_dust[ind] = neighbor_rho 
        vx_on_dust[ind] = vx_neighb
        vy_on_dust[ind] = vy_neighb
        vz_on_dust[ind] = vz_neighb

    gas_velocity_on_dust = np.vstack((vx_on_dust, vy_on_dust, vz_on_dust)).T

    tstop = stoppingtime(rho_dust_data, rhog_on_dust, gas_velocity_on_dust,
                         dust_velocities, rho_grain, grain_size, gamma, c_s)
    stokes_number = tstop * c_s * rhog_on_dust * (1/3) / \
        data_gas.params['hfact'] * data_gas.params['mass']**(1/3)

    return stokes_number

def calc_stokes_number(data_gas,
                       rho_dust_data,
                       rhog_on_dust, 
                       gas_velocity_on_dust,
                       dust_velocities,
                       rho_grain,
                       grain_size,
                       gamma,
                       c_s):
    tstop = stoppingtime(rho_dust_data, rhog_on_dust, gas_velocity_on_dust,
                         dust_velocities, rho_grain, grain_size, gamma, c_s)
    stokes_number = tstop * c_s * rhog_on_dust * (1/3) / \
                    data_gas.params['hfact'] * data_gas.params['mass']**(1/3)

    return stokes_number