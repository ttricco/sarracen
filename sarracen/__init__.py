from .readers.read_csv import read_csv
from .readers.read_gasoline import read_gasoline
from .readers.read_gradsph import read_gradsph
from .readers.read_marisa import read_marisa
from .readers.read_phantom import read_phantom
from .readers.read_phantom_ev import read_phantom_ev
from .readers.read_shamrock import read_shamrock

try:
    from .readers.read_shamrock_vtk import read_shamrock_vtk
except ModuleNotFoundError as e:
    # temporary workaround for no VTK in Python 3.8 on Mac ARM
    if e.name == "vtk":
        def read_shamrock_vtk(*args, **kwargs):
            raise ImportError("vtk is not installed.")
    else:
        raise


from .writers.write_phantom import write_phantom

from .sarracen_dataframe import SarracenDataFrame

from .interpolate import interpolate_2d, interpolate_2d_line, \
    interpolate_3d_proj, interpolate_3d_cross
from .render import render, streamlines, arrowplot

from . import disc
from . import ptmass

__version__ = "1.3.1"

__all__ = ["read_csv", "read_gasoline", "read_gradsph", "read_marisa",
           "read_phantom", "read_phantom_ev", "read_shamrock",
           "read_shamrock_vtk", "write_phantom",
           "SarracenDataFrame", "disc", "ptmass",
           "interpolate_2d", "interpolate_2d_line", "interpolate_3d_proj",
           "interpolate_3d_cross", "render", "streamlines", "arrowplot"]
