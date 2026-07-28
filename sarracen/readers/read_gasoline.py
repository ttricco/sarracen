from typing import Tuple, Union

import numpy as np
import pandas as pd
import os
import struct

from ..sarracen_dataframe import SarracenDataFrame


def read_gasoline(filename: str,
                  outtype: str = "sarracen") -> Union[Tuple[SarracenDataFrame,
                                                            SarracenDataFrame,
                                                            SarracenDataFrame],
                                                      Tuple[dict, dict,
                                                            dict, dict], int]:
    """
    Read data from a Gasoline tipsy file.

    Particles are separated into three data structures -- one for SPH gas
    particles, one for dark matter particles, and one for star particles.

    Global values stored in the dump file are stored within the
    SarracenDataFrames in the dictionary ``params``. If the output type is
    Python dictionaries, then an additional dictionary is returned with the
    global variables.

    Parameters
    ----------
    filename : str
        Name of the file to be loaded.
    outtype : {'sarracen', 'dict'}, default='sarracen'
        Whether to return data in SarracenDataFrames or Python dictionaries.

    Returns
    -------
    list of SarracenDataFrames or Python dictionaries

    Notes
    --------
    Adapted from `PyTipsy <https://github.com/bwkeller/pytipsy>`_.

    Examples
    --------
    Particles are separated into gas, dark matter, and star particles.

    >>> sdf_g, sdf_dm, sdf_star = sarracen.read_gasoline('dumpfile')

    The `outtype` parameter can be used to change the return type to Python
    dictionaries. This will return global variables in a separate dictionary.

    >>> header, catg, catd, cats = sarracen.read_gasoline('dumpfile', outtype='dict')
    """

    if outtype.lower() in ["dic", "dict", "dictionary"]:
        dictcheck = 1
    else:
        dictcheck = 0

    # Get all relevant files
    filin = filename.split("/")[-1]
    dirin = filename[0:len(filename)-len(filin)]
    if len(dirin) > 0:
        dirlist = os.listdir(dirin)
    else:
        dirlist = os.listdir()
        dirin = "."
    ii = 0
    while ii < len(dirlist):
        if filin not in dirlist[ii]:
            dirlist.remove(dirlist[ii])
        else:
            ii += 1
    ii = 0
    varlist = dirlist[1:]
    for var in dirlist:
        if var != filin:
            varlist[ii] = var[len(filin)+1:]
            ii += 1

    # From PyTipsy
    fp = open("{}/{}".format(dirin, filin), 'rb')
    fs = len(fp.read())
    fp.seek(0)

    # Take in the Header
    t, n, ndim, ng, nd, ns = struct.unpack("<diiiii", fp.read(28))
    endianswap = False

    # check Endianness
    if (ndim < 2 or ndim > 3):
        endianswap = True
        fp.seek(0)
        t, n, ndim, ng, nd, ns = struct.unpack(">diiiii", fp.read(28))

    # Catch for 4 byte padding
    if (fs == 32+48*ng+36*nd+44*ns):
        fp.read(4)
    # File is borked if this is true
    elif (fs != 28+48*ng+36*nd+44*ns):
        print("Tipsy ERROR: Header and file size inconsistend")
        print("Estimates: Header bytes: 28 or 32 (either is OK)")
        print("     ngas: ", ng, " bytes:", 48*ng)
        print("    ndark: ", nd, " bytes:", 38*nd)
        print("    nstar: ", ns, " bytes:", 44*ns)
        print("Actual File bytes:", fs, " does not work")
        fp.close()
        return 1

    # Make dicitonaries for data
    catg = {'mass': np.zeros(ng), 'pos': np.zeros((ng, 3)),
            'vel': np.zeros((ng, 3)), 'rho': np.zeros(ng),
            'tempg': np.zeros(ng), 'h_gas': np.zeros(ng),
            'zmetal': np.zeros(ng),  'phi': np.zeros(ng)}
    catd = {'mass': np.zeros(nd), 'pos': np.zeros((nd, 3)),
            'vel': np.zeros((nd, 3)), 'eps': np.zeros(nd),
            'phi': np.zeros(nd)}
    cats = {'mass': np.zeros(ns), 'pos': np.zeros((ns, 3)),
            'vel': np.zeros((ns, 3)), 'metals': np.zeros(ns),
            'tform': np.zeros(ns), 'eps': np.zeros(ns),
            'phi': np.zeros(ns)}
    for cat in ['g', 'd', 's']:
        j = 0
        for qty in ['x', 'y', 'z']:
            locals()['cat'+cat][qty] = locals()['cat'+cat]['pos'][:, j]
            locals()['cat'+cat]['v'+qty] = locals()['cat'+cat]['vel'][:, j]
            j += 1

    # Read in additional variables
    for var in varlist:
        fvar = open("{}.{}".format(filename, var), "rb")
        nfvar = len(fvar.read())
        fvar.seek(0)
        fvar.read(4)
        if int((nfvar-4)/4) == (ng+ns+nd):
            catg[var] = np.zeros(ng)
            catd[var] = np.zeros(nd)
            cats[var] = np.zeros(ns)
            if ng > 0:
                for i in range(ng):
                    if endianswap:
                        catg[var][i], = struct.unpack(">f", fvar.read(4))
                    else:
                        catg[var][i], = struct.unpack("<f", fvar.read(4))
            if ns > 0:
                for i in range(ns):
                    if endianswap:
                        cats[var][i], = struct.unpack(">f", fvar.read(4))
                    else:
                        cats[var][i], = struct.unpack("<f", fvar.read(4))
            if nd > 0:
                for i in range(nd):
                    if endianswap:
                        catd[var][i], = struct.unpack(">f", fvar.read(4))
                    else:
                        catd[var][i], = struct.unpack("<f", fvar.read(4))
        # read 3D variables
        elif int((nfvar-4)/12) == (ng+ns+nd):
            if dictcheck:
                catg[var] = np.zeros((ng, 3))
                catd[var] = np.zeros((nd, 3))
                cats[var] = np.zeros((ns, 3))
            varx = "{}x".format(var)
            vary = "{}y".format(var)
            varz = "{}z".format(var)
            catg[varx] = np.zeros(ng)
            catd[varx] = np.zeros(nd)
            cats[varx] = np.zeros(ns)
            catg[vary] = np.zeros(ng)
            catd[vary] = np.zeros(nd)
            cats[vary] = np.zeros(ns)
            catg[varz] = np.zeros(ng)
            catd[varz] = np.zeros(nd)
            cats[varz] = np.zeros(ns)
            if dictcheck:
                if (ng > 0):
                    for i in range(ng):
                        if endianswap:
                            catg[varx][i] = struct.unpack(">f", fvar.read(4))
                            catg[vary][i] = struct.unpack(">f", fvar.read(4))
                            catg[varz][i] = struct.unpack(">f", fvar.read(4))
                        else:
                            catg[varx][i] = struct.unpack("<f", fvar.read(4))
                            catg[vary][i] = struct.unpack("<f", fvar.read(4))
                            catg[varz][i] = struct.unpack("<f", fvar.read(4))
                        catg[var][i] = (catg[varx][i],
                                        catg[vary][i],
                                        catg[varz][i])
                if (ns > 0):
                    for i in range(ns):
                        if endianswap:
                            cats[varx][i] = struct.unpack(">f", fvar.read(4))
                            cats[vary][i] = struct.unpack(">f", fvar.read(4))
                            cats[varz][i] = struct.unpack(">f", fvar.read(4))
                        else:
                            cats[varx][i] = struct.unpack("<f", fvar.read(4))
                            cats[vary][i] = struct.unpack("<f", fvar.read(4))
                            cats[varz][i] = struct.unpack("<f", fvar.read(4))
                        catg[var][i] = (catd[varx][i],
                                        catd[vary][i],
                                        catd[varz][i])
                if (nd > 0):
                    for i in range(nd):
                        if endianswap:
                            catd[varx][i] = struct.unpack(">f", fvar.read(4))
                            catd[vary][i] = struct.unpack(">f", fvar.read(4))
                            catd[varz][i] = struct.unpack(">f", fvar.read(4))
                        else:
                            catd[varx][i] = struct.unpack("<f", fvar.read(4))
                            catd[vary][i] = struct.unpack("<f", fvar.read(4))
                            catd[varz][i] = struct.unpack("<f", fvar.read(4))
                        catg[var][i] = (cats[varx][i],
                                        cats[vary][i],
                                        cats[varz][i])
            else:
                if (ng > 0):
                    for i in range(ng):
                        if endianswap:
                            catg[varx][i] = struct.unpack(">f", fvar.read(4))
                            catg[vary][i] = struct.unpack(">f", fvar.read(4))
                            catg[varz][i] = struct.unpack(">f", fvar.read(4))
                        else:
                            catg[varx][i] = struct.unpack("<f", fvar.read(4))
                            catg[vary][i] = struct.unpack("<f", fvar.read(4))
                            catg[varz][i] = struct.unpack("<f", fvar.read(4))
                if (ns > 0):
                    for i in range(ns):
                        if endianswap:
                            cats[varx][i] = struct.unpack(">f", fvar.read(4))
                            cats[vary][i] = struct.unpack(">f", fvar.read(4))
                            cats[varz][i] = struct.unpack(">f", fvar.read(4))
                        else:
                            cats[varx][i] = struct.unpack("<f", fvar.read(4))
                            cats[vary][i] = struct.unpack("<f", fvar.read(4))
                            cats[varz][i] = struct.unpack("<f", fvar.read(4))
                if (nd > 0):
                    for i in range(nd):
                        if endianswap:
                            catd[varx][i] = struct.unpack(">f", fvar.read(4))
                            catd[vary][i] = struct.unpack(">f", fvar.read(4))
                            catd[varz][i] = struct.unpack(">f", fvar.read(4))
                        else:
                            catd[varx][i] = struct.unpack("<f", fvar.read(4))
                            catd[vary][i] = struct.unpack("<f", fvar.read(4))
                            catd[varz][i] = struct.unpack("<f", fvar.read(4))
        else:
            # Ignore Files outside of format
            print("Error With Variable {}".format(var))
        fvar.close()

    # Read in standard variables
    # Gas
    if (ng > 0):
        for i in range(ng):
            if endianswap:
                data = struct.unpack(">ffffffffffff", fp.read(48))
            else:
                data = struct.unpack("<ffffffffffff", fp.read(48))
            mass, x, y, z, vx, vy, vz, dens, tempg, h, zmetal, phi = data
            catg['mass'][i] = mass
            catg['x'][i] = x
            catg['y'][i] = y
            catg['z'][i] = z
            catg['vx'][i] = vx
            catg['vy'][i] = vy
            catg['vz'][i] = vz
            catg['rho'][i] = dens
            catg['tempg'][i] = tempg
            catg['h_gas'][i] = h
            catg['zmetal'][i] = zmetal
            catg['phi'][i] = phi

    # Dark Matter
    if (nd > 0):
        for i in range(nd):
            if endianswap:
                data = struct.unpack(">fffffffff", fp.read(36))
            else:
                data = struct.unpack("<fffffffff", fp.read(36))
            mass, x, y, z, vx, vy, vz, eps, phi = data
            catd['mass'][i] = mass
            catd['x'][i] = x
            catd['y'][i] = y
            catd['z'][i] = z
            catd['vx'][i] = vx
            catd['vy'][i] = vy
            catd['vz'][i] = vz
            catd['eps'][i] = eps
            catd['phi'][i] = phi

    # Stars
    if (ns > 0):
        for i in range(ns):
            if endianswap:
                data = struct.unpack(">fffffffffff", fp.read(44))
            else:
                data = struct.unpack("<fffffffffff", fp.read(44))
            mass, x, y, z, vx, vy, vz, metals, tform, eps, phi = data
            cats['mass'][i] = mass
            cats['x'][i] = x
            cats['y'][i] = y
            cats['z'][i] = z
            cats['vx'][i] = vx
            cats['vy'][i] = vy
            cats['vz'][i] = vz
            cats['metals'][i] = metals
            cats['tform'][i] = tform
            cats['eps'][i] = eps
            cats['phi'][i] = phi

    header = {'time': t, 'n': n, 'ndim': ndim,
              'ngas': ng, 'ndark': nd, 'nstar': ns}

    # Send dictionaries if desired
    if dictcheck:
        return header, catg, catd, cats
    # Otherwise, convert to PANDAS datframe
    else:
        catg.pop("pos")
        catg.pop("vel")
        catd.pop("pos")
        catd.pop("vel")
        cats.pop("pos")
        cats.pop("vel")
        if "smooth" not in varlist:
            catg['smooth'] = np.cbrt(catg['mass']/catg['rho'])
        catg['h'] = catg['smooth']
        dfg = pd.DataFrame.from_dict(catg)
        sdfg = SarracenDataFrame(dfg, params=header)
        dfd = pd.DataFrame.from_dict(catd)
        sdfd = SarracenDataFrame(dfd, params=header)
        dfs = pd.DataFrame.from_dict(cats)
        sdfs = SarracenDataFrame(dfs, params=header)
        return sdfg, sdfd, sdfs
