from pandas import Series
from astropy import units as u

from .sarracen_dataframe import SarracenDataFrame


class SarracenSeries(Series):
    _metadata = ['_units']

    def __init__(self, data=None, unit=u.dimensionless_unscaled,
                 *args, **kwargs):

        if isinstance(data, u.Quantity):
            unit = data.unit
            data = data.value
        elif not isinstance(unit, u.UnitBase):
            raise TypeError("unit must be an astropy UnitBase instance.")

        super().__init__(data, *args, **kwargs)

        self.__unit = unit

    def __repr__(self):
        base_repr = super().__repr__()
        unit_repr = f", unit: {self.__unit}"
        return base_repr + unit_repr

    @property
    def _constructor(self):
        return SarracenSeries

    @property
    def _constructor_expanddim(self):
        return SarracenDataFrame

    # unit conversions

    def to(self, new_unit):
        if not isinstance(new_unit, u.UnitBase):
            raise TypeError("Unit must be an astropy UnitBase instance.")
        converted_values = (self.values * self.__unit).to(new_unit).value
        return SarracenSeries(converted_values, unit=new_unit,
                              index=self.index)

    def si(self):
        si_unit = (1 * self.__unit).si.unit
        return self.to(si_unit)

    def cgs(self):
        cgs_unit = (1 * self.__unit).cgs.unit
        return self.to(cgs_unit)
