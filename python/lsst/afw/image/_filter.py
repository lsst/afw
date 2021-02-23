# This file is part of afw.
#
# Developed for the LSST Data Management System.
# This product includes software developed by the LSST Project
# (https://www.lsst.org).
# See the COPYRIGHT file at the top-level directory of this distribution
# for details of code ownership.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""This file only exists to deprecate the Filter and FilterProperty classes.
"""
import warnings
from lsst.utils import continueClass
from ._imageLib import Filter, FilterProperty

__all__ = ['Filter', 'FilterProperty']


@continueClass
class Filter:
    # NOTE: Using `deprecate_pybind11` causes the `AUTO` and `UNKNOWN` static
    # members to become `pybind11_builtins.pybind11_static_property` instead
    # of `int`, which breaks any uses of them in the python layer, so we
    # deprecate this way instead.
    _init = Filter.__init__
    _reset = Filter.reset
    _define = Filter.define
    _getNames = Filter.getNames
    _deprecate_warning = "Replaced by FilterLabel. Will be removed after v22."

    def __init__(self, *args, **kwargs):
        warnings.warn(self._deprecate_warning, FutureWarning, stacklevel=2)
        return self._init(*args, **kwargs)

    @staticmethod
    def reset(*args, **kwargs):
        warnings.warn(Filter._deprecate_warning, FutureWarning, stacklevel=2)
        Filter._reset(*args, **kwargs)

    @staticmethod
    def define(*args, **kwargs):
        warnings.warn(Filter._deprecate_warning, FutureWarning, stacklevel=2)
        return Filter._define(*args, **kwargs)

    @staticmethod
    def getNames(*args, **kwargs):
        warnings.warn(Filter._deprecate_warning, FutureWarning, stacklevel=2)
        return Filter._getNames(*args, **kwargs)


@continueClass
class FilterProperty:
    # NOTE: Using `deprecate_pybind11` breaks the ability to call the `reset`
    # static method so we deprecate this way instead.
    _init = FilterProperty.__init__
    _lookup = FilterProperty.lookup
    _reset = FilterProperty.reset
    _deprecate_warning = ("Removed with no replacement (but see lsst.afw.image.TransmissionCurve)."
                          "Will be removed after v22.")

    def __init__(self, *args, **kwargs):
        warnings.warn(self._deprecate_warning, FutureWarning, stacklevel=2)
        return self._init(*args, **kwargs)

    @staticmethod
    def reset(*args, **kwargs):
        warnings.warn(FilterProperty._deprecate_warning, FutureWarning, stacklevel=2)
        FilterProperty._reset(*args, **kwargs)

    @staticmethod
    def lookup(*args, **kwargs):
        warnings.warn(FilterProperty._deprecate_warning, FutureWarning, stacklevel=2)
        return FilterProperty._lookup(*args, **kwargs)
