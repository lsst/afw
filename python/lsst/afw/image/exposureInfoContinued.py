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
"""This file only exists to deprecate the get/setFilter methods.
"""
from lsst.utils.deprecated import deprecate_pybind11

from . import ExposureInfo

ExposureInfo.getFilter = deprecate_pybind11(
    ExposureInfo.getFilter,
    reason="Replaced by getFilterLabel. Will be removed after v22.",
    version="v22.0")

ExposureInfo.setFilter = deprecate_pybind11(
    ExposureInfo.setFilter,
    reason="Replaced by setFilterLabel. Will be removed after v22.",
    version="v22.0")
