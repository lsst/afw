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
from lsst.utils import continueClass
from ._apCorrMap import ApCorrMap

__all__ = ['ApCorrMap']


@continueClass  # noqa: F811 (FIXME: remove for py 3.8+)
class ApCorrMap:  # noqa: F811

    def keys(self):
        for item in self.items():
            yield item[0]

    def values(self):
        for item in self.items():
            yield item[1]

    __iter__ = keys
