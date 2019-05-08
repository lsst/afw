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

__all__ = []

from lsst.utils import continueClass
from ._typehandling import SimpleGenericMapS


# pybind11-generated constructor, can only create empty map
_oldInit = SimpleGenericMapS.__init__


@continueClass  # noqa F811
class SimpleGenericMapS:
    def __init__(self, source=None, **kwargs):
        _oldInit(self)
        if source:
            self.update(source, **kwargs)
        else:
            self.update(**kwargs)

    @classmethod
    def fromkeys(cls, iterable, value=None):
        mapping = cls()
        mapping.update({key: value for key in iterable})
        return mapping
