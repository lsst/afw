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

__all__ = ["SimpleGenericMap"]

from lsst.utils import continueClass, TemplateMeta
from ._typehandling import SimpleGenericMapS


class SimpleGenericMap(metaclass=TemplateMeta):
    """A `dict`-like `~collections.abc.MutableMapping` for use when sharing a
    map between C++ and Python.

    For compatibility with C++, ``SimpleGenericMap`` has the following
    restrictions:

        - all keys must be of the same type
        - values must be built-in types or subclasses of
          `lsst.afw.typehandling.Storable`. Almost any user-defined class in
          C++ or Python can have `~lsst.afw.typehandling.Storable` as a mixin.

    As a safety precaution, `~lsst.afw.typehandling.Storable` objects that are
    added from C++ may be copied when you retrieve them from Python, making it
    impossible to modify them in-place. This issue does not affect objects that
    are added from Python, or objects that are always passed by
    :cpp:class:`shared_ptr` in C++.

    Parameters
    ----------
    mapping : `collections.abc.Mapping`, optional
    iterable : iterable, optional
    dtype : `type`
        The type of key the map accepts.
    **kwargs
        Aside from the ``dtype`` keyword, a ``SimpleGenericMap`` takes the same
        input arguments as `dict`.
    """
    @classmethod
    def fromkeys(cls, iterable, value=None):
        return cls({key: value for key in iterable})


SimpleGenericMap.register(str, SimpleGenericMapS)


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
