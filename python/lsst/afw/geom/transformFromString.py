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

from ._python import transformRegistry

__all__ = ["transformFromString"]


def transformFromString(data):
    """Read a Transform from a string that was saved using
    ``Transform.writeString``

    Unlike ``Transform.readString``, you need not know the ``Transform`` class in
    advance

    Parameters
    ----------
    data : `str`
        A string in the format produced by ``Transform.writeString``.

    Returns
    -------
    transform : a ``Transform`` class
        An object of the same type used to create ``data``.
    """
    version, transformClassName, remainder = data.split(" ", 2)
    try:
        transformClass = transformRegistry[transformClassName]
    except LookupError:
        raise RuntimeError(f"Unknown transform class {transformClassName!r}")

    return transformClass.readString(data)
