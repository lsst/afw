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
#

__all__ = []

from lsst.utils import continueClass
from .detectorCollection import DetectorCollection


@continueClass  # noqa: F811
class DetectorCollection:
    """!An immutable collection of Detectors that can be accessed by name or ID
    """

    def __iter__(self):
        for k, v in sorted(self.getIdMap().items()):
            yield v

    def __getitem__(self, key):
        r = self.get(key)
        if r is None:
            raise KeyError("Detector for key {} not found.".format(key))
        return r

    def getNameIter(self):
        """Get an iterator over detector names
        """
        for k, v in self.getNameMap().items():
            yield k

    def getIdIter(self):
        """Get an iterator over detector IDs
        """
        for k, v in self.getIdMap().items():
            yield k
