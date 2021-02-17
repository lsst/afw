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

from ._geom import TransformGenericToPoint2
from ._geom import TransformGenericToSpherePoint
from ._geom import TransformGenericToGeneric
from ._geom import TransformPoint2ToGeneric
from ._geom import TransformPoint2ToSpherePoint
from ._geom import TransformPoint2ToPoint2
from ._geom import TransformSpherePointToGeneric
from ._geom import TransformSpherePointToPoint2
from ._geom import TransformSpherePointToSpherePoint

from ._python import addTransformMethods

__all__ = []


addTransformMethods(TransformGenericToPoint2)
addTransformMethods(TransformGenericToSpherePoint)
addTransformMethods(TransformGenericToGeneric)
addTransformMethods(TransformPoint2ToGeneric)
addTransformMethods(TransformPoint2ToSpherePoint)
addTransformMethods(TransformPoint2ToPoint2)
addTransformMethods(TransformSpherePointToGeneric)
addTransformMethods(TransformSpherePointToPoint2)
addTransformMethods(TransformSpherePointToSpherePoint)
