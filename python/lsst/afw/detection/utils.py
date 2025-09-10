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

__all__ = ["writeFootprintAsDefects", "footprintsToNumpy"]

import numpy as np

import lsst.geom as geom
from lsst.afw.table import SourceCatalog

from . import footprintToBBoxList


def writeFootprintAsDefects(fd, foot):
    """
    Write foot as a set of Defects to fd

    Given a detection footprint, convert it to a BBoxList and write the output to the file object fd.

    Parameters
    ----------
    fd : `typing.TextIO`
    foot : `lsst.afw.detection.Footprint`

    See Also
    --------
    lsst.afw.detection.footprintToBBoxList
    """

    bboxes = footprintToBBoxList(foot)
    for bbox in bboxes:
        print("""\
Defects: {
    x0:     %4d                         # Starting column
    width:  %4d                         # number of columns
    y0:     %4d                         # Starting row
    height: %4d                         # number of rows
}""" % (bbox.getMinX(), bbox.getWidth(), bbox.getMinY(), bbox.getHeight()), file=fd)


def footprintsToNumpy(
    catalog: SourceCatalog,
    bbox: geom.Box2I | None = None,
    shape: tuple[int, int] | None = None,
    xy0: tuple[int, int] | None = None,
    asBool: bool = True,
) -> np.ndarray:
    """Convert all of the footprints in a catalog into a boolean array.

    Parameters
    ----------
    catalog:
        The source catalog containing the footprints.
        This is typically a mergeDet catalog, or a full source catalog
        with the parents removed.
    shape:
        The final shape of the output array.
    xy0:
        The lower-left corner of the array that will contain the spans.

    Returns
    -------
    result:
        The array with pixels contained in `spans` marked as `True`.
    """
    if bbox is None and shape is None:
        raise RuntimeError("Must provide either bbox or shape")

    if bbox is not None:
        width, height = bbox.getDimensions()
        shape = (height, width)
        xy0 = (bbox.getMinX(), bbox.getMinY())

    if xy0 is None:
        offset = (0, 0)
    else:
        offset = (-xy0[0], -xy0[1])

    result = np.zeros(shape, dtype=int)
    for src in catalog:
        spans = src.getFootprint().spans
        yidx, xidx = spans.shiftedBy(*offset).indices()
        result[yidx, xidx] = src.getId()
    if asBool:
        result = result != 0
    return result
