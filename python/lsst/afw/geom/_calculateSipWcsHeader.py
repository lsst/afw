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


from lsst.geom import Extent2I, Box2D
from ._geom import SipApproximation, getPixelToIntermediateWorldCoords
from ._geom import makeTanSipMetadata

__all__ = ("calculateSipWcsHeader",)


def calculateSipWcsHeader(wcs, order, bbox, spacing, header=None):
    """Generate a SIP WCS header approximating a given ``SkyWcs``

    Parameters
    ----------
    wcs : `lsst.afw.geom.SkyWcs`
        World Coordinate System to approximate as SIP.
    order : `int`
        SIP order (equal to the maximum sum of the polynomial exponents).
    bbox : `lsst.geom.Box2I`
        Bounding box over which to approximate the ``wcs``.
    spacing : `float`
        Spacing between sample points.
    header : `lsst.daf.base.PropertyList`, optional
        Header to which to add SIP WCS keywords.

    Returns
    -------
    header : `lsst.daf.base.PropertyList`
        Header including SIP WCS keywords.

    Examples
    --------
    >>> header = calculateSipWcsHeader(exposure.getWcs(), 3, exposure.getBBox(), 20)
    >>> sipWcs = SkyWcs(header)
    """
    transform = getPixelToIntermediateWorldCoords(wcs)
    crpix = wcs.getPixelOrigin()
    cdMatrix = wcs.getCdMatrix()
    crval = wcs.getSkyOrigin()
    gridNum = Extent2I(int(bbox.getWidth()/spacing + 0.5), int(bbox.getHeight()/spacing + 0.5))

    sip = SipApproximation(transform, crpix, cdMatrix, Box2D(bbox), gridNum, order)

    md = makeTanSipMetadata(sip.getPixelOrigin(), crval, sip.getCdMatrix(), sip.getA(), sip.getB(),
                            sip.getAP(), sip.getBP())

    if header is not None:
        header.combine(md)
    else:
        header = md

    return header
