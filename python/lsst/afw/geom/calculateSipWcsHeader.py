from lsst.geom import Extent2I, Box2D
from lsst.afw.geom import SipApproximation, getPixelToIntermediateWorldCoords
from lsst.afw.geom.wcsUtils import makeTanSipMetadata

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
