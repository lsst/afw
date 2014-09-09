
from .tableLib import SourceCatalog
import lsst.afw.geom as afwGeom


def sourceRecalibrate(sources, wcs, angleUnit=afwGeom.arcseconds, inPlace=False, shapesToConvert=None):
    """Convert shapes in each source in a source catalog to be in specified anglular units

    @param sources         SourceCatalog object to be converted
    @param wcs             Wcs object to use to orient moments to celestial north
    @param angleUnit       afwGeom angle to use for moments
    @param inPlace         Overwrite existing values with the new ones.
    @param shapeToConvert  List or tuple of shape names (strings) to convert (e.g. shape.sdss, shape.sdss.pfs)

    @return A new converted SourceCatalog, or the original changed insite (as per isPlace parameter)
    """
 
    outSources = sources if inPlace else sources.cast(SourceCatalog, deep=True)

    if shapesToConvert:
        for s in outSources:
            trans = wcs.linearizePixelToSky(s.getCentroid(), angleUnit).getLinear()
            for shape in shapesToConvert:
                moment = s.get(shape).transform(trans)
                s.set(shape, moment)

    return outSources

