import re
import lsst.afw.detection as afwDetection
import lsst.afw.image as afwImage

def clipImage(im, min, max):
    """Clip an image to lie between min and max (None to ignore)"""

    if re.search("::MaskedImage<", im.__repr__()):
        mi = im
    else:
        mi = afwImage.makeMaskedImage(im, afwImage.MaskU(im.getDimensions()))

    if min is not None:
        ds = afwDetection.makeFootprintSet(mi, afwDetection.Threshold(-min, afwDetection.Threshold.VALUE, False))
        afwDetection.setImageFromFootprintList(mi.getImage(), ds.getFootprints(), min)

    if max is not None:
        ds = afwDetection.makeFootprintSet(mi, afwDetection.Threshold(max))
        afwDetection.setImageFromFootprintList(mi.getImage(), ds.getFootprints(), max)
