import re
import lsst.afw.detection as afwDetect
import lsst.afw.image as afwImage

def clipImage(im, minClip, maxClip):
    """Clip an image to lie between minClip and maxclip (None to ignore)"""

    if re.search("::MaskedImage<", im.__repr__()):
        mi = im
    else:
        mi = afwImage.makeMaskedImage(im, afwImage.MaskU(im.getDimensions()))

    if minClip is not None:
        ds = afwDetect.makeFootprintSet(mi, afwDetect.Threshold(-minClip, afwDetect.Threshold.VALUE, False))
        afwDetect.setImageFromFootprintList(mi.getImage(), ds.getFootprints(), minClip)

    if maxclip is not None:
        ds = afwDetect.makeFootprintSet(mi, afwDetect.Threshold(maxclip))
        afwDetect.setImageFromFootprintList(mi.getImage(), ds.getFootprints(), maxclip)
