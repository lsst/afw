import lsst.afw.image as afwImage

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def makeMosaic(*args):
    """Return mosaic of zero or more identically-sized images (or Masks or MaskedImages)"""
    gutter = 3

    nImage = len(args)
    image1 = args[0]
    mosaic = image1.Factory((nImage - 1)*gutter + nImage*image1.getWidth(), image1.getHeight())
    mosaic.set(10)                      # gutter value

    for i in range(len(args)):
        image = args[i]
        smosaic = mosaic.Factory(mosaic, afwImage.BBox(afwImage.PointI(i*(gutter + image1.getWidth()), 0),
                                                       image1.getWidth(), image1.getHeight()))
        smosaic <<= image

    return mosaic

