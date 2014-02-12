import lsst.afw.geom as afwGeom
import lsst.afw.cameraGeom as cameraGeom

class DetectorWrapper(object):
    """Construct a detector, with various errors possible
    """
    def __init__(self, numAmps=3, pixelSize=0.02, ampExtent=afwGeom.Extent2I(5,6), 
                 offset=afwGeom.Point2D(0., 0.), tryDuplicateAmpNames=False, tryBadCameraSys=False):
        self.name = "detector 1"
        self.type = cameraGeom.SCIENCE
        self.serial = "xkcd722"
        self.ampList = []
        for i in range(numAmps):
            ampName = "amp %d" % (i + 1,)
            if i == 1 and tryDuplicateAmpNames:
                ampName = self.ampList[0].getName()
            bbox = afwGeom.Box2I(afwGeom.Point2I(-1, 1), ampExtent)
            gain = 1.71234e3
            readNoise = 0.521237e2
            self.ampList.append(cameraGeom.Amplifier(ampName, bbox, gain, readNoise, None))
        self.orientation = cameraGeom.Orientation(offset)
        self.pixelSize = pixelSize
        self.transMap = {
            cameraGeom.FOCAL_PLANE: self.orientation.makeFpPixelTransform(afwGeom.Extent2D(self.pixelSize, self.pixelSize)),
            cameraGeom.CameraSys(cameraGeom.ACTUAL_PIXELS, self.name): afwGeom.RadialXYTransform([0, 0.95, 0.01]),
        }
        if tryBadCameraSys:
            self.transMap[cameraGeom.CameraSys("foo", "wrong detector")] = afwGeom.IdentityXYTransform(False)
        self.detector = cameraGeom.Detector(
            self.name,
            self.type,
            self.serial,
            self.ampList,
            self.orientation,
            self.pixelSize,
            self.transMap,
        )
