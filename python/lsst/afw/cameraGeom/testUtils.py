import lsst.afw.geom as afwGeom
import lsst.afw.table as afwTable
import lsst.afw.cameraGeom as cameraGeom

class DetectorWrapper(object):
    """Construct a detector, with various errors possible
    """
    def __init__(self, numAmps=3, pixelSize=afwGeom.Extent2D(0.02), ampExtent=afwGeom.Extent2I(5,6), 
                 offset=afwGeom.Point2D(0., 0.), refposition=afwGeom.Point2D(0., 0.), 
                 tryDuplicateAmpNames=False, tryBadCameraSys=False):
        # note that (0., 0.) for the reference position is the center of the first pixel
        self.name = "detector 1"
        self.type = cameraGeom.SCIENCE
        self.serial = "xkcd722"
        schema = afwTable.AmpInfoTable.makeMinimalSchema()
        self.ampInfo = afwTable.AmpInfoCatalog(schema)
        for i in range(numAmps):
            record = self.ampInfo.addNew()
            ampName = "amp %d" % (i + 1,)
            if i == 1 and tryDuplicateAmpNames:
                ampName = self.ampInfo[0].getName()
            record.setName(ampName)
            record.setBBox(afwGeom.Box2I(afwGeom.Point2I(-1, 1), ampExtent))
            record.setGain(1.71234e3)
            record.setReadNoise(0.521237e2)
            record.setHasRawInfo(False)
        self.orientation = cameraGeom.Orientation(offset, refposition)
        self.pixelSize = pixelSize
        self.transMap = {
            cameraGeom.FOCAL_PLANE: self.orientation.makeFpPixelTransform(self.pixelSize),
            cameraGeom.CameraSys(cameraGeom.ACTUAL_PIXELS, self.name): afwGeom.RadialXYTransform([0, 0.95, 0.01]),
        }
        if tryBadCameraSys:
            self.transMap[cameraGeom.CameraSys("foo", "wrong detector")] = afwGeom.IdentityXYTransform(False)
        self.detector = cameraGeom.Detector(
            self.name,
            self.type,
            self.serial,
            self.ampInfo,
            self.orientation,
            self.pixelSize,
            self.transMap,
        )
