import lsst.afw.geom as afwGeom
import lsst.afw.table as afwTable
import lsst.afw.cameraGeom as cameraGeom

__all__ = ["DetectorWrapper"]

class DetectorWrapper(object):
    """Construct a simple detector

    Intended for use with unit tests, thus saves a copy of all input parameters.
    Does not support setting details of amplifiers.

    @param[in] name: detector name
    @param[in] detType: detector type
    @param[in] serial: serial "number" (a string)
    @param[in] bbox: bounding box; defaults to something sensible
    @param[in] numAmps: number of amplifiers
    @param[in] pixelSize: pixel size (mm)
    @param[in] ampExtent: dimensions of amplifier image bbox
    @param[in] orientation: orientation of CCC in focal plane (lsst.afw.cameraGeom.Orientation)
    @param[in] tryDuplicateAmpNames: create 2 amps with the same name (should result in an error)
    @param[in] tryBadCameraSys: add a transform for an unsupported coord. system (should result in an error)
    """
    def __init__(self,
        name = "detector 1",
        detType = cameraGeom.SCIENCE,
        serial = "xkcd722",
        bbox = None,    # do not use mutable objects as defaults
        numAmps = 3,
        pixelSize = afwGeom.Extent2D(0.02),
        ampExtent = afwGeom.Extent2I(5,6), 
        orientation = cameraGeom.Orientation(),
        tryDuplicateAmpNames = False,
        tryBadCameraSys = False,
    ):
        # note that (0., 0.) for the reference position is the center of the first pixel
        self.name = name
        self.type = detType
        self.serial = serial
        if bbox is None:
            bbox = afwGeom.Box2I(afwGeom.Point2I(0, 0), afwGeom.Extent2I(1024, 1048))
        self.bbox = bbox
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
            record.setReadoutCorner(afwTable.LL)
            record.setHasRawInfo(False)
        self.orientation = orientation
        self.pixelSize = pixelSize
        self.transMap = {
            cameraGeom.FOCAL_PLANE: self.orientation.makePixelFpTransform(self.pixelSize),
            cameraGeom.CameraSys(cameraGeom.ACTUAL_PIXELS, self.name): afwGeom.RadialXYTransform([0, 0.95, 0.01]),
        }
        if tryBadCameraSys:
            self.transMap[cameraGeom.CameraSys("foo", "wrong detector")] = afwGeom.IdentityXYTransform()
        self.detector = cameraGeom.Detector(
            self.name,
            self.type,
            self.serial,
            self.bbox,
            self.ampInfo,
            self.orientation,
            self.pixelSize,
            self.transMap,
        )
