import lsst.afw.image as afwImage
import lsst.pex.config as pexConfig
import lsst.afw.cameraGeom as afwCameraGeom
import lsst.afw.geom as afwGeom

class CameraGeomBuilderConfig(pexConfig.Config):
    raise NotImplementedError()

class HeaderMap(dict):
    """ Class to hold mapping of header cards to attributes"""
    def __init__(self, keyname, attribute_name, default, transform=None):
        self.keyname = keyname
        self.attribute_name = attribute_name
        self.transform = transform
        self.default = default

    def addEntry(self, keyname, attribute_name, default=None, transform=None):
        self.__setitem__(self, keyname, {'attrName':attribute_name,
                                         'default':default,
                                         'transform':transform})

    def setAttributes(self, obj, metadata):
        for key, attrDict in self.iteritems():
            value = metadata.get(key)
            if value is not None:
                self._applyVal(obj, value, attrDict['attrName'], attrDict['transform'])
            else:
                #Only apply transform if the metadata has a value for this key
                #otherwise assume the default value is transformed.
                self._applyVal(obj, value, attrDict['attrName'], None)

    def _applyVal(self, obj, value, attrName, transform):
        raise NotImplementedError('Must be implemented in sub-class')

class HeaderAmpMap(HeaderMap):
    """ Class to hold mapping of header cards to AmpInfoTable attributes"""
    def _applyVal(self, obj, value, attrName, transform):
        obj.get_attribute(attrName)(transform(value))

class HeaderDetectorMap(HeaderMap):
    """ Class to hold mapping of header cards to Detector attributes"""
    def _applyVal(self, obj, value, attrName, transform):
        obj.__setattr__(attrName, transform(value))

class CameraGeomBuilder(object):
    def __init__(self, fileNameList):
        self.defaultAmpMap = self._makeDefaultAmpMap()
        self.defaultDetectorMap = self._makeDefaultDetectorMap()
        self.mdList = []
        self.detectorList = []
        for fileName in fileNameList:
            self.mdList.append(afwImage.readFitsMetadata(fileName))
            self.detectorList.append(CameraGeomBuilder.buildDetector(self.mdList[-1]))
        self.camera = self.buildCamera()

    def _makeDefaultAmpMap(self):
        raise NotImplementedError()

    def _makeDefaultAmpMap(self):
        hMap = HeaderAmpMap()
        emptyBBox = afwGeom.Box2()
        mapList = [('EXTNAME', 'setName', None, None),
                   ('DETSEC', 'setBBox', None, self.makeBbox),
                   ('GAIN', 'setGain', 1., None),
                   ('RDNOISE', 'setReadnoise', 0., None),
                   ('SATURATE', 'setSatruration', 2<<15, None),
                   ('RDCRNR', 'setReadCorner', afwCameraGeom.LLC, None),
                   ('LINCOEFF', 'setLinearityCoeffs', [0., 1.], None),
                   ('LINTYPE', 'setLinearityType', 'POLY', None),
                   ('RAWBBOX', 'setRawBBox', None, self.makeBbox),
                   ('DATASIC', 'setRawDataBBox', None, self.makeBbox),
                   ('FLIPX', 'setFlipX', False, None),
                   ('FLIPY', 'setFlipY', False, None),
                   ('XYOFF', 'setRawXYOffset', [0,0], None),
                   ('HOSCAN', 'setRawHorizontalOverscanBbox', emptyBBox, self.makeBbox),
                   ('VOSCAN', 'setRawHorizontalOverscanBbox', emptyBBox, self.makeBbox),
                   ('PRESCAN', 'setRawHorizontalOverscanBbox', emptyBBox, self.makeBbox),
                   ]
        for tup in mapList:
            hMap.addEntry(*tup)
        return hMap

    def _makeDefaultDetectorMap(self):
        hMap = HeaderDetectorMap()
        mapList = [('CCDNAME', 'name', None, None),
                   ('DETSIZE', 'bbox_x0', None, self._getBboxX0),
                   ('DETSIZE', 'bbox_y0', None, self._getBboxY0),
                   ('DETSIZE', 'bbox_x1', None, self._getBboxX1),
                   ('DETSIZE', 'bbox_y1', None, self._getBboxY1),
                   ('OBSTYPE', 'detectorType', afwCameraGeom.SCIENCE, None),
                   ('SERSTR', 'serial', 'none', None),
                   ('XPOS', 'offset_x', 0., None),
                   ('YPOS', 'offset_y', 0., None),
                   ('XPIX', 'refpos_x', 0., None),
                   ('YPIX', 'refpos_y', 0., None),
                   ('YAWDEG', 'yawDeg', 0., None),
                   ('PITCHDEG', 'pitchDeg', 0., None),
                   ('ROLLDEG', 'rollDeg', 0., None),
                   ('XPIXSIZE', 'pixelSize_x', None),
                   ('YPIXSIZE', 'pixelSize_y', None),
                   ('TRNSPOSE', 'transposeDetector', False),
                   ]
        for tup in mapList:
            hMap.addEntry(*tup)
        return hMap

    def _makeBbox(boxString):
        raise NotImplementedError()

    @staticmethod
    def buildDetector(metadata):
        raise NotImplementedError()

    def buildCamera(self):
        raise NotImplementedError()
