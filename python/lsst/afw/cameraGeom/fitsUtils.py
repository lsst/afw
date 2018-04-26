import re
import warnings
import lsst.afw.image as afwImage
import lsst.afw.table as afwTable
from lsst.afw.fits import readMetadata
import lsst.afw.cameraGeom as afwCameraGeom
import lsst.afw.geom as afwGeom


def getByKey(metadata, key):
    """Wrapper for getting a value from a metadata object by key.
       @param[in] metadata  metadata object to query for value
       @param[in] key       key to use for value lookup
       @returns value associated with key, None if key does not exist
    """
    mdKeys = metadata.paramNames()
    if key in mdKeys:
        return metadata.get(key)
    else:
        return None


def setByKey(metadata, key, value, clobber):
    """Wrapper for setting a value in a metadata object.  Deals with case
       where the key already exists.
       @param[in, out] metadata  metadata object ot modify in place.
       @param[in] key       key to associate with value
       @param[in] value     value to assign in the metadata object
       @param[in] clobber   Clobber the value if the key already exisists?
    """
    mdKeys = metadata.paramNames()
    if key not in mdKeys or (key in mdKeys and clobber):
        metadata.set(key, value)


class HeaderMap(dict):
    """ Class to hold mapping of header cards to attributes"""

    def addEntry(self, keyname, attribute_name, default=None, transform=lambda x: x):
        """Adds an entry to the registr
           @param[in] keyname         Key used to retrieve the header record
           @param[in] attribute_name  Name of the attribute to store the value in
           @param[jn] default         Default velue to store if the header card is not available
           @param[in] transform       Transform to apply to the header value before assigning it to the
                                      attribute.
        """
        self.__setitem__(attribute_name, {'keyName': keyname,
                                          'default': default,
                                          'transform': transform})

    def setAttributes(self, obj, metadata, doRaise=True):
        """Sets the attributes on the give object given a metadata object.
           @param[in, out] obj       Object on which to operate in place
           @param[in]      metadata  Metadata object used for applying the mapping
           @param[in]      doRaise   Raise exceptions on calling methods on the
                                     input object that do not exist?
        """
        for key, attrDict in self.items():
            try:
                value = getByKey(metadata, attrDict['keyName'])
                # if attrDict['keyName'] == "RDCRNR" and value == 0:
                #     import ipdb; ipdb.set_trace()
                if value is not None:
                    self._applyVal(obj, value, key, attrDict['transform'])
                else:
                    # Only apply transform if the metadata has a value for this key
                    # otherwise assume the default value is transformed.
                    value = attrDict['default']
                    if value is not None:
                        self._applyVal(obj, value, key, lambda x: x)
            except Exception as e:
                if doRaise:
                    raise
                else:
                    warnings.warn('WARNING: Failed to set %s attribute with %s value: %s' %
                                  (key, value, str(e)))

    def _applyVal(self, obj, value, attrName, transform):
        raise NotImplementedError('Must be implemented in sub-class')


class HeaderAmpMap(HeaderMap):
    """ Class to hold mapping of header cards to AmpInfoTable attributes
        The amp info is stored using setters, thus calling the attribute as a function.
    """

    def _applyVal(self, obj, value, attrName, transform):
        getattr(obj, attrName)(transform(value))


class HeaderDetectorMap(HeaderMap):
    """ Class to hold mapping of header cards to Detector attributes
        Detector information is stored as attributes on a Config object.
    """

    def _applyVal(self, obj, value, attrName, transform):
        obj.__setattr__(attrName, transform(value))


class DetectorBuilder:
    def __init__(self, detectorFileName, ampFileNameList, inAmpCoords=True, plateScale=1.,
                 radialCoeffs=(0., 1.), clobberMetadata=False, doRaise=True):
        ''' @param[in] detectorFileName  FITS file containing the detector description.
                                         May use [] notation to specify an extension in an MEF.
            @param[in] ampFileNameList   List of FITS file names to use in building the amps.
                                         May contain duplicate entries if the raw data are assembled.
            @param[in] inAmpCoords       Boolean, True if raw data are in amp coordinates, False if raw data
                                         are assembled into pseudo detector pixel arrays
            @param[in] plateScale        Nominal platescale (arcsec/mm)
            @param[in] radialCoeffs      Radial distortion coefficients for a radial polynomial in normalized
                                         units.
            @param[in] clobberMetadata   Clobber metadata from input files if
                                         overridden in the _sanitizeMetadata method
            @param[in] doRaise           Raise exception if not all non-defaulted
                                         keywords are defined?  Default is True.
        '''
        self.inAmpCoords = inAmpCoords
        self.defaultAmpMap = self._makeDefaultAmpMap()
        self.defaultDetectorMap = self._makeDefaultDetectorMap()
        self.detectorMetadata = readMetadata(detectorFileName)
        self._sanitizeHeaderMetadata(
            self.detectorMetadata, clobber=clobberMetadata)
        self.ampMetadataList = []
        self.detector = None
        self.doRaise = doRaise
        for fileName in ampFileNameList:
            self.ampMetadataList.append(readMetadata(fileName))
            self._sanitizeHeaderMetadata(
                self.ampMetadataList[-1], clobber=clobberMetadata)
        self.plateScale = plateScale
        self.focalPlaneToField = self._makeRadialTransform(radialCoeffs)

    def _sanitizeHeaderMetadata(self, metadata, clobber):
        """This method is called for all metadata and gives an opportunity to add/modify
           header information for use downstream.
           Override this method if more than the default is needed.
           @param[in, out] metadata  Metadata to read/modify
           @param[in]      clobber   Clobber keys that exist with default keys?
        """
        self._defaultSanitization(metadata, clobber)

    def _defaultSanitization(self, metadata, clobber):
        """Does the default sanitization of the header metadata.
           @param[in,out] metadata  Header metadata to extend/modify
           @param[in]     clobber   Override values in existing header cards?
        """

        if self.inAmpCoords:
            # Deal with DTM to get flipX and flipY for assembly and add as 'FLIPX', 'FLIPY'
            # The DTM array is a transformation matrix.  As I understand it, it transforms between
            # electronic and assembled coordintates.  As such, a negative value in the DTM1_1 spot
            # corresponds to a flip of the x-axis and a negative value in the DTM2_2 spot
            # corresponds to a flip of the y-axis.
            dtm1 = getByKey(metadata, 'DTM1_1')
            dtm2 = getByKey(metadata, 'DTM2_2')
            if dtm1 is not None and dtm2 is not None:
                setByKey(metadata, 'FLIPX', dtm1 < 0, clobber)
                setByKey(metadata, 'FLIPY', dtm2 < 0, clobber)
            setByKey(metadata, 'RDCRNR', int(
                afwTable.ReadoutCorner.LL), clobber)
        else:
            setByKey(metadata, 'FLIPX', False, clobber)
            setByKey(metadata, 'FLIPY', True, clobber)
            # I don't know how to figure out the read corner if already
            # assembled
            setByKey(metadata, 'RDCRNR', None, clobber)

        # Deal with NAXIS1, NAXIS2 to make rawBBox as 'RAWBBOX'
        xext = getByKey(metadata, 'NAXIS1')
        yext = getByKey(metadata, 'NAXIS2')
        if xext is not None and yext is not None:
            setByKey(metadata, 'RAWBBOX', '[%i:%i,%i:%i]'%(
                1, xext, 1, yext), clobber)
        # Deal with DTV1, DTV2 to make 'XYOFF
        dtv1 = getByKey(metadata, 'DTV1')
        dtv2 = getByKey(metadata, 'DTV2')
        if dtv1 is not None and dtv2 is not None:
            setByKey(metadata, 'XYOFF', [dtv1, dtv2], clobber)
        # map biassec[1] to HOSCAN
        # map biassec[3] to VOSCAN
        # map biassec[2] to PRESCAN
        if metadata.isArray('BIASSEC'):
            keylist = ['HOSCAN', 'PRESCAN', 'VOSCAN']
            biassecs = getByKey(metadata, 'BIASSEC')
            for i, biassec in enumerate(biassecs):
                setByKey(metadata, keylist[i], biassec, clobber)
        else:
            biassec = getByKey(metadata, 'BIASSEC')
            if biassec is not None:
                setByKey(metadata, 'HOSCAN', biassec, clobber)

    def _makeDefaultAmpMap(self):
        """Make the default map from header information to amplifier information
           @return  The HeaderAmpMap object containing the mapping
        """
        hMap = HeaderAmpMap()
        emptyBBox = afwGeom.BoxI()
        mapList = [('EXTNAME', 'setName'),
                   ('DETSEC', 'setBBox', None, self._makeBbox),
                   ('GAIN', 'setGain', 1.),
                   ('RDNOISE', 'setReadNoise', 0.),
                   ('SATURATE', 'setSaturation', 2 << 15),
                   ('RDCRNR', 'setReadoutCorner', int(
                       afwTable.ReadoutCorner.LL), afwTable.ReadoutCorner),
                   ('LINCOEFF', 'setLinearityCoeffs', [0., 1.]),
                   ('LINTYPE', 'setLinearityType', 'POLY'),
                   ('RAWBBOX', 'setRawBBox', None, self._makeBbox),
                   ('DATASEC', 'setRawDataBBox', None, self._makeBbox),
                   ('FLIPX', 'setRawFlipX', False),
                   ('FLIPY', 'setRawFlipY', False),
                   ('XYOFF', 'setRawXYOffset', afwGeom.ExtentI(0, 0), self._makeExt),
                   ('HOSCAN', 'setRawHorizontalOverscanBBox',
                    emptyBBox, self._makeBbox),
                   ('VOSCAN', 'setRawVerticalOverscanBBox',
                    emptyBBox, self._makeBbox),
                   ('PRESCAN', 'setRawPrescanBBox', emptyBBox, self._makeBbox),
                   ]
        for tup in mapList:
            hMap.addEntry(*tup)
        return hMap

    def _makeDefaultDetectorMap(self):
        """Make the default map from header information to detector information
           @return  The HeaderDetectorMap object containing the mapping
        """
        hMap = HeaderDetectorMap()
        mapList = [('CCDNAME', 'name', 'ccdName'),
                   ('DETSIZE', 'bbox_x0', 0, self._getBboxX0),
                   ('DETSIZE', 'bbox_y0', 0, self._getBboxY0),
                   ('DETSIZE', 'bbox_x1', 0, self._getBboxX1),
                   ('DETSIZE', 'bbox_y1', 0, self._getBboxY1),
                   ('DETID', 'id', 0),
                   # DetectorConfig.detectorType is of type `int`, not
                   # `DetectorType`
                   ('OBSTYPE', 'detectorType', int(
                       afwCameraGeom.DetectorType.SCIENCE)),
                   ('SERSTR', 'serial', 'none'),
                   ('XPOS', 'offset_x', 0.),
                   ('YPOS', 'offset_y', 0.),
                   ('XPIX', 'refpos_x', 0.),
                   ('YPIX', 'refpos_y', 0.),
                   ('YAWDEG', 'yawDeg', 0.),
                   ('PITCHDEG', 'pitchDeg', 0.),
                   ('ROLLDEG', 'rollDeg', 0.),
                   ('XPIXSIZE', 'pixelSize_x', 1.),
                   ('YPIXSIZE', 'pixelSize_y', 1.),
                   ('TRNSPOSE', 'transposeDetector', False),
                   ]
        for tup in mapList:
            hMap.addEntry(*tup)
        return hMap

    def _makeExt(self, extArr):
        """Helper function to make an extent from an array
           @param[in] extArr Length 2 array to use in creating the Extent object
           @return  Extent2I constructed from the input list
        """
        return afwGeom.ExtentI(*extArr)

    def _makeBbox(self, boxString):
        """Helper funtion to make a bounding box from a string representing a FITS style bounding box
           @param[in] boxString  String describing the bounding box
           @return    Box2I for the bounding box
        """
        # strip off brackets and split into parts
        x1, x2, y1, y2 = [int(el) for el in re.split(
            '[:,]', boxString.strip()[1:-1])]
        box = afwGeom.BoxI(afwGeom.PointI(x1, y1), afwGeom.PointI(x2, y2))
        # account for the difference between FITS convention and LSST convention for
        # index of LLC.
        box.shift(afwGeom.Extent2I(-1, -1))
        return box

    def _getBboxX0(self, boxString):
        return self._makeBbox(boxString).getMinX()

    def _getBboxX1(self, boxString):
        return self._makeBbox(boxString).getMaxX()

    def _getBboxY0(self, boxString):
        return self._makeBbox(boxString).getMinY()

    def _getBboxY1(self, boxString):
        return self._makeBbox(boxString).getMaxY()

    def _makeRadialTransform(self, radialCoeffs):
        """Helper function to get the radial transform given the radial polynomial coefficients given in
           the constructor.
           @param[in]  radialCoeffs  List of coefficients describing a polynomial radial distortion in
                                     normalized units.
           @return     Transform object describing the radial distortion
        """
        pScaleRad = afwGeom.arcsecToRad(self.plateScale)
        return afwGeom.makeRadialTransform([el/pScaleRad for el in radialCoeffs])

    def buildDetector(self):
        """Take all the information and build a Detector object.  The Detector object is necessary for doing
        things like assembly.
        @return  Detector object
        """
        if self.detector is not None:
            return self.detector

        schema = afwTable.AmpInfoTable.makeMinimalSchema()
        ampInfo = afwTable.AmpInfoCatalog(schema)
        for ampMetadata in self.ampMetadataList:
            record = ampInfo.addNew()
            self.defaultAmpMap.setAttributes(record, ampMetadata, self.doRaise)
            record.setHasRawInfo(True)

        detConfig = afwCameraGeom.DetectorConfig()
        self.defaultDetectorMap.setAttributes(
            detConfig, self.detectorMetadata, self.doRaise)
        self.detector = afwCameraGeom.makeDetector(
            detConfig, ampInfo, self.focalPlaneToField)
        return self.detector

    def makeCalib(self):
        """PLaceholder for subclasses to implement construction of a calib to associate with the exposure.
        @return empty afwImage.Calib object
        """
        return afwImage.Calib()

    def makeExposure(self, im, mask=None, variance=None):
        """Method for constructing an exposure object from an image and the information contained in this
           class to construct the Detector and Calib objects.
           @param[in]  im        Image used to construct the exposure
           @param[in]  mask      Optional mask plane as a <askU
           @param[in]  variance  Optional variance plance as an image of the same type as im
           @param[out] Exposure object
        """
        if mask is None:
            mask = afwImage.Mask(im.getDimensions())
        if variance is None:
            variance = im
        mi = afwImage.makeMaskedImage(im, mask, variance)
        detector = self.buildDetector()

        calib = self.makeCalib()
        exp = afwImage.makeExposure(mi)
        exp.setCalib(calib)
        exp.setDetector(detector)
        return exp
