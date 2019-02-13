# This file is part of afw.
#
# Developed for the LSST Data Management System.
# This product includes software developed by the LSST Project
# (https://www.lsst.org).
# See the COPYRIGHT file at the top-level directory of this distribution
# for details of code ownership.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

__all__ = ['getByKey', 'setByKey', 'HeaderMap', 'HeaderAmpMap',
           'HeaderDetectorMap', 'DetectorBuilder']

import re
import warnings

import lsst.geom
import lsst.afw.image as afwImage
import lsst.afw.table as afwTable
from lsst.afw.fits import readMetadata
import lsst.afw.cameraGeom as afwCameraGeom
import lsst.afw.geom


def getByKey(metadata, key):
    """Wrapper for getting a value from a metadata object by key.

    Parameters
    ----------
    metadata : `lsst.daf.base.PropertySet`
        Metadata object to query for value.
    key : `str`
        Key to use for value lookup.

   Returns
   -------
   value : `object`
       Value associated with key, None if key does not exist.
    """
    mdKeys = metadata.paramNames()
    if key in mdKeys:
        return metadata.getScalar(key)
    else:
        return None


def setByKey(metadata, key, value, clobber):
    """Wrapper for setting a value in a metadata object.  Deals with case
    where the key already exists.

    Parameters
    ----------
    metadata : `lsst.daf.base.PropertySet`
        Metadata object ot modify in place.
    key : `str`
        Key to associate with value.
    value : `object`
        Value to assign in the metadata object.
    clobber : `bool`
        Clobber the value if the key already exists?
    """
    mdKeys = metadata.paramNames()
    if key not in mdKeys or (key in mdKeys and clobber):
        metadata.set(key, value)


class HeaderMap(dict):
    """ Class to hold mapping of header cards to attributes"""

    def addEntry(self, keyname, attribute_name, default=None, transform=lambda x: x):
        """Adds an entry to the registry.

        Parameters
        ----------
        keyname : `str`
            Key used to retrieve the header record.
        attribute_name : `str`
            Name of the attribute to store the value in.
        default : `object`
            Default value to store if the header card is not available.
        transform : `callable`
            Transform to apply to the header value before assigning it to the
            attribute.
        """
        self.__setitem__(attribute_name, {'keyName': keyname,
                                          'default': default,
                                          'transform': transform})

    def setAttributes(self, obj, metadata, doRaise=True):
        """Sets the attributes on the given object given a metadata object.

        Parameters
        ----------
        obj : `object`
            Object on which to operate in place.
        metadata : `lsst.daf.base.PropertySet`
            Metadata object used for applying the mapping.
        doRaise : `bool`
            Raise exceptions on calling methods on the input object that
            do not exist?
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
    """
    Parameters
    ----------
    detectorFileName : `str`
        FITS file containing the detector description.
        May use [] notation to specify an extension in an MEF.
    ampFileNameList : `list` of `str`
        List of FITS file names to use in building the amps.
        May contain duplicate entries if the raw data are assembled.
    inAmpCoords : `bool`
        True if raw data are in amp coordinates, False if raw data
        are assembled into pseudo detector pixel arrays.
    plateScale : `float`
        Nominal platescale (arcsec/mm).
    radialCoeffs : `iterable` of `float`
        Radial distortion coefficients for a radial polynomial in
        normalized units.
    clobberMetadata : `bool`
        Clobber metadata from input files if overridden in
        _sanitizeMetadata().
    doRaise : `bool`
        Raise exception if not all non-defaulted keywords are defined?
    """
    def __init__(self, detectorFileName, ampFileNameList, inAmpCoords=True, plateScale=1.,
                 radialCoeffs=(0., 1.), clobberMetadata=False, doRaise=True):
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
        """This method is called for all metadata and gives an opportunity to
        add/modify header information for use downstream.

        Override this method if more than the default is needed.

        Parameters
        ----------
        metadata : `lsst.daf.base.PropertySet`
            Metadata to read/modify
        clobber : `bool`
            Clobber keys that exist with default keys?
        """
        self._defaultSanitization(metadata, clobber)

    def _defaultSanitization(self, metadata, clobber):
        """Does the default sanitization of the header metadata.

        Parameters
        ----------
        metadata : `lsst.daf.base.PropertySet`
            Header metadata to extend/modify.
        clobber : `bool`
            Override values in existing header cards?
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
        """Make the default map from header information to amplifier
        information.

        Returns
        -------
        headerAmMap : `HeaderAmpMap`
             The HeaderAmpMap object containing the mapping
        """
        hMap = HeaderAmpMap()
        emptyBBox = lsst.geom.BoxI()
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
                   ('XYOFF', 'setRawXYOffset', lsst.geom.ExtentI(0, 0), self._makeExt),
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
        """Make the default map from header information to detector information.

        Returns
        -------
        headerDetectorMap : `HeaderDetectorMap`
             The HeaderDetectorMap object containing the mapping.
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

        Parameters
        ----------
        extArr : `array` of `int`
            Length 2 array to use in creating the Extent object.

        Returns
        -------
        extent : `lsst.geom.Extent2I`
             Extent constructed from the input list.
        """
        return lsst.geom.ExtentI(*extArr)

    def _makeBbox(self, boxString):
        """Helper funtion to make a bounding box from a string representing a
        FITS style bounding box.

        Parameters
        ----------
        boxString : `str`
            String describing the bounding box.

        Returns
        -------
        bbox : `lsst.geom.Box2I`
            The bounding box.
        """
        # strip off brackets and split into parts
        x1, x2, y1, y2 = [int(el) for el in re.split(
            '[:,]', boxString.strip()[1:-1])]
        box = lsst.geom.BoxI(lsst.geom.PointI(x1, y1), lsst.geom.PointI(x2, y2))
        # account for the difference between FITS convention and LSST convention for
        # index of LLC.
        box.shift(lsst.geom.Extent2I(-1, -1))
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
        """Helper function to get the radial transform given the radial
        polynomial coefficients given in the constructor.

        Parameters
        ----------
        radialCoeffs : `iterable` of `float`
            List of coefficients describing a polynomial radial distortion in
            normalized units. The first value must be 0.

        Returns
        -------
        transform : `lsst.afw.geom.TransformPoint2ToPoint2`
            Transform object describing the radial distortion
        """
        pScaleRad = lsst.geom.arcsecToRad(self.plateScale)
        return lsst.afw.geom.makeRadialTransform([el/pScaleRad for el in radialCoeffs])

    def buildDetector(self):
        """Take all the information and build a Detector object.
        The Detector object is necessary for doing things like assembly.

        Returns
        -------
        detector : `lsst.afw.cameraGeom.Detector`
             The detector.
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

    def makeExposure(self, im, mask=None, variance=None):
        """Method for constructing an exposure object from an image and the
        information contained in this class to construct the Detector.

        Parameters
        ----------
        im : `lsst.afw.image.Image`
            Image used to construct the exposure.
        mask : `lsst.afw.image.MaskU`
            Optional mask plane.
        variance : `lsst.afw.image.Image`
            Optional variance plance as an image of the same type as im.

        Returns
        -------
        exposure : `lsst.afw.image.Exposure`
            Constructed exposure (specific type will match that of ``im``).
        """
        if mask is None:
            mask = afwImage.Mask(im.getDimensions())
        if variance is None:
            variance = im
        mi = afwImage.makeMaskedImage(im, mask, variance)
        detector = self.buildDetector()

        exp = afwImage.makeExposure(mi)
        exp.setDetector(detector)
        return exp
