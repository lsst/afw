#!/usr/bin/env python2
#
# LSST Data Management System
# Copyright 2014 LSST Corporation.
#
# This product includes software developed by the
# LSST Project (http://www.lsst.org/).
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
# You should have received a copy of the LSST License Statement and
# the GNU General Public License along with this program.  If not,
# see <http://www.lsstcorp.org/LegalNotices/>.
#
from __future__ import absolute_import, division
import argparse
import os
import re
import shutil

import lsst.utils
import lsst.afw.geom as afwGeom
import lsst.afw.table as afwTable
from lsst.afw.cameraGeom import makeCameraFromCatalogs
from lsst.afw.cameraGeom import (DetectorConfig, CameraConfig, PUPIL, FOCAL_PLANE, PIXELS,
                                 SCIENCE, FOCUS, GUIDER, WAVEFRONT)


__all__ = ["BasicCameraFactory"]


class BasicCameraFactory(object):

    def __init__(self, detectorLayoutFile, segmentationFile,
                 readCorner='LL',
                 gainFile=None,
                 expandDetectorName=None,
                 detectorIdFromAbbrevName=None,
                 detTypeMap=None,
                 radialTransform=[0.0, 1.0, 0.0, 0.0],
                 cameraName='LSST',
                 saturation=65535,
                 version=None
                 ):

        """
        @param [in] detectorLayoutFile is the absolute path to the file
        listing the layout of all of the chips

        @param [in] segmentationFile is the absolute path to the file
        describing the details of the amplifiers

        @param [in] readCorner is the corner from which each detector is
        read.  'LL' for lower left (default).  'UL' for upper left.
        'LR' for lower right.  'UR' for upper right.

        @param [in] gainFile is an (optional) file providing gain and saturation
        information.

        @param [in] expandDetectorName is an (optional) method that takes the name
        of detectors as recorded in the detectorLayoutFile and expands them into
        their names as they will be stored in the returned afw.camerGeom.camera
        object.  If 'None', the names used in detectorLayoutFile will be stored
        in the afw.cameraGeom.camera object.

        @param [in] detectorIdFromAbbrevName is a (required) method mapping the
        names of detectors as stored in the detectorLayoutFile to unique integers
        identifying each detector.

        @param[in] detTypeMap is a dict mapping the group name of the sensors in
        detectorLayoutFile to the integers:

        0 = SCIENCE
        1 = FOCUS
        2 = GUIDER
        3 = WAVEFRONT

        if left as None, this will just try to cast the contents of detectoLayoutFile
        as an int

        @param [in] radialTransform is a list of coefficents that transform positions
        from pupil coordinates (x, y, radians) to focal plane coordinates (x, y, mm)
        according to the convention

        sum_i radialTransform[i] * r^i/r

        where r is the magnitude of the point in pupil coordinates.  Note that the [1]
        element of this list should be 1/plateScale in radians per mm.

        @param [in] cameraName is a string referring to the name of the camera
        (default 'LSST')

        @param [in] saturation is the default number of counts at which a pixel
        is saturated (default 65535)

        @param [in] version is a string denoting the version of the camera
        """

        self._default_saturation = saturation
        self._readCorner=readCorner
        self._detectorLayoutFile = detectorLayoutFile
        self._segmentationFile = segmentationFile
        self._gainFile = gainFile
        self._version = version
        self._shortNameFromLongName = {}

        self._cameraName = cameraName
        self._radialTransform = radialTransform #[1] should be 1/rad per mm

        if expandDetectorName is None:
            self._expandDetectorName = self._default_expandDetectorName
        else:
            self._expandDetectorName = expandDetectorName

        if detectorIdFromAbbrevName is None:
            self._detectorIdFromAbbrevName = self._default_detectorIdFromAbbrevName
        else:
            self._detectorIdFromAbbrevName = detectorIdFromAbbrevName

        self._detTypeMap = detTypeMap

        self._camConfig = None
        self._ampTableDict = None


    def _default_expandDetectorName(self, name):
        return name


    def _default_detectorIdFromAbbrevName(self, name):
        raise RuntimeError('You cannot run cameraRepofactory without specifying detectorIdFromAbbrevName')

    def expandDetectorName(self, shortName):
        longName = self._expandDetectorName(shortName)

        if shortName not in self._shortNameFromLongName:
            self._shortNameFromLongName[shortName] = longName

        return longName


    def detTypeMap(self, typeName):
        if self._detTypeMap is None:
            return int(typeName)
        else:
            return self._detTypeMap[typeName]


    def makeAmpTables(self):
        """
        Read the segments file from a PhoSim release and produce the appropriate AmpInfo
        @param segmentsFile -- String indicating where the file is located
        """
        gainDict = {}
        if self._gainFile is not None:
            with open(gainFile) as fh:
                for l in fh:
                    els = l.rstrip().split()
                    gainDict[els[0]] = {'gain':float(els[1]), 'saturation':int(els[2])}
        returnDict = {}
        #TODO currently there is no linearity provided, but we should identify
        #how to get this information.
        linearityCoeffs = (0.,1.,0.,0.)
        linearityType = "Polynomial"
        readoutMap = {'LL':afwTable.LL, 'LR':afwTable.LR, 'UR':afwTable.UR, 'UL':afwTable.UL}
        ampCatalog = None
        detectorName = [] # set to a value that is an invalid dict key, to catch bugs
        correctY0 = False
        with open(self._segmentationFile) as fh:
            for l in fh:
                if l.startswith("#"):
                    continue

                els = l.rstrip().split()
                if len(els) == 4:
                    if ampCatalog is not None:
                        returnDict[detectorName] = ampCatalog
                    detectorName = self.expandDetectorName(els[0])
                    numy = int(els[2])
                    schema = afwTable.AmpInfoTable.makeMinimalSchema()
                    ampCatalog = afwTable.AmpInfoCatalog(schema)
                    if len(els[0].split('_')) == 3:   #wavefront sensor
                        correctY0 = True
                    else:
                        correctY0 = False
                    continue
                record = ampCatalog.addNew()
                name = els[0].split("_")[-1]
                name = '%s,%s'%(name[1], name[2])
                #Because of the camera coordinate system, we choose an
                #image coordinate system that requires a -90 rotation to get
                #the correct pixel positions from the
                #phosim segments file
                y0 = numy - 1 - int(els[2])
                y1 = numy - 1 - int(els[1])
                #Another quirk of the phosim file is that one of the wavefront sensor
                #chips has an offset of 2000 pix in y.  It's always the 'C1' chip.
                if correctY0:
                    if y0 > 0:
                        y1 -= y0
                        y0 = 0
                x0 = int(els[3])
                x1 = int(els[4])
                try:
                    saturation = gainDict[els[0]]['saturation']
                    gain = gainDict[els[0]]['gain']
                except KeyError:
                    # Set default if no gain exists
                    saturation = self._default_saturation
                    gain = float(els[7])
                readnoise = float(els[11])
                bbox = afwGeom.Box2I(afwGeom.Point2I(x0, y0), afwGeom.Point2I(x1, y1))

                if int(els[5]) == -1:
                    flipx = False
                else:
                    flipx = True
                if int(els[6]) == 1:
                    flipy = False
                else:
                    flipy = True

                #Since the amps are stored in amp coordinates, the readout is the same
                #for all amps
                readCorner = readoutMap[self._readCorner]

                ndatax = x1 - x0 + 1
                ndatay = y1 - y0 + 1
                #Because in versions v3.3.2 and earlier there was no overscan, we use the extended register as the overscan region

                parallel_prescan = int(els[15])
                serial_overscan = int(els[16])
                serial_prescan = int(els[17])
                parallel_overscan = int(els[18])

                rawBBox = afwGeom.Box2I(afwGeom.Point2I(0,0), afwGeom.Extent2I(serial_prescan+ndatax+serial_overscan, \
                                        parallel_prescan+ndatay+parallel_overscan))

                rawDataBBox = afwGeom.Box2I(afwGeom.Point2I(serial_prescan, parallel_prescan), afwGeom.Extent2I(ndatax, ndatay))

                rawHorizontalOverscanBBox = afwGeom.Box2I(afwGeom.Point2I(0, parallel_prescan), afwGeom.Extent2I(serial_prescan, ndatay))

                rawVerticalOverscanBBox = afwGeom.Box2I(afwGeom.Point2I(serial_prescan, parallel_prescan+ndatay), \
                                                        afwGeom.Extent2I(ndatax, parallel_overscan))

                rawPrescanBBox = afwGeom.Box2I(afwGeom.Point2I(serial_prescan, 0), afwGeom.Extent2I(ndatax, parallel_prescan))

                extraRawX = serial_prescan + serial_overscan
                extraRawY = parallel_prescan + parallel_overscan
                rawx0 = x0 + extraRawX*(x0//ndatax)
                rawy0 = y0 + extraRawY*(y0//ndatay)
                #Set the elements of the record for this amp
                record.setBBox(bbox)
                record.setName(name)
                record.setReadoutCorner(readCorner)
                record.setGain(gain)
                record.setSaturation(saturation)
                record.setReadNoise(readnoise)
                record.setLinearityCoeffs(linearityCoeffs)
                record.setLinearityType(linearityType)
                record.setHasRawInfo(True)
                record.setRawFlipX(flipx)
                record.setRawFlipY(flipy)
                record.setRawBBox(rawBBox)
                record.setRawXYOffset(afwGeom.Extent2I(rawx0, rawy0))
                record.setRawDataBBox(rawDataBBox)
                record.setRawHorizontalOverscanBBox(rawHorizontalOverscanBBox)
                record.setRawVerticalOverscanBBox(rawVerticalOverscanBBox)
                record.setRawPrescanBBox(rawPrescanBBox)
        returnDict[detectorName] = ampCatalog
        return returnDict


    def makeDetectorConfigs(self):
        """
        Create the detector configs to use in building the Camera
        @param detectorLayoutFile -- String describing where the focalplanelayout.txt file is located.

        @todo:
        * set serial to something other than name (e.g. include git sha)
        * deal with the extra orientation angles (not that they really matter)
        """
        detectorConfigs = []
        #We know we need to rotate 3 times and also apply the yaw perturbation
        nQuarter = 1
        with open(self._detectorLayoutFile) as fh:
            for l in fh:
                if l.startswith("#"):
                    continue
                detConfig = DetectorConfig()
                els = l.rstrip().split()
                detConfig.name = self.expandDetectorName(els[0])
                detConfig.id = self._detectorIdFromAbbrevName(els[0])
                detConfig.bbox_x0 = 0
                detConfig.bbox_y0 = 0
                detConfig.bbox_x1 = int(els[5]) - 1
                detConfig.bbox_y1 = int(els[4]) - 1
                detConfig.detectorType = self._detTypeMap[els[8]]
                if self._version is not None:
                    detConfig.serial = els[0]+"_"+self._version
                else:
                    detConfig.serial = els[0]

                # Convert from microns to mm.
                detConfig.offset_x = float(els[1])/1000. + float(els[12])
                detConfig.offset_y = float(els[2])/1000. + float(els[13])

                detConfig.refpos_x = (int(els[5]) - 1.)/2.
                detConfig.refpos_y = (int(els[4]) - 1.)/2.
                # TODO translate between John's angles and Orientation angles.
                # It's not an issue now because there is no rotation except about z in John's model.
                detConfig.yawDeg = 90.*nQuarter + float(els[9])
                detConfig.pitchDeg = float(els[10])
                detConfig.rollDeg = float(els[11])
                detConfig.pixelSize_x = float(els[3])/1000.
                detConfig.pixelSize_y = float(els[3])/1000.
                detConfig.transposeDetector = False
                detConfig.transformDict.nativeSys = PIXELS.getSysName()
                # The FOCAL_PLANE and TAN_PIXEL transforms are generated by the Camera maker,
                # based on orientaiton and other data.
                # Any additional transforms (such as ACTUAL_PIXELS) should be inserted here.
                detectorConfigs.append(detConfig)
        return detectorConfigs


    def _makeCameraData(self):
        """
        Create the configs for building a camera.  This runs on the files distributed with PhoSim.  Currently gain and
        saturation need to be supplied as well.  The file should have three columns: on disk amp id (R22_S11_C00), gain, saturation.
        For example:
        DetectorLayoutFile -- https://dev.lsstcorp.org/cgit/LSST/sims/phosim.git/plain/data/lsst/focalplanelayout.txt?h=dev
        SegmentsFile -- https://dev.lsstcorp.org/cgit/LSST/sims/phosim.git/plain/data/lsst/segmentation.txt?h=dev
        """

        ampTableDict = self.makeAmpTables()
        detectorConfigList = self.makeDetectorConfigs()

        #Build the camera config.
        camConfig = CameraConfig()
        camConfig.detectorList = dict([(i,detectorConfigList[i]) for i in xrange(len(detectorConfigList))])
        camConfig.name = self._cameraName
        camConfig.plateScale = 1.0/afwGeom.radToArcsec(self._radialTransform[1])
        pScaleRad = afwGeom.arcsecToRad(camConfig.plateScale)
        # Don't have this yet ticket/3155
        #camConfig.boresiteOffset_x = 0.
        #camConfig.boresiteOffset_y = 0.
        tConfig = afwGeom.TransformConfig()
        tConfig.transform.name = 'inverted'
        radialClass = afwGeom.xyTransformRegistry['radial']
        tConfig.transform.active.transform.retarget(radialClass)
        # According to Dave M. the simulated LSST transform is well approximated (1/3 pix)
        # by a scale and a pincusion.
        tConfig.transform.active.transform.coeffs = self._radialTransform
        #tConfig.transform.active.boresiteOffset_x = camConfig.boresiteOffset_x
        #tConfig.transform.active.boresiteOffset_y = camConfig.boresiteOffset_y
        tmc = afwGeom.TransformMapConfig()
        tmc.nativeSys = FOCAL_PLANE.getSysName()
        tmc.transforms = {PUPIL.getSysName():tConfig}
        camConfig.transformDict = tmc

        self._camConfig = camConfig
        self._ampTableDict = ampTableDict


    def makeCamera(self):
        if self._camConfig is None:
            self._makeCameraData()

        outputCamera = makeCameraFromCatalogs(self._camConfig, self._ampTableDict)

        return outputCamera


    def makeCameraRepo(self, outputDir):

        if self._camConfig is NOne:
            self._makeCameraData()

        def makeDir(dirPath, doClobber=False):
            """Make a directory; if it exists then clobber or fail, depending on doClobber

            @param[in] dirPath: path of directory to create
            @param[in] doClobber: what to do if dirPath already exists:
                if True and dirPath is a dir, then delete it and recreate it, else raise an exception
            @throw RuntimeError if dirPath exists and doClobber False
            """
            if os.path.exists(dirPath):
                if doClobber and os.path.isdir(dirPath):
                    print "Clobbering directory %r" % (dirPath,)
                    shutil.rmtree(dirPath)
                else:
                    raise RuntimeError("Directory %r exists" % (dirPath,))
            print "Creating directory %r" % (dirPath,)
            os.makedirs(dirPath)

        # write data products
        makeDir(dirPath=outputDir)

        camConfigPath = os.path.join(outputDir, "camera.py")
        self._camConfig.save(camConfigPath)

        for detectorName, ampTable in self._ampTableDict.iteritems():
            shortDetectorName = self._shortNameFromLongName[detectorName]
            ampInfoPath = os.path.join(outDir, shortDetectorName + ".fits")
            ampTable.writeFits(ampInfoPath)
