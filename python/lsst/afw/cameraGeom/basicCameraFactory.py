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

    def __init__(self, detectorLayoutFile,
                 expandDetectorName=None,
                 detectorIdFromAbbrevName=None,
                 detTypeMap=None,
                 radialTransform=[0.0, 1.0, 0.0, 0.0],
                 cameraName='LSST'
                 ):

        """
        @param [in] detectorLayoutFile is the absolute path to the file
        listing the layout of all of the chips

        @param [in] readCorner is the corner from which each detector is
        read.  'LL' for lower left (default).  'UL' for upper left.
        'LR' for lower right.  'UR' for upper right.

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
        """

        self._detectorLayoutFile = detectorLayoutFile
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
                detConfig.detectorType = self._detTypeMap[els[6]]
                detConfig.serial = els[0]

                # Convert from microns to mm.
                detConfig.offset_x = float(els[1])/1000.
                detConfig.offset_y = float(els[2])/1000.

                detConfig.refpos_x = (int(els[5]) - 1.)/2.
                detConfig.refpos_y = (int(els[4]) - 1.)/2.
                # TODO translate between John's angles and Orientation angles.
                # It's not an issue now because there is no rotation except about z in John's model.
                detConfig.yawDeg = 90.*nQuarter + float(els[7])
                detConfig.pitchDeg = float(els[8])
                detConfig.rollDeg = float(els[9])
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

        detectorConfigList = self.makeDetectorConfigs()
        schema = afwTable.AmpInfoTable.makeMinimalSchema()
        ampTableDict = {}
        for detConfig in detectorConfigList:
            ampCatalog = afwTable.AmpInfoCatalog(schema)
            ampTableDict[detConfig.name] = ampCatalog

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
