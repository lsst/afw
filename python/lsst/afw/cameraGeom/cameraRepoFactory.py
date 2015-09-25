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
from lsst.afw.cameraGeom import (DetectorConfig, CameraConfig, PUPIL, FOCAL_PLANE, PIXELS,
                                 SCIENCE, FOCUS, GUIDER, WAVEFRONT)

class CameraRepositoryFactory(object):

    def __init__(self, detectorLayoutFile, segmentationFile,
                 readCorner='LL',
                 gainFile=None,
                 expandDetectorName=None,
                 detectorIdFromAbbrevName=None,
                 detTypeMap=None):

        self._default_saturation = 65535
        self._readCorner=readCorner
        self._detectorLayoutFile = detectorLayoutFile
        self._segmentationFile = degmentationFile
        self._gainFile = gainFile
        self._shortNameFromLongName = {}

        self._prescan = 1
        self._hoverscan = 0
        self._extended = 4
        self._voverscan = 0

        if expandDetectorName is None:
            self._expandDetectorName = self._default_expandDetectorName
        else:
            self._expandDetectorName = expandDetectorName

        if detectorIdFromAbbrevName is None:
            self._detectorIdFromAbbrevName = self._default_detectorIdFromAbbrevName
        else:
            self._detectorIdFromAbbrevName = detectorIdFromAbbrevName

        self._detTypeMap = detTypeMap


    def _default_expandDetectorName(self, name):
        return name


    def _default_detectorIdFromAbbrevName(self, name):
        raise RuntimeError('You cannot run cameraRepofactory without specifying detectorIdFromAbbrevName')

    def expandDetectorName(self, shortName):
        longName = self._expandDetectorName(shortName)

        if shortName not in self._shortNameFromLongName:
            self._shortNameFromLongName[shortName] = longName


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
        if self._gainFile not None:
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
        with open(self._segmentsFile) as fh:
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

                rawBBox = afwGeom.Box2I(afwGeom.Point2I(0,0), afwGeom.Extent2I(self._extended+ndatax+self._hoverscan, \
                                        self._prescan+ndatay+self._voverscan))

                rawDataBBox = afwGeom.Box2I(afwGeom.Point2I(self._extended, self._prescan), afwGeom.Extent2I(ndatax, ndatay))

                rawHorizontalOverscanBBox = afwGeom.Box2I(afwGeom.Point2I(0, self._prescan), afwGeom.Extent2I(self._extended, ndatay))

                rawVerticalOverscanBBox = afwGeom.Box2I(afwGeom.Point2I(self._extended, self._prescan+ndatay), \
                                                        afwGeom.Extent2I(ndatax, self._voverscan))

                rawPrescanBBox = afwGeom.Box2I(afwGeom.Point2I(self._extended, 0), afwGeom.Extent2I(ndatax, self._prescan))

                extraRawX = self._extended + self._hoverscan
                extraRawY = self._prescan + self._voverscan
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


    def makeLongName(self, shortName):
        """
        Make the long name from the PhoSim short name
        @param shortName -- string name like R??_S??[_C??] to parse
        """
        parts = shortName.split("_")
        if len(parts) == 2:
            return " ".join(["%s:%s"%(el[0], ",".join(el[1:])) for el in parts])
        elif len(parts) == 3:
            #This must be a wavefront sensor
            wsPartMap = {'S':{'C0':'A', 'C1':'B'},
                         'R':{'C0':'', 'C1':''}}
            return " ".join(["%s:%s"%(el[0], ",".join(el[1:]+wsPartMap[el[0]][parts[-1]])) for el in parts[:-1]])
        else:
            raise ValueError("Could not parse %s: has %i parts"%(shortName, len(parts)))


    def makeDetectorConfigs(self, detectorLayoutFile, phosimVersion):
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
        with open(detectorLayoutFile) as fh:
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
                detConfig.detectorType = self.detTypeMap[els[8]]
                detConfig.serial = els[0]+"_"+phosimVersion

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


    def makeCameraRepo(self):
        """
        Create the configs for building a camera.  This runs on the files distributed with PhoSim.  Currently gain and
        saturation need to be supplied as well.  The file should have three columns: on disk amp id (R22_S11_C00), gain, saturation.
        For example:
        DetectorLayoutFile -- https://dev.lsstcorp.org/cgit/LSST/sims/phosim.git/plain/data/lsst/focalplanelayout.txt?h=dev
        SegmentsFile -- https://dev.lsstcorp.org/cgit/LSST/sims/phosim.git/plain/data/lsst/segmentation.txt?h=dev
        """
        baseDir = lsst.utils.getPackageDir('obs_lsstsim')
        defaultOutDir = os.path.join(os.path.normpath(baseDir), "description", "camera")

        parser = argparse.ArgumentParser()
        parser.add_argument("DetectorLayoutFile", help="Path to detector layout file")
        parser.add_argument("SegmentsFile", help="Path to amp segments file")
        parser.add_argument("GainFile", help="Path to gain and saturation file")
        parser.add_argument("phosimVersion", help="String id of the version of phosim used to construct this camera repository")
        parser.add_argument("OutputDir",
            help = "Path to dump configs and AmpInfo Tables; defaults to %r" % (defaultOutDir,),
            nargs = "?",
            default = defaultOutDir,
        )
        parser.add_argument("--clobber", action="store_true", dest="clobber", default=False,
            help=("remove and re-create the output directory if it already exists?"))
        args = parser.parse_args()
        ampTableDict = self.makeAmpTables(args.SegmentsFile, args.GainFile)
        detectorConfigList = self.makeDetectorConfigs(args.DetectorLayoutFile, args.phosimVersion)

        #Build the camera config.
        camConfig = CameraConfig()
        camConfig.detectorList = dict([(i,detectorConfigList[i]) for i in xrange(len(detectorConfigList))])
        camConfig.name = 'LSST'
        camConfig.plateScale = 20.0
        pScaleRad = afwGeom.arcsecToRad(camConfig.plateScale)
        pincushion = 0.925
        # Don't have this yet ticket/3155
        #camConfig.boresiteOffset_x = 0.
        #camConfig.boresiteOffset_y = 0.
        tConfig = afwGeom.TransformConfig()
        tConfig.transform.name = 'inverted'
        radialClass = afwGeom.xyTransformRegistry['radial']
        tConfig.transform.active.transform.retarget(radialClass)
        # According to Dave M. the simulated LSST transform is well approximated (1/3 pix)
        # by a scale and a pincusion.
        tConfig.transform.active.transform.coeffs = [0., 1./pScaleRad, 0., pincushion/pScaleRad]
        #tConfig.transform.active.boresiteOffset_x = camConfig.boresiteOffset_x
        #tConfig.transform.active.boresiteOffset_y = camConfig.boresiteOffset_y
        tmc = afwGeom.TransformMapConfig()
        tmc.nativeSys = FOCAL_PLANE.getSysName()
        tmc.transforms = {PUPIL.getSysName():tConfig}
        camConfig.transformDict = tmc

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
        outDir = args.OutputDir
        makeDir(dirPath=outDir, doClobber=args.clobber)

        camConfigPath = os.path.join(outDir, "camera.py")
        camConfig.save(camConfigPath)

        for detectorName, ampTable in ampTableDict.iteritems():
            shortDetectorName = self._shortNameFromLongName[detectorName]
            ampInfoPath = os.path.join(outDir, shortDetectorName + ".fits")
            ampTable.writeFits(ampInfoPath)
