#!/usr/bin/env python
#
# LSST Data Management System
# Copyright 2008-2014 LSST Corporation.
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

import unittest
import lsst.utils.tests

import os
import numpy as np
from lsst.utils import getPackageDir
import lsst.afw.geom as afwGeom
from lsst.afw.cameraGeom import BasicCameraFactory
from lsst.afw.cameraGeom import SCIENCE, FOCUS, GUIDER, WAVEFRONT
from lsst.afw.cameraGeom import PUPIL, FOCAL_PLANE, PIXELS

class BasicCameraFactoryTest(unittest.TestCase):

    def setUp(self):
        self. cameraDataDir = os.path.join(getPackageDir('afw'),
                                           'tests', 'data')

    def testBBox(self):
        """
        Test that BasicCameraFactory creates detectors with the correct BBoxes
        """
        layoutFile = os.path.join(self.cameraDataDir, 'testFocalPlaneLayout_0.txt')

        def getIdFromName(name):
            return int(name[-2:])

        factory = BasicCameraFactory(detectorLayoutFile=layoutFile,
                                     detectorIdFromAbbrevName=getIdFromName,
                                     detTypeMap = {'science':SCIENCE,
                                                   'focus':FOCUS,
                                                   'guider':GUIDER,
                                                   'wave':WAVEFRONT})


        camera = factory.makeCamera()

        bboxCorners = camera['Det00'].getBBox().getCorners()
        self.assertEqual(bboxCorners[0][0], 0)
        self.assertEqual(bboxCorners[0][1], 0)
        self.assertEqual(bboxCorners[1][0], 399)
        self.assertEqual(bboxCorners[1][1], 0)
        self.assertEqual(bboxCorners[2][0], 399)
        self.assertEqual(bboxCorners[2][1], 399)
        self.assertEqual(bboxCorners[3][0], 0)
        self.assertEqual(bboxCorners[3][1], 399)

        bboxCorners = camera['Det01'].getBBox().getCorners()
        self.assertEqual(bboxCorners[0][0], 0)
        self.assertEqual(bboxCorners[0][1], 0)
        self.assertEqual(bboxCorners[1][0], 199)
        self.assertEqual(bboxCorners[1][1], 0)
        self.assertEqual(bboxCorners[2][0], 199)
        self.assertEqual(bboxCorners[2][1], 299)
        self.assertEqual(bboxCorners[3][0], 0)
        self.assertEqual(bboxCorners[3][1], 299)

        bboxCorners = camera['Det02'].getBBox().getCorners()
        self.assertEqual(bboxCorners[0][0], 0)
        self.assertEqual(bboxCorners[0][1], 0)
        self.assertEqual(bboxCorners[1][0], 299)
        self.assertEqual(bboxCorners[1][1], 0)
        self.assertEqual(bboxCorners[2][0], 299)
        self.assertEqual(bboxCorners[2][1], 299)
        self.assertEqual(bboxCorners[3][0], 0)
        self.assertEqual(bboxCorners[3][1], 299)


    def testFocalPlaneCoords(self):
        """
        Test that, when we generate a camera using BasicCameraFactory,
        it maps focal plane coordinates to pixel coordinates in the way we
        expect
        """
        layoutFile = os.path.join(self.cameraDataDir, 'testFocalPlaneLayout_0.txt')

        def getIdFromName(name):
            return int(name[-2:])

        factory = BasicCameraFactory(detectorLayoutFile=layoutFile,
                                     detectorIdFromAbbrevName=getIdFromName,
                                     detTypeMap = {'science':SCIENCE,
                                                   'focus':FOCUS,
                                                   'guider':GUIDER,
                                                   'wave':WAVEFRONT})


        camera = factory.makeCamera()

        detNameList = ['Det00', 'Det01', 'Det02']
        yawList = np.radians([20.0, 10.0, 30.0])
        xCenterList = [0.0, 0.0, -9.0]
        yCenterList = [0.0, 9.0, 0.0]
        nxList = [400, 300, 300]
        nyList = [400, 200, 300]
        mmPerPixelList = [2.0e-3, 1.0e-3, 3.0e-3]

        for detName, yaw, xCenter, yCenter, nx, ny, mmPerPixel in \
        zip(detNameList, yawList, xCenterList, yCenterList, \
            nxList, nyList, mmPerPixelList):

            pixelSystem = camera[detName].makeCameraSys(PIXELS)
            focalSystem = camera[detName].makeCameraSys(FOCAL_PLANE)
            xPix_control = []
            yPix_control = []
            xFocalList = []
            yFocalList = []
            for xx in range(0, 400, 100):
                for yy in range(0, 400, 100):
                    xPix_control.append(xx)
                    yPix_control.append(yy)

                    # Note: because of the demand that the x-axis in pixel coordinates
                    # be along the direction of readout, the pixel x and y axes are rotated
                    # 90 degrees with respect to the focal plane x and y axes
                    xxFocalUnRotated = (0.5*(nx-1)-yy)*mmPerPixel
                    yyFocalUnRotated = (xx-0.5*(ny-1))*mmPerPixel

                    xxFocal = xCenter + xxFocalUnRotated*np.cos(yaw) - yyFocalUnRotated*np.sin(yaw)
                    yyFocal = yCenter + xxFocalUnRotated*np.sin(yaw) + yyFocalUnRotated*np.cos(yaw)

                    xFocalList.append(xxFocal)
                    yFocalList.append(yyFocal)

            for xxF, yyF, xxPcontrol, yyPcontrol in \
            zip(xFocalList, yFocalList, xPix_control, yPix_control):

                focalPoint = camera.makeCameraPoint(afwGeom.Point2D(xxF, yyF), FOCAL_PLANE)
                pixelPoint = camera.transform(focalPoint, pixelSystem).getPoint()

                self.assertAlmostEqual(float(xxPcontrol), pixelPoint.getX(), 9)
                self.assertAlmostEqual(float(yyPcontrol), pixelPoint.getY(), 9)


    def testPupilCoords(self):
        """
        Test that cameras generated with BasicCameraFactory convert from pupil
        coordinates to pixel coordinates as expected
        """
        layoutFile = os.path.join(self.cameraDataDir, 'testFocalPlaneLayout_0.txt')

        def getIdFromName(name):
            return int(name[-2:])

        np.random.seed(435)

        detNameList = ['Det00', 'Det01', 'Det02']
        yawList = np.radians([20.0, 10.0, 30.0])
        xCenterList = [0.0, 0.0, -9.0]
        yCenterList = [0.0, 9.0, 0.0]
        nxList = [400, 300, 300]
        nyList = [400, 200, 300]
        mmPerPixelList = [2.0e-3, 1.0e-3, 3.0e-3]

        for iteration in range(3):

            # try random radial transforms between pupil and pixel coordinates
            radial_coeffs = np.random.random_sample(5)
            radial_coeffs[0] = 0.0

            factory = BasicCameraFactory(detectorLayoutFile=layoutFile,
                                         detectorIdFromAbbrevName=getIdFromName,
                                         radialTransform = radial_coeffs,
                                         detTypeMap = {'science':SCIENCE,
                                                       'focus':FOCUS,
                                                       'guider':GUIDER,
                                                       'wave':WAVEFRONT})


            camera = factory.makeCamera()

            for detName, yaw, xCenter, yCenter, nx, ny, mmPerPixel in \
            zip(detNameList, yawList, xCenterList, yCenterList, \
                nxList, nyList, mmPerPixelList):

                pupilSystem = camera[detName].makeCameraSys(PUPIL)
                pixelSystem = camera[detName].makeCameraSys(PIXELS)
                focalSystem = camera[detName].makeCameraSys(FOCAL_PLANE)

                # approximate radians per pixel
                scale_approx = mmPerPixel/radial_coeffs[1]

                for theta in np.arange(0.0, 2.0*np.pi, 0.5*np.pi):
                    for rr in np.arange(10.0*scale_approx, 200.0*scale_approx, 50.0*scale_approx):

                        xp = rr*np.cos(theta)
                        yp = rr*np.sin(theta)

                        rr_scale = np.array([radial_coeffs[ii]*rr**ii \
                                             for ii in range(len(radial_coeffs))]).sum()/rr

                        xf = rr_scale*rr*np.cos(theta)
                        yf = rr_scale*rr*np.sin(theta)

                        pupilPoint = camera.makeCameraPoint(afwGeom.Point2D(xp, yp), PUPIL)
                        focalPoint = camera.transform(pupilPoint, focalSystem).getPoint()

                        # Below we verify that the transformation from
                        # pupil coordinates to focal plane coordinates
                        # behaves the way we expect it to
                        #
                        # Multiplying by 10^10 achives an
                        # approximate conversion from radians to 10^-4 arcsec
                        dx = np.abs(xf-focalPoint.getX())*1.0e10
                        self.assertAlmostEqual(dx, 0.0, 5)

                        dy = np.abs(yf - focalPoint.getY())*1.0e10
                        self.assertAlmostEqual(dy, 0.0, 5)

                        # re-center on chip's origin
                        xfCenter = xf - xCenter
                        yfCenter = yf - yCenter

                        # rotate by -yaw to orient to the x, y axes of the chip
                        xfChip = xfCenter*np.cos(yaw) + yfCenter*np.sin(yaw)
                        yfChip = -xfCenter*np.sin(yaw) + yfCenter*np.cos(yaw)

                        yPix = 0.5*(nx-1) - xfChip/mmPerPixel
                        xPix = 0.5*(ny-1) + yfChip/mmPerPixel

                        pixelPoint = camera.transform(pupilPoint, pixelSystem).getPoint()

                        self.assertAlmostEqual(xPix, pixelPoint.getX(), 10)
                        self.assertAlmostEqual(yPix, pixelPoint.getY(), 10)



def suite():
    """Returns a suite containing all the test cases in this module."""

    lsst.utils.tests.init()

    suites = []
    suites += unittest.makeSuite(BasicCameraFactoryTest)
    return unittest.TestSuite(suites)

def run(shouldExit = False):
    """Run the tests"""
    lsst.utils.tests.run(suite(), shouldExit)

if __name__ == "__main__":
    run(True)
