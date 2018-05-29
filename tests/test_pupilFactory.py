#
# LSST Data Management System
# Copyright 2017 LSST Corporation.
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
import numpy as np

import lsst.utils.tests
import lsst.afw.image as afwImage
import lsst.afw.cameraGeom as afwCameraGeom
from lsst.afw.geom import degrees


class PupilFactoryTestCase(lsst.utils.tests.TestCase):
    """Test lsst.afw.cameraGeom.PupilFactory"""

    def setUp(self):
        self.visitInfo = afwImage.VisitInfo()
        self.size = 16.8
        self.npix = 1024
        self.scale = self.size / self.npix

    def testBasePupilFactory(self):
        pupilFactory = afwCameraGeom.PupilFactory(
            self.visitInfo, self.size, self.npix)
        self.assertEqual(pupilFactory.pupilSize, self.size)
        self.assertEqual(pupilFactory.pupilScale, self.scale)
        self.assertEqual(pupilFactory.npix, self.npix)
        with self.assertRaises(NotImplementedError):
            pupilFactory.getPupil(point=None)

    def testBasePupilFactoryMethods(self):
        pupilFactory = afwCameraGeom.PupilFactory(
            self.visitInfo, self.size, self.npix)
        pupil = pupilFactory._fullPupil()
        self.assertTrue(np.all(pupil.illuminated))
        nFull = np.sum(pupil.illuminated)

        # Cut out a primary aperture
        pupilFactory._cutCircleExterior(pupil, (0.0, 0.0), 8.4/2)
        nCircle = np.sum(pupil.illuminated)
        self.assertFloatsAlmostEqual(
            nCircle/nFull, np.pi*(8.4/2)**2 / 16.8**2, rtol=3e-4)
        np.testing.assert_array_equal(pupil.illuminated, pupil.illuminated.T)

        # Cut out a central obstruction making an annulus
        pupilFactory._cutCircleInterior(pupil, (0.0, 0.0), 8.4/2 * 0.6)
        nAnnulus = np.sum(pupil.illuminated)
        self.assertFloatsAlmostEqual(
            nAnnulus/nFull, nCircle/nFull * (1-0.6**2), rtol=3e-4)
        np.testing.assert_array_equal(pupil.illuminated, pupil.illuminated.T)

        # Cut a horizontal ray, which preserves vertical reflection symmetry
        # but removes horizontal reflection symmetry, and transpositional
        # symmetry.
        pupilFactory._cutRay(pupil, (0.0, 0.0), 0*degrees, 0.1)
        np.testing.assert_array_equal(pupil.illuminated, pupil.illuminated[::-1, :])
        self.assertTrue(np.any(pupil.illuminated !=
                        pupil.illuminated[:, ::-1]))
        self.assertTrue(np.any(pupil.illuminated != pupil.illuminated.T))

        # Cut a vertical ray, which then gives transpositional symmetry but
        # removes vertical and horizontal reflection symmetry
        pupilFactory._cutRay(pupil, (0.0, 0.0), 90*degrees, 0.1)
        self.assertTrue(np.any(pupil.illuminated !=
                        pupil.illuminated[::-1, :]))
        self.assertTrue(np.any(pupil.illuminated !=
                        pupil.illuminated[:, ::-1]))
        self.assertTrue(np.any(pupil.illuminated == pupil.illuminated.T))


def setup_module(module):
    lsst.utils.tests.init()


class MemoryTester(lsst.utils.tests.MemoryTestCase):
    pass


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
