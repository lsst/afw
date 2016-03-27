from __future__ import absolute_import, division
#
# LSST Data Management System
# See the COPYRIGHT and LICENSE files in the top-level directory of this
# package for notices and licensing terms.
#
"""Utilities that should be imported into the lsst.afw.coord namespace when lsst.afw.coord is used

In the case of the assert functions, importing them makes them available in lsst.utils.tests.TestCase
"""
import lsst.utils.tests
import lsst.afw.geom as afwGeom

__all__ = ["assertCoordsNearlyEqual"]

@lsst.utils.tests.inTestCase
def assertCoordsNearlyEqual(testCase, coord0, coord1, maxDiff=0.001*afwGeom.arcseconds, msg="Coords differ"):
    """!Assert that two coords represent nearly the same point on the sky

    @warning the coordinate systems are not compared; instead both angles are converted to ICRS
    and the angular separation measured.

    @param[in] testCase  unittest.TestCase instance the test is part of;
                        an object supporting one method: fail(self, msgStr)
    @param[in] coord0  coord 0 (an lsst.afw.geom.Coord)
    @param[in] coord1  coord 1 (an lsst.afw.geom.Coord)
    @param[in] maxDiff  maximum angular separation between the two coords (an lsst.afw.geom.Angle)
    @param[in] msg  exception message prefix; details of the error are appended after ": "

    @throw AssertionError if the unwrapped difference is greater than maxDiff
    """
    measDiff = coord0.toIcrs().angularSeparation(coord1.toIcrs())
    if measDiff > maxDiff:
        testCase.fail("%s: measured angular separation %s arcsec > max allowed %s arcsec" %
            (msg, measDiff.asArcseconds(), maxDiff.asArcseconds()))
