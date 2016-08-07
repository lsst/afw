from __future__ import absolute_import, division
#
# LSST Data Management System
# Copyright 2015 LSST Corporation.
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
