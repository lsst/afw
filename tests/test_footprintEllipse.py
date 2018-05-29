#
# LSST Data Management System
# Copyright 2008, 2009, 2010 LSST Corporation.
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

import math
import unittest

import lsst.utils.tests
import lsst.afw.geom as afwGeom
import lsst.afw.detection as afwDet


class FootprintTestCase(lsst.utils.tests.TestCase):

    def testCircle(self):
        xc, yc = 30, 50
        radius = 10

        spanSet = afwGeom.SpanSet.fromShape(radius).shiftedBy(xc, yc)
        test = afwDet.Footprint(spanSet)

        # Here's how it used to be done using circles, before #1556
        r2 = int(radius**2 + 0.5)
        r = int(math.sqrt(r2))
        spanList = []
        for i in range(-r, r+1):
            hlen = int(math.sqrt(r2 - i**2))
            spanList.append(afwGeom.Span(yc + i, xc - hlen, xc + hlen))
        control = afwDet.Footprint(afwGeom.SpanSet(spanList))

        self.assertEqual(len(test.getSpans()), len(control.getSpans()))
        for s0, s1 in zip(test.getSpans(), control.getSpans()):
            self.assertEqual(s0.getX0(), s1.getX0())
            self.assertEqual(s0.getX1(), s1.getX1())
            self.assertEqual(s0.getY(), s1.getY())
        self.assertEqual(test.getArea(), control.getArea())


class MemoryTester(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
