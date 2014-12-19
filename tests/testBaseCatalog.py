#!/usr/bin/env python2
from __future__ import absolute_import, division
#
# LSST Data Management System
# Copyright 2008-2014 AURA/LSST.
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

"""
Tests for BaseCatalog
"""

import os
import unittest

import lsst.afw.table
import lsst.utils.tests

class BaseCatalogTestCase(lsst.utils.tests.TestCase):
    def testDM1710(self):
        schema = lsst.afw.table.Schema()
        cat1 = lsst.afw.table.BaseCatalog(schema)
        cat2 = lsst.afw.table.BaseCatalog(schema)

        # Should not raise.
        cat1.extend(cat2)

def suite():
    """Returns a suite containing all the test cases in this module."""

    lsst.utils.tests.init()

    suites = []
    suites += unittest.makeSuite(BaseCatalogTestCase)
    return unittest.TestSuite(suites)

def run(shouldExit = False):
    """Run the tests"""
    lsst.utils.tests.run(suite(), shouldExit)

if __name__ == "__main__":
    run(True)
