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

import unittest

import lsst.afw.table as afwTable
import lsst.utils.tests


class SchemaOffsetTest(unittest.TestCase):

    def setUp(self):
        self.schema = afwTable.Schema()
        self.schema.addField('test1', type='ArrayF', size=430000025)
        self.schema.addField('test2', type='ArrayF', size=430000025)
        self.cat = afwTable.BaseCatalog(self.schema)

    def testPreAllocate(self):
        self.cat.table.preallocate(1)

    def tearDown(self):
        del self.schema
        del self.cat


class MemoryTester(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
