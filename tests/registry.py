#!/usr/bin/env python

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
import os
import unittest
import lsst.utils.tests as utilsTests
import lsst.pex.config as pexConfig
import lsst.afw.registry as afwReg

class ConfigTest(unittest.TestCase):
    def setUp(self):
        """Note: the classes are defined here in order to test the register decorator
        """
        class ParentConfig(pexConfig.Config):
            pass
        
        self.registry = afwReg.makeRegistry(doc="unit test configs", configBaseType=ParentConfig)
        
        class FooConfig1(ParentConfig):
            pass
        self.fooConfig1Class = FooConfig1
        
        class FooConfig2(ParentConfig):
            pass
        self.fooConfig2Class = FooConfig2

        class Config1(pexConfig.Config):
            pass
        self.config1Class = Config1
        
        class Config2(pexConfig.Config):
            pass
        self.config2Class = Config2
        
        @afwReg.registerFactory("foo1", self.registry)
        class FooAlg1(object):
            ConfigClass = FooConfig1
            def __init__(self, config):
                self.config = config
            def foo(self):
                pass
        self.fooAlg1Class = FooAlg1
        
        class FooAlg2(object):
            ConfigClass = FooConfig2
            def __init__(self, config):
                self.config = config
            def foo(self):
                pass
        self.registry.register("foo2", FooAlg2, FooConfig2)
        self.fooAlg2Class = FooAlg2
        
        # override Foo2 with FooConfig1
        self.registry.register("foo21", FooAlg2, FooConfig1)
    
    def tearDown(self):
        del self.registry
        del self.fooConfig1Class
        del self.fooConfig2Class
        del self.fooAlg1Class
        del self.fooAlg2Class
        
    def testBasics(self):
        self.assertEqual(self.registry["foo1"], self.fooAlg1Class)
        self.assertEqual(self.registry["foo2"].ConfigClass, self.fooConfig2Class)
        self.assertEqual(self.registry["foo21"].ConfigClass, self.fooConfig1Class)
        
        self.assertEqual(set(self.registry.keys()), set(("foo1", "foo2", "foo21")))

    def testWrapper(self):
        wrapper21 = self.registry["foo21"]
        foo21 = wrapper21(wrapper21.ConfigClass())
        self.assertTrue(isinstance(foo21, self.fooAlg2Class))
        
    def testReplace(self):
        """Test replacement in registry (should always fail)
        """
        self.assertRaises(Exception, self.registry.register, "foo1", self.fooAlg2Class)
        self.assertEqual(self.registry["foo1"], self.fooAlg1Class)

def  suite():
    utilsTests.init()
    suites = []
    suites += unittest.makeSuite(ConfigTest)
    suites += unittest.makeSuite(utilsTests.MemoryTestCase)
    return unittest.TestSuite(suites)

def run(exit=False):
    utilsTests.run(suite(), exit)

if __name__=='__main__':
    run(True)
