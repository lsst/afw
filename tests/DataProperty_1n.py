import unittest
import lsst.fw.Core.fwLib as fwCore

try:
    type(verbose)
except NameError:
    verbose = 0
    fwCore.Trace_setVerbosity("fw.DataProperty", verbose)

class DataPropertyTestCase(unittest.TestCase):
    """A test case for DataProperty"""
    def setUp(self):
        self.root = fwCore.DataProperty("root")

        self.values = {}
        props = []
        n = "name1"; self.values[n] = "value1"
        props += [fwCore.DataProperty(n, self.values[n])]

        n = "name2"; self.values[n] = 2
        props += [fwCore.DataProperty(n, self.values[n])]
        props += [fwCore.DataProperty(n, 2*self.values[n])] # add with different value

        n = "name3"
        if False:                           # this won't work, as I don't
                                            # have boost::any working from python
            class Foo:
                __slots__ = ["gurp", "murp", "durp"]

            self.values[n] = Foo()
        else:
            self.values[n] = "Foo()"
        props += [fwCore.DataProperty(n, self.values[n])]

        for prop in props:
            self.root.addProperty(prop)

    def tearDown(self):
        del self.root
        self.root = None

    def testName1(self):
        """Check "name1", a string valued DataProperty"""
        
        dpPtr = self.root.find("name1")
        assert dpPtr.get() != None
        self.assertEqual(dpPtr.getValueString(), self.values["name1"])

    def testDataPtrType(self):
        """Check that the getValueXXX routines get the right types"""
        dpPtr = self.root.find("name1")
        self.assertRaises(RuntimeError, dpPtr.getValueInt)

        dpPtr = self.root.find("name2")
        self.assertRaises(RuntimeError, dpPtr.getValueString)

    def testName2(self):
        """Check "name2", an int valued DataProperty with two definitions"""
        
        dpPtr = self.root.find("name2")
        assert dpPtr.get() != None
        assert dpPtr.getValueInt() == self.values["name2"]

        dpPtr = self.root.find("name2", False)
        assert dpPtr.get() != None
        assert dpPtr.getValueInt() == 2*self.values["name2"]

        dpPtr = self.root.find("name2", False)
        assert dpPtr.get() == None

        dpPtr = self.root.find("name2")
        assert dpPtr.getValueInt() == self.values["name2"]

    def testName3(self):
        """Check name3, which (should have) a non-{int,string} type"""
        dpPtr = self.root.find("name3")
        assert dpPtr.get() != None

    def testUndefined(self):
        """Check that we can't find a data property that isn't defined"""
        dpPtr = self.root.find("undefined")
        assert dpPtr.get() == None

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

class NestedDataPropertyTestCase(unittest.TestCase):
    """A test case for nested DataProperty"""
    def setUp(self):
        self.root = fwCore.DataProperty("root")

        nested = fwCore.DataProperty("nested")

        nprop1 = fwCore.DataProperty("name1n", "value1")
        nprop2 = fwCore.DataProperty("name2n", 2)

        nested.addProperty(nprop1)
        nested.addProperty(nprop2)

        self.root.addProperty(nested)
    
    def tearDown(self):
        del self.root
        self.root = None

    def testCopyConstructor(self):
        """Check copy constructor"""
    
        rootCopy = fwCore.DataProperty(self.root)

        # Explicitly destroy root
        del self.root; self.root = None
    
        # Check that rootCopy is still OK...
        rootCopy.repr("\t")
        assert rootCopy != None

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
        
class MemoryTestCase(unittest.TestCase):
    """Check for memory leaks"""
    def setUp(self):
        pass

    def testLeaks(self):
        """Check for memory leaks in the preceding tests"""
        if fwCore.Citizen_census(0) != 0:
            if False:
                print fwCore.Citizen_census(0), "Objects leaked:"
                print fwCore.Citizen_census(fwCore.cout)
                
            self.fail("Leaked %d blocks" % fwCore.Citizen_census(0))
            
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def suite():
    """Returns a suite containing all the test cases in this module."""
    # Build a TestSuite containing all the possible test case instances
    # that can be made from the ListTestCase class using its 'test*'
    # functions.
    suites = []
    suites += unittest.makeSuite(DataPropertyTestCase)
    suites += unittest.makeSuite(MemoryTestCase)
    return unittest.TestSuite(suites)

if __name__ == "__main__":
    unittest.main()
