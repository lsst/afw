import pdb                              # we may want to say pdb.set_trace()
import unittest
import lsst.fw.Core.tests as tests
import lsst.fw.Core.fwLib as fw

try:
    type(verbose)
except NameError:
    verbose = 0
    fw.Trace_setVerbosity("fw.DataProperty", verbose)

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

class DataPropertyTestCase(unittest.TestCase):
    """A test case for DataProperty"""
    def setUp(self):
        self.root = fw.DataProperty("root")

        self.values = {}; props = []
        
        n = "name1"; self.values[n] = "value1"
        props += [fw.DataProperty(n, self.values[n])]

        n = "name2"; self.values[n] = 2
        props += [fw.DataProperty(n, self.values[n])]
        props += [fw.DataProperty(n, 2*self.values[n])] # add with different value

        n = "name3"
        if False:                           # this won't work, as I don't
                                            # have boost::any working from python
            class Foo:
                __slots__ = ["gurp", "murp", "durp"]

            self.values[n] = Foo()
        else:
            self.values[n] = "Foo()"
        props += [fw.DataProperty(n, self.values[n])]

        for prop in props:
            self.root.addProperty(prop)

    def tearDown(self):
        del self.root
        self.root = None

    def testName1(self):
        """Check "name1", a string valued DataProperty"""
        
        n = "name1"
        dpPtr = self.root.find(n)
        assert dpPtr.get() != None, "Failed to find %s" % n
        self.assertEqual(dpPtr.getValueString(), self.values[n])

    def testDataPtrType(self):
        """Check that the getValueXXX routines get the right types"""
        dpPtr = self.root.find("name1")
        self.assertRaises(RuntimeError, dpPtr.getValueInt)

        dpPtr = self.root.find("name2")
        self.assertRaises(RuntimeError, dpPtr.getValueString)

    def testName2(self):
        """Check "name2", an int valued DataProperty with two definitions"""
        
        n = "name2"

        dpPtr = self.root.find(n)
        assert dpPtr.get() != None, "Failed to find %s" % n
        assert dpPtr.getValueInt() == self.values[n]

        dpPtr = self.root.find(n, False)
        assert dpPtr.get() != None, "Failed to find %s" % n
        assert dpPtr.getValueInt() == 2*self.values[n]

        dpPtr = self.root.find(n, False)
        assert dpPtr.get() == None, "Failed to find %s" % n

        dpPtr = self.root.find(n)
        self.assertEqual(dpPtr.getValueInt(), self.values[n])

    def testName3(self):
        """Check name3, which (should have) a non-{int,string} type"""
        n = "name3"
        dpPtr = self.root.find(n)
        assert dpPtr.get() != None, "Failed to find %s" % n

    def testUndefined(self):
        """Check that we can't find a data property that isn't defined"""
        dpPtr = self.root.find("undefined")
        assert dpPtr.get() == None, "Found non-existent DataProperty"

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

class NestedDataPropertyTestCase(unittest.TestCase):
    """A test case for nested DataProperty"""
    def setUp(self):
        self.root = fw.DataProperty("root")

        nested = fw.DataProperty("nested")

        self.values = {}; props = []
           
        n = "name1"; self.values[n] = "value1"
        props += [fw.DataProperty(n, self.values[n])]

        n = "name2"; self.values[n] = 2
        props += [fw.DataProperty(n, self.values[n])]

        for prop in props:
            nested.addProperty(prop)

        self.root.addProperty(nested)
        
    def tearDown(self):
        del self.root
        self.root = None

    def testCopyConstructor(self):
        """Check copy constructor"""
    
        rootCopy = fw.DataProperty(self.root)

        # Explicitly destroy root
        del self.root; self.root = None
    
        # Check that rootCopy is still OK...
        assert rootCopy.repr() != None, "rootCopy is mangled"

    def testNested(self):
        """Extract root node"""
        contents = self.root.getContents()
        self.assertEqual(len(contents), 1)
        self.assertEqual(contents[0].getName(), "nested")

    def testNested2(self):
        """Extract nested contents"""
        contents = self.root.getContents()

        for n in ("name1", "name2"):
            dpPtr = contents[0].find(n)
            assert dpPtr.get() != None, "failed to find %s" % n
            getValue = (dpPtr.getValueString if (n == "name1") else dpPtr.getValueInt)
            self.assertEqual(getValue(), self.values[n])

    def testRegex(self):
        """Find DataProperty using boost::regex"""
        contents = self.root.getContents()

        while True:
            dpPtr = contents[0].match("^name[0-9]+", False)
            if not dpPtr.get():
                break
            n = dpPtr.getName()

            getValue = (dpPtr.getValueString if (n == "name1") else dpPtr.getValueInt)
            self.assertEqual(getValue(), self.values[n])

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def suite():
    """Returns a suite containing all the test cases in this module."""
    tests.init()

    suites = []
    suites += unittest.makeSuite(DataPropertyTestCase)
    suites += unittest.makeSuite(NestedDataPropertyTestCase)
    suites += unittest.makeSuite(tests.MemoryTestCase)
    return unittest.TestSuite(suites)

if __name__ == "__main__":
    unittest.main()
