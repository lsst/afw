"""demonstrate simple use of the Log facility."""
"""
Test Log facility

Run with:
   python Log_1.py
or
   python
   >>> import unittest; T=load("Log_1"); unittest.TextTestRunner(verbosity=1).run(T.suite())
"""

import pdb                              # we may want to say pdb.set_trace()
import unittest
import lsst.fw.Core.tests as tests
import lsst.fw.Core.fwLib as fw

try:
    type(verbose)
except NameError:
    verbose = 0
    fw.Trace_setVerbosity("fw.Log", verbose)

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-


class LogTestCase(unittest.TestCase):
    """A test case for DetectionSet"""

    def setUp(self):
        print "\nVerbosity levels:\n";
        fw.Log_printVerbosity(fw.cout);

        # Branching Node:  root
        root = fw.DataProperty("root")
        # Terminal Node:  prop1
        prop1 = fw.DataProperty("prop1", "value1")
        # Terminal Node:  prop2
        prop2 = fw.DataProperty("prop2", 2)
        # Terminal Node:  root.prop1
        root.addProperty(prop1)
        # Terminal Node:  root.prop2
        root.addProperty(prop2)

        # Branching Node:  branch1
        branch = fw.DataProperty("branch1")
        # Terminal Node:  prop3
        prop3 = fw.DataProperty("prop3", 3)
        # Terminal Node:  branch1.prop3
        branch.addProperty(prop3)
        # Terminal Node:  root.branch1.prop3
        root.addProperty(branch)

        # Branching Node:  newroot
        root1 = fw.DataProperty("newroot")
        # Terminal Node:  prop4
        prop4 = fw.DataProperty("prop4", 4)
        root1.addProperty(prop4)
        # Terminal Node:  root.prop4

        self.root = root
        self.root1 = root1
        self.branch = branch

    def tearDown(self):
        del self.root
        del self.root1
        del self.branch        

    def workLog(self):
        # Test use of various UI options for logging
        fw.Log("foo", 1, self.branch)
        fw.Log("foo.bar",2) << self.root << self.root1

        fw.Log("foo.bar.goo", 4, fw.DataProperty("inlineKeyword", "inlineKvalue"))

        fw.Log("foo.bar.goo.hoo", 3, fw.DataProperty("CurKeyword", "CurKvalue")) << self.branch
        fw.Log("foo.tar",5) << \
                            fw.DataProperty("NewKeyword1", "NewKvalue1") \
                            << fw.DataProperty("NewKeyword2", "NewKvalue2")
        
    def testLog1(self):
        """Test the Log class.

        If you set logName to None, diagnostic messages and logging records should
        be emitted to the same output stream.  Otherwise, the logging
        records will be emitted to logName
        """

        logName = "MyLog.log" if False else None
        if logName:
            # direct the Log records to a separate file.
            fw.Log_setDestination(logName)
        else:
            # write to stdout
            fw.Log_setDestination(fw.cout)

        fw.Log_setVerbosity(".", 100)
        self.workLog()

        fw.Log_setVerbosity(".", 0)
        fw.Log_setVerbosity("foo.bar", 3)
        fw.Log_setVerbosity("foo.bar.goo", 10)
        fw.Log_setVerbosity("foo.tar", 5)
        self.workLog()

        fw.Log_setVerbosity("foo.tar")
        fw.Log_setVerbosity("foo.bar")
        self.workLog()

        print "\nReset."
        fw.Log_reset()
        self.workLog()

        fw.Log_setVerbosity("", 1)
        fw.Log_setVerbosity("foo.bar.goo.hoo", 10)
        self.workLog()

        fw.Log_setVerbosity("", 2)
        self.workLog()

        fw.Log_setVerbosity("")
        fw.Log_setVerbosity("foo.bar.goo.hoo")
        fw.Log_setVerbosity("foo.bar.goo.hoo.joo", 10)
        fw.Log_setVerbosity("foo.bar.goo", 3)
        self.workLog()

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def suite():
    """Returns a suite containing all the test cases in this module."""
    tests.init()

    suites = []
    suites += unittest.makeSuite(LogTestCase)
    suites += unittest.makeSuite(tests.MemoryTestCase)
    return unittest.TestSuite(suites)

if __name__ == "__main__":
    unittest.main()
