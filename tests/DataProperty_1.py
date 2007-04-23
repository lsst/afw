import lsst.fw.Core.fwLib as fwCore

def doTest():
    root = fwCore.DataProperty("root")

    prop1 = fwCore.DataProperty("name1", "value1")
    prop2 = fwCore.DataProperty("name2", 2)
    prop2a = fwCore.DataProperty("name2", 4)

    root.addProperty(prop1)
    root.addProperty(prop2)

    if False:                           # this won't work, as I don't
                                        # have boost::any working from python
        class Foo:
            __slots__ = ["gurp", "murp", "durp"]
            
        foo1 = Foo()
        prop3 = fwCore.DataProperty("name3", foo1)
        root.addProperty(prop3)

    root.addProperty(prop2a)

    if True:
        dpPtr = root.find("name2")
        dpPtr._print("\t")
    
        # check find without reset to beginning
        dpPtr = root.find("name2", False)
        dpPtr._print("\t")
    
        dpPtr = root.find("name1")
        dpPtr._print("\t")
        dpPtr = root.find("name3")
        if dpPtr.get():
            dpPtr._print("\t")
        else:
            print "Failed to look up name3"
    
    # Try nested property list
     
    nested = fwCore.DataProperty("nested")
    
    nprop1 = fwCore.DataProperty("name1n", "value1")
    nprop2 = fwCore.DataProperty("name2n", 2)
    
    nested.addProperty(nprop1)
    nested.addProperty(nprop2)
    
    root.addProperty(nested)
    
    root._print("\t")
    
    # Check copy constructor
    
    rootCopy = fwCore.DataProperty(root)

    # Explicitly destroy root
    del root
    
    print "Explicit destruction done"
    
    # Check that rootCopy is still OK...
    
    rootCopy._print("\t")

def test(verbose = 0):
    fwCore.Trace_setVerbosity("fw.DataProperty", verbose)

    doTest()

    if fwCore.Citizen_census(0):
        print fwCore.Citizen_census(0), "Objects leaked:"
        print fwCore.Citizen_census(fwCore.cout)

if __name__ == "__main__":
    test()
