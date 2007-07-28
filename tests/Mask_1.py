import lsst.fw.Core.fwLib as fwCore
import lsst.mwi.data as mwid
import fwTests

def doMask_1():
    """Run the code in Mask_1.cc"""
    if True:
        maskImage = fwCore.ImageViewMaskPtr(300,400)
    else:
        maskImage = fwCore.ImageViewMask(300,400)
        
    print maskImage.use_count()
    testMask = fwCore.MaskD(maskImage)
    print maskImage.use_count()

    # ------------- Test mask plane addition
    
    for p in ("CR", "BP"):
        print "Assigned %s to plane %d" % (p, testMask.addMaskPlane(p))

    for p in range(0,8):
        sp = "P%d" % p
        try:
            print "Assigned %s to plane %d" % (sp, testMask.addMaskPlane(sp))
        except IndexError, e:
            print e
            
    for p in range(0,8):
        sp = "P%d" % p
        try:
            testMask.removeMaskPlane(sp)
        except:
            pass
        
    planes = lookupPlanes(testMask, ["CR", "BP"])

    # ------------ Test mask plane metaData

    metaData = mwid.DataPropertyPtr("testMetaData")
    testMask.addMaskPlaneMetaData(metaData)
    print "MaskPlane metadata:"
    metaData._print("\t");

    print "Printing metadata from Python:"
    d = testMask.getMaskPlaneDict()
    for p in d.keys():
        if d[p]:
            print "\t", d[p], p
    
    newPlane = mwid.DataProperty("Whatever", 5)
    metaData.addProperty(newPlane)
    
    testMask.parseMaskPlaneMetaData(metaData)
    print "After loading metadata: "
    testMask.printMaskPlanes()

    # ------------ Test mask plane operations

    testMask.clearMaskPlane(planes['CR'])

    pixelList = fwCore.listPixelCoord()
    for x in range(0, 300):
        for y in range(300, 400, 20):
            pixelList.push_back(fwCore.PixelCoord(x, y))

    for p in planes.keys():
        testMask.setMaskPlaneValues(planes[p], pixelList)

    printMaskPlane(testMask, planes['CR'])

    print "\nClearing mask"
    testMask.clearMaskPlane(planes['CR'])

    printMaskPlane(testMask, planes['CR'])

    # ------------------ Test |= operator
   
    testMask3 = fwCore.MaskD(
        fwCore.ImageViewMaskPtr(testMask.getCols(), testMask.getRows())
        )

    testMask3.addMaskPlane("CR")

    testMask |= testMask3

    print "Applied |= operator"
     
    # -------------- Test mask plane removal

    testMask.clearMaskPlane(planes['BP'])
    testMask.removeMaskPlane("BP")

    print "Expect to fail to find BP:"
    planes = lookupPlanes(testMask, ["CR", "BP"])

    # --------------- Test submask methods

    testMask.setMaskPlaneValues(planes['CR'], pixelList)
    region = fwCore.BBox2i(100, 300, 10, 40)
    subTestMask = testMask.getSubMask(region)

    testMask.clearMaskPlane(planes['CR'])

    testMask.replaceSubMask(region, subTestMask)

    printMaskPlane(testMask, planes['CR'], range(90, 120), range(295, 350, 5))

    # --------------------- Test MaskPixelBooleanFunc
    testCrFuncInstance = fwTests.testCrFuncD(testMask)
    testCrFuncInstance.init() # get the latest plane info from testMask
    count = testMask.countMask(testCrFuncInstance, region)
    print "%d pixels had CR set in region" % (count)

    del testCrFuncInstance

    # should generate a vw exception - dims. of region and submask must be =
    region.expand(10)
    print "This should throw an exception:"
    try:
        testMask.replaceSubMask(region, subTestMask)
    except IndexError, e:
        print "Caught exception:", e

def test():
    """Actually run the Mask_1 test; call doMask_1 so that it can
    clean up its local variables"""
    
    doMask_1()

    if mwid.Citizen_census(0):
        print mwid.Citizen_census(0), "Objects leaked:"
        print mwid.Citizen_census(fwCore.cout)

def lookupPlanes(mask, planeNames):
    planes = {}
    for p in planeNames:
        try:
            planes[p] = mask.getMaskPlane(p)
            print "%s plane is %d" % (p, planes[p])
        except Exception, e:
            print "No %s plane found: %s" % (p, e)

    return planes

def printMaskPlane(mask, plane,
                   xrange=range(250, 300, 10), yrange=range(300, 400, 20)):
    """Print parts of the specified plane of the mask"""
    
    if True:
        xrange = range(min(xrange), max(xrange), 25)
        yrange = range(min(yrange), max(yrange), 25)

    for x in xrange:
        for y in yrange:
            if False:                   # mask(x,y) confuses swig
                print x, y, mask(x, y), mask(x, y, plane)
            else:
                print x, y, mask(x, y, plane)


if __name__ == "__main__":
    test()
