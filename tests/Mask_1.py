import sys
import fw                               # Search multiple lsst dirs on sys.path
import lsst.fw.Display.fwLib as fwLib

def test():
    if True:
        maskImage = fwLib.ImageMask(300,400)
        testMask = fwLib.MaskD(maskImage)
    else:
        testMask = fwLib.MaskD(fwLib.ImageMask(300,400))

    for p in ("CR", "BP"):
        print "Assigned %s to plane %d" % (p, testMask.addMaskPlane(p))
        
    planes = lookupPlanes(testMask, ["CR", "BP"])

    testMask.clearMaskPlane(planes['CR'])

    pixelList = fwLib.listPixelCoord()
    for x in range(0, 300):
        for y in range(300, 400, 20):
            pixelList.push_back(fwLib.PixelCoord(x, y))

    for p in planes.keys():
        testMask.setMaskPlaneValues(planes[p], pixelList);

    printMaskPlane(testMask, planes['CR'])

    print "\nClearing mask"
    testMask.clearMaskPlane(planes['CR']);

    printMaskPlane(testMask, planes['CR'])

    # -------------- Test mask plane removal

    testMask.clearMaskPlane(planes['BP']);
    testMask.removeMaskPlane("BP");

    planes = lookupPlanes(testMask, ["CR", "BP"])

    # --------------- Test submask methods

    testMask.setMaskPlaneValues(planes['CR'], pixelList)
    region = fwLib.BBox2i(100, 300, 10, 40)
    print "region:", region
    subTestMask = testMask.getSubMask(region)
    
    testMask.clearMaskPlane(planes['CR']);
    
    testMask.replaceSubMask(region, subTestMask);
    
    printMaskPlane(testMask, planes['CR'], range(90, 120), range(295, 350, 5))

    # --------------------- Test MaskPixelBooleanFunc
    testCrFuncInstance = fwLib.testCrFuncD(testMask)
    testCrFuncInstance.init() # get the latest plane info from testMask
    count = testMask.countMask(testCrFuncInstance, region)
    print "%d pixels had CR set in region" % (count)

    # should generate a vw exception - dims. of region and submask must be =
    print "This should throw an exception:"
    try:
        region.expand(10)
        testMask.replaceSubMask(region, subTestMask)
    except IndexError, e:
        print "Caught exception:", e

def lookupPlanes(mask, planeNames):
    planes = {}
    for p in planeNames:
        found, planes[p] = mask.getMaskPlane(p)
        if found:
            print "%s plane is %d" % (p, planes[p])
        else:
            print "No %s plane found" % (p)

    return planes

def printMaskPlane(mask, plane,
                   xrange=range(250, 300, 10), yrange=range(300, 400, 20)):
    """Print parts of the specified plane of the mask"""
    
    for x in xrange:
        for y in yrange:
            print x, y, mask(x, y, plane)

