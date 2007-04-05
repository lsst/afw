import sys
import lsst.fw.Core.fwLib as fwCore

def test():
    if True:
        maskImage = fwCore.ImageMaskPtr(300,400)
    else:
        maskImage = fwCore.ImageMask(300,400)
        
    print maskImage.use_count()
    testMask = fwCore.MaskD(maskImage)
    print maskImage.use_count()

    for p in ("CR", "BP"):
        print "Assigned %s to plane %d" % (p, testMask.addMaskPlane(p))
        
    planes = lookupPlanes(testMask, ["CR", "BP"])

    testMask.clearMaskPlane(planes['CR'])

    pixelList = fwCore.listPixelCoord()
    for x in range(0, 300):
        for y in range(300, 400, 20):
            pixelList.push_back(fwCore.PixelCoord(x, y))

    for p in planes.keys():
        testMask.setMaskPlaneValues(planes[p], pixelList);

    printMaskPlane(testMask, planes['CR'])

    print "\nClearing mask"
    testMask.clearMaskPlane(planes['CR']);

    printMaskPlane(testMask, planes['CR'])

    # ------------------ Test |= operator
   
    testMask3 = fwCore.MaskD(
        fwCore.ImageMaskPtr(testMask.getImageCols(), testMask.getImageRows())
        )

    testMask3.addMaskPlane("CR")

    testMask |= testMask3

    print "Applied |= operator"
     
    # -------------- Test mask plane removal

    testMask.clearMaskPlane(planes['BP']);
    testMask.removeMaskPlane("BP");

    planes = lookupPlanes(testMask, ["CR", "BP"])

    # --------------- Test submask methods

    testMask.setMaskPlaneValues(planes['CR'], pixelList)
    region = fwCore.BBox2i(100, 300, 10, 40)
    subTestMask = testMask.getSubMask(region)
    
    testMask.clearMaskPlane(planes['CR']);
    
    testMask.replaceSubMask(region, subTestMask);
    
    printMaskPlane(testMask, planes['CR'], range(90, 120), range(295, 350, 5))

    # --------------------- Test MaskPixelBooleanFunc
    testCrFuncInstance = fwCore.testCrFuncD(testMask)
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

