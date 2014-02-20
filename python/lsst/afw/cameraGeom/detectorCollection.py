from lsst.afw.geom import Box2D, Point2D
from lsst.afw.cameraGeom import FOCAL_PLANE, PIXELS

class DetectorCollection(object):
    """An immutable collection of Detectors that can be accessed in various ways
    """
    def __init__(self, detectorList):
        """Construct a DetectorCollection
        
        @param[in] detectorList: a sequence of detectors in index order
        """
        self._detectorList = tuple(detectorList)
        self._nameDetectorDict = dict((d.getName(), d) for d in detectorList)
        self._serialDectorDict = dict((d.getSerial(), d) for d in detectorList)
        self._fpBBox = Box2D()
        for detector in self._detectorList:
            for corner in detector.getCorners(FOCAL_PLANE):
                self._fpBBox.include(corner)

    def __iter__(self):
        """Return an iterator over all detectors in this collection"""
        return self._detectorList.__iter__()

    def __len__(self):
        """Return the number of detectors in this collection"""
        return len(self._detectorList)

    def getDetectorByName(self, name):
        """Return a detector given its name"""
        return self._nameDetectorDict[name]
    
    def getDetectorByIndex(self, index):
        """Return a detector given its index"""
        return self._detectorList[index]
    
    def getDetectorBySerial(self, serial):
        """Return a detector given its serial number"""
        return self._serialDectorDict[serial]
    
    def getFpBBox(self):
        """Return a focal plane bounding box that encompasses all detectors
        """
        return self._fpBBox
