#
# LSST Data Management System
# Copyright 2016 LSST Corporation.
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
from __future__ import absolute_import, division

__all__ = ["BoxGrid"]

from builtins import range
from builtins import object

import os

import astshim
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from astshim.test import makeForwardPolyMap, makeTwoWayPolyMap

from .box import Box2I, Box2D
import lsst.afw.geom as afwGeom
from lsst.pex.exceptions import InvalidParameterError
import lsst.utils
import lsst.utils.tests


class BoxGrid(object):
    """!Divide a box into nx by ny sub-boxes that tile the region
    """
    def __init__(self, box, numColRow):
        """!Construct a BoxGrid

        The sub-boxes will be of the same type as `box` and will exactly tile `box`;
        they will also all be the same size, to the extent possible (some variation
        is inevitable for integer boxes that cannot be evenly divided.

        @param[in] box  box (an lsst.afw.geom.Box2I or Box2D);
                        the boxes in the grid will be of the same type
        @param[in] numColRow  number of columns and rows (a pair of ints)
        """
        if len(numColRow) != 2:
            raise RuntimeError("numColRow=%r; must be a sequence of two integers" % (numColRow,))
        self._numColRow = tuple(int(val) for val in numColRow)

        if isinstance(box, Box2I):
            stopDelta = 1
        elif isinstance(box, Box2D):
            stopDelta = 0
        else:
            raise RuntimeError("Unknown class %s of box %s" % (type(box), box))
        self.boxClass = type(box)
        self.stopDelta = stopDelta

        minPoint = box.getMin()
        self.pointClass = type(minPoint)
        dtype = np.array(minPoint).dtype

        self._divList = [np.linspace(start=box.getMin()[i],
                                     stop=box.getMax()[i] + self.stopDelta,
                                     num=self._numColRow[i] + 1,
                                     endpoint=True,
                                     dtype=dtype) for i in range(2)]

    @property
    def numColRow(self):
        return self._numColRow

    def __getitem__(self, indXY):
        """!Return the box at the specified x,y index (a pair of ints)
        """
        beg = self.pointClass(*[self._divList[i][indXY[i]] for i in range(2)])
        end = self.pointClass(*[self._divList[i][indXY[i] + 1] - self.stopDelta for i in range(2)])
        return self.boxClass(beg, end)

    def __len__(self):
        return self.shape[0]*self.shape[1]

    def __iter__(self):
        """!Return an iterator over all boxes, where column varies most quickly
        """
        for row in range(self.numColRow[1]):
            for col in range(self.numColRow[0]):
                yield self[col, row]


class TransformTestBaseClass(lsst.utils.tests.TestCase):
    """Base class for unit tests of Transform<X>To<Y> and subclasses

    Subclasses must call `TransformTestBaseClass.setUp(self)`
    if they provide their own version.

    If a package other than afw uses this class then it must
    override the `getTestDir` method to avoid writing into
    afw's test directory.
    """

    def getTestDir(self):
        """Return a directory where temporary test files can be written

        This is typically the test dir of the package in which the test lives
        """
        return os.path.join(lsst.utils.getPackageDir("afw"), "tests")

    def setUp(self):
        # tell unittest to use the msg argument of asserts as a supplement
        # to the error message, rather than as the whole error message
        self.longMessage = True

        # list of endpoint class name prefixes; the full name is prefix + "Endpoint"
        self.endpointPrefixes = ("Generic", "Point2", "Point3", "SpherePoint")

        # GoodNaxes is dict of endpoint class name prefix:
        #    tuple containing 0 or more valid numbers of axes
        self.goodNaxes = {
            "Generic": (1, 2, 3, 4),  # all numbers of axes are valid for GenericEndpoint
            "Point2": (2,),
            "Point3": (3,),
            "SpherePoint": (2,),
        }

        # BadAxes is dict of endpoint class name prefix:
        #    tuple containing 0 or more invalid numbers of axes
        self.badNaxes = {
            "Generic": (),  # all numbers of axes are valid for GenericEndpoint
            "Point2": (1, 3, 4),
            "Point3": (1, 2, 4),
            "SpherePoint": (1, 3, 4),
        }

    @staticmethod
    def makeRawArrayData(nPoints, nAxes, delta=0.123):
        return np.array([[i + j*delta for j in range(nAxes)] for i in range(nPoints)])

    @staticmethod
    def makeRawPointData(nAxes, delta=0.123):
        return [i*delta for i in range(nAxes)]

    @staticmethod
    def makeEndpoint(name, nAxes):
        """Make an endpoint

        Parameters
        ----------
        name : string
            one of "Generic", "Point2", "Point3" or "SpherePoint"
        nAxes : integer
            number of axes; ignored if the name is not "Generic"
        """
        endpointClassName = name + "Endpoint"
        endpointClass = getattr(afwGeom, endpointClassName)
        if name == "Generic":
            return endpointClass(nAxes)
        return endpointClass()

    @classmethod
    def makeGoodFrame(cls, name, nAxes):
        """Return the appropriate frame for the given name and nAxes

        Parameters
        ----------
        name : string
            one of "Generic", "Point2", "Point3" or "SpherePoint"
        nAxes : integer
            number of axes; ignored if the name is not "Generic"
        """
        return cls.makeEndpoint(name, nAxes).makeFrame()

    @staticmethod
    def makeBadFrames(name):
        """Return a list of 0 or more frames that are not a valid match for the named endpoint

        Parameters
        ----------
        name : string
            one of "Generic", "Point2", "Point3" or "SpherePoint"
        """
        if name == "Generic":
            return []
        elif name == "Point2":
            return [
                astshim.SkyFrame(),
                astshim.Frame(1),
                astshim.Frame(3),
            ]
        elif name == "Point3":
            return [
                astshim.SkyFrame(),
                astshim.Frame(2),
                astshim.Frame(4),
            ]
        if name == "SpherePoint":
            return [
                astshim.Frame(1),
                astshim.Frame(2),
                astshim.Frame(3),
            ]
        raise RuntimeError("Unrecognized name={}".format(name))

    @staticmethod
    def makeFrameSet(baseFrame, currFrame):
        """Make a FrameSet

        The FrameSet will contain 4 frames and three transforms conneting them:

        Frame       Index   Ident       Mapping from this frame to the next
        baseFrame     1     baseFrame   UnitMap(nIn)
        Frame(nIn)    2     frame2      polyMap
        Frame(nOut)   3     frame3      UnitMap(nOut)
        currFrame     4     currFrame

        where:
        - nIn = baseFrame.getNaxes()
        - nOut = currFrame.getNaxes()
        - polyMap = makeTwoWayPolyMap(nIn, nOut)

        Parameters
        ----------
        baseFrame : astshim.Frame
            base frame
        currFrame : astshim.Frame
            current frame
        """
        nIn = baseFrame.getNaxes()
        nOut = currFrame.getNaxes()
        polyMap = makeTwoWayPolyMap(nIn, nOut)

        # The only to set the Ident of a frame in a FrameSet is to set it in advance,
        # and I don't want to modify the inputs, so replace the input frames with copies
        baseFrame = baseFrame.copy()
        baseFrame.setIdent("baseFrame")
        currFrame = currFrame.copy()
        currFrame.setIdent("currFrame")

        frameSet = astshim.FrameSet(baseFrame)
        frame2 = astshim.Frame(nIn, "Ident=frame2")
        frameSet.addFrame(astshim.FrameSet.CURRENT, astshim.UnitMap(nIn), frame2)
        frame3 = astshim.Frame(nOut, "Ident=frame3")
        frameSet.addFrame(astshim.FrameSet.CURRENT, polyMap, frame3)
        frameSet.addFrame(astshim.FrameSet.CURRENT, astshim.UnitMap(nOut), currFrame)
        frameSet.setIdent("currFrame")
        return frameSet

    @staticmethod
    def permFrameSetIter(frameSet):
        """Iterator over 0 or more frameSets with SkyFrames axes permuted

        Only base and current SkyFrames are permuted. If neither the base nor the
        current frame is a SkyFrame then no frames are returned.

        Each returned value is a tuple:
        - permFrameSet: a copy of frameSet with the base and/or current frame permuted
        - isBaseSkyFrame: a boolean
        - isCurrSkyFrame: a boolean
        - isBasePermuted: a boolean
        - isCurrPermuted: a boolean
        """
        baseInd = frameSet.getBase()
        currInd = frameSet.getCurrent()
        isBaseSkyFrame = frameSet.getFrame(baseInd).getClass() == "SkyFrame"
        isCurrSkyFrame = frameSet.getFrame(currInd).getClass() == "SkyFrame"
        if not (isBaseSkyFrame or isCurrSkyFrame):
            return

        baseList = [False, True] if isBaseSkyFrame else [False]
        currList = [False, True] if isCurrSkyFrame else [False]
        for isBasePermuted in baseList:
            for isCurrPermuted in currList:
                frameSetCopy = frameSet.copy()
                if isBasePermuted:
                    assert isBaseSkyFrame
                    frameSetCopy.setCurrent(baseInd)
                    frameSetCopy.permAxes([2, 1])
                    frameSetCopy.setCurrent(currInd)
                if isCurrPermuted:
                    assert isCurrSkyFrame
                    frameSetCopy.permAxes([2, 1])
                yield (frameSetCopy, isBaseSkyFrame, isCurrSkyFrame, isBasePermuted, isCurrPermuted)

    @staticmethod
    def makeJacobian(nIn, nOut, inArray):
        """Make a Jacobian matrix for the equation described by makeTwoWayPolyMap.

        Parameters
        ----------
        nIn, nOut : integers
            the dimensions of the input and output data; see makeTwoWayPolyMap
        inArray : ndarray
            an array of size `nIn` representing the point at which the Jacobian
            is measured

        Returns
        ----------
        J : numpy.ndarray
            an nOut x nIn array of first derivatives
        """
        baseCoeff = 2.0 * 0.001
        coeffs = np.empty((nOut, nIn))
        for iOut in range(nOut):
            coeffOffset = baseCoeff * iOut
            for iIn in range(nIn):
                coeffs[iOut, iIn] = baseCoeff * (iIn + 1) + coeffOffset
                coeffs[iOut, iIn] *= inArray[iIn]
        assert coeffs.ndim == 2
        assert coeffs.shape == (nOut, nIn)
        # Avoid spurious errors when comparing to a simplified array
        return coeffs

    def checkTransformation(self, transform, mapping, msg=""):
        """Check tranForward and tranInverse for a transform

        Parameters
        ----------
        transform : Transform
            The transform to check
        mapping : astshim.Mapping
            The mapping the transform should use. This mapping
            must contain valid forward or inverse transformations,
            but they need not match if both present. Hence the
            mappings returned by make*PolyMap are acceptable.
        msg : string
            Error message suffix describing test parameters
        """
        fromEndpoint = transform.getFromEndpoint()
        toEndpoint = transform.getToEndpoint()
        frameSet = transform.getFrameSet()

        nIn = mapping.getNin()
        nOut = mapping.getNout()
        self.assertEqual(nIn, fromEndpoint.getNAxes(), msg=msg)
        self.assertEqual(nOut, toEndpoint.getNAxes(), msg=msg)

        # forward transformation of one point
        rawInPoint = self.makeRawPointData(nIn)
        inPoint = fromEndpoint.pointFromData(rawInPoint)

        # forward transformation of an array of points
        nPoints = 7  # arbitrary
        rawInArray = self.makeRawArrayData(nPoints, nIn)
        inArray = fromEndpoint.arrayFromData(rawInArray)

        if mapping.hasForward():
            self.assertTrue(transform.hasForward())
            outPoint = transform.tranForward(inPoint)
            rawOutPoint = toEndpoint.dataFromPoint(outPoint)
            assert_allclose(rawOutPoint, mapping.tranForward(rawInPoint), err_msg=msg)
            assert_allclose(rawOutPoint, frameSet.tranForward(rawInPoint), err_msg=msg)

            outArray = transform.tranForward(inArray)
            rawOutArray = toEndpoint.dataFromArray(outArray)
            self.assertFloatsAlmostEqual(rawOutArray, mapping.tranForward(rawInArray), msg=msg)
            self.assertFloatsAlmostEqual(rawOutArray, frameSet.tranForward(rawInArray), msg=msg)
        else:
            # Need outPoint, but don't need it to be consistent with inPoint
            rawOutPoint = self.makeRawPointData(nOut)
            outPoint = toEndpoint.pointFromData(rawOutPoint)
            rawOutArray = self.makeRawArrayData(nPoints, nOut)
            outArray = toEndpoint.arrayFromData(rawOutArray)

            self.assertFalse(transform.hasForward())

        if mapping.hasInverse():
            self.assertTrue(transform.hasInverse())
            # inverse transformation of one point;
            # remember that the inverse will not give the original values!
            inversePoint = transform.tranInverse(outPoint)
            rawInversePoint = fromEndpoint.dataFromPoint(inversePoint)
            assert_allclose(rawInversePoint, mapping.tranInverse(rawOutPoint), err_msg=msg)
            assert_allclose(rawInversePoint, frameSet.tranInverse(rawOutPoint), err_msg=msg)

            # inverse transformation of an array of points;
            # remember that the inverse will not give the original values!
            inverseArray = transform.tranInverse(outArray)
            rawInverseArray = fromEndpoint.dataFromArray(inverseArray)
            self.assertFloatsAlmostEqual(rawInverseArray, mapping.tranInverse(rawOutArray), msg=msg)
            self.assertFloatsAlmostEqual(rawInverseArray, frameSet.tranInverse(rawOutArray), msg=msg)
        else:
            self.assertFalse(transform.hasInverse())

    def checkInverseTransformation(self, forward, inverse, msg=""):
        """Check that two Transforms are each others' inverses.

        Parameters
        ----------
        forward : Transform
            the reference Transform to test
        inverse : Transform
            the transform that should be the inverse of `forward`
        msg : string
            error message suffix describing test parameters
        """
        fromEndpoint = forward.getFromEndpoint()
        toEndpoint = forward.getToEndpoint()
        frameSet = forward.getFrameSet()
        invFrameSet = inverse.getFrameSet()

        # properties
        self.assertEqual(forward.getFromEndpoint(),
                         inverse.getToEndpoint(), msg=msg)
        self.assertEqual(forward.getToEndpoint(),
                         inverse.getFromEndpoint(), msg=msg)
        self.assertEqual(forward.hasForward(), inverse.hasInverse(), msg=msg)
        self.assertEqual(forward.hasInverse(), inverse.hasForward(), msg=msg)

        # transformations of one point
        # we don't care about whether the transformation itself is correct
        # (see checkTransformation), so inPoint/outPoint need not be related
        rawInPoint = self.makeRawPointData(fromEndpoint.getNAxes())
        inPoint = fromEndpoint.pointFromData(rawInPoint)
        rawOutPoint = self.makeRawPointData(toEndpoint.getNAxes())
        outPoint = toEndpoint.pointFromData(rawOutPoint)

        # transformations of arrays of points
        nPoints = 7  # arbitrary
        rawInArray = self.makeRawArrayData(nPoints, fromEndpoint.getNAxes())
        inArray = fromEndpoint.arrayFromData(rawInArray)
        rawOutArray = self.makeRawArrayData(nPoints, toEndpoint.getNAxes())
        outArray = toEndpoint.arrayFromData(rawOutArray)

        if forward.hasForward():
            self.assertEqual(forward.tranForward(inPoint),
                             inverse.tranInverse(inPoint), msg=msg)
            self.assertEqual(frameSet.tranForward(rawInPoint),
                             invFrameSet.tranInverse(rawInPoint), msg=msg)
            # Assertions must work with both lists and numpy arrays
            assert_array_equal(forward.tranForward(inArray),
                               inverse.tranInverse(inArray),
                               err_msg=msg)
            assert_array_equal(frameSet.tranForward(rawInArray),
                               invFrameSet.tranInverse(rawInArray),
                               err_msg=msg)

        if forward.hasInverse():
            self.assertEqual(forward.tranInverse(outPoint),
                             inverse.tranForward(outPoint), msg=msg)
            self.assertEqual(frameSet.tranInverse(rawOutPoint),
                             invFrameSet.tranForward(rawOutPoint), msg=msg)
            assert_array_equal(forward.tranInverse(outArray),
                               inverse.tranForward(outArray),
                               err_msg=msg)
            assert_array_equal(frameSet.tranInverse(rawOutArray),
                               invFrameSet.tranForward(rawOutArray),
                               err_msg=msg)

    def checkTransformFromMapping(self, fromName, toName):
        """Check a Transform_<fromName>_<toName> using the Mapping constructor

        Parameters
        ----------
        fromName, toName : string
            one of self.endpointPrefixes
        fromAxes, toAxes : integer
            number of axes in fromFrame and toFrame, respectively
        """
        transformClassName = "Transform{}To{}".format(fromName, toName)
        transformClass = getattr(afwGeom, transformClassName)
        baseMsg = "transformClass={}".format(transformClass.__name__)
        for nIn in self.goodNaxes[fromName]:
            # check invalid numbers of output for the given toName
            for badNout in self.badNaxes[toName]:
                badPolyMap = makeTwoWayPolyMap(nIn, badNout)
                msg = "{}, nIn={}, badNout={}".format(baseMsg, nIn, badNout)
                with self.assertRaises(InvalidParameterError, msg=msg):
                    transformClass(badPolyMap)

            # check valid numbers of outputs for the given toName
            for nOut in self.goodNaxes[toName]:
                msg = "{}, nIn={}, nOut={}".format(baseMsg, nIn, nOut)
                polyMap = makeTwoWayPolyMap(nIn, nOut)
                transform = transformClass(polyMap)

                desStr = "{}[{}->{}]".format(transformClassName, nIn, nOut)
                self.assertEqual("{}".format(transform), desStr)
                self.assertEqual(repr(transform), "lsst.afw.geom." + desStr)

                self.checkTransformation(transform, polyMap, msg=msg)

                # Forward transform but no inverse
                polyMap = makeForwardPolyMap(nIn, nOut)
                transform = transformClass(polyMap)
                self.checkTransformation(transform, polyMap, msg=msg)

                # Inverse transform but no forward
                polyMap = makeForwardPolyMap(nOut, nIn).getInverse()
                transform = transformClass(polyMap)
                self.checkTransformation(transform, polyMap, msg=msg)

        # check invalid numbers of inputs with valid and invalid #s of inputs
        for badNin in self.badNaxes[fromName]:
            for nOut in self.goodNaxes[toName] + self.badNaxes[toName]:
                badPolyMap = makeTwoWayPolyMap(badNin, nOut)
                msg = "{}, badNin={}, nOut={}".format(baseMsg, nIn, nOut)
                with self.assertRaises(InvalidParameterError, msg=msg):
                    transformClass(badPolyMap)

    def checkTransformFromFrameSet(self, fromName, toName):
        transformClassName = "Transform{}To{}".format(fromName, toName)
        transformClass = getattr(afwGeom, transformClassName)
        baseMsg = "transformClass={}".format(transformClass.__name__)
        for nIn in self.goodNaxes[fromName]:
            for nOut in self.goodNaxes[toName]:
                msg = "{}, nIn={}, nOut={}".format(baseMsg, nIn, nOut)

                baseFrame = self.makeGoodFrame(fromName, nIn)
                currFrame = self.makeGoodFrame(toName, nOut)
                frameSet = self.makeFrameSet(baseFrame, currFrame)
                self.assertEqual(frameSet.getNframe(), 4)

                # construct 0 or more frame sets that are invalid for this transform class
                for badBaseFrame in self.makeBadFrames(fromName):
                    badFrameSet = self.makeFrameSet(badBaseFrame, currFrame)
                    with self.assertRaises(InvalidParameterError):
                        transformClass(badFrameSet)
                    for badCurrFrame in self.makeBadFrames(toName):
                        reallyBadFrameSet = self.makeFrameSet(badBaseFrame, badCurrFrame)
                        with self.assertRaises(InvalidParameterError):
                            transformClass(reallyBadFrameSet)
                for badCurrFrame in self.makeBadFrames(toName):
                    badFrameSet = self.makeFrameSet(baseFrame, badCurrFrame)
                    with self.assertRaises(InvalidParameterError):
                        transformClass(badFrameSet)

                transform = transformClass(frameSet)

                desStr = "{}[{}->{}]".format(transformClassName, nIn, nOut)
                self.assertEqual("{}".format(transform), desStr)
                self.assertEqual(repr(transform), "lsst.afw.geom." + desStr)

                self.checkPersistence(transform)

                frameSetCopy = transform.getFrameSet()

                self.assertEqual(frameSet.getNframe(), frameSetCopy.getNframe())
                self.assertEqual(frameSet.getFrame(1).getIdent(), "baseFrame")
                self.assertEqual(frameSet.getFrame(2).getIdent(), "frame2")
                self.assertEqual(frameSet.getFrame(3).getIdent(), "frame3")
                self.assertEqual(frameSet.getFrame(4).getIdent(), "currFrame")

                polyMap = makeTwoWayPolyMap(nIn, nOut)

                self.checkTransformation(transform, mapping=polyMap, msg=msg)

                # If the base and/or current frame of frameSet is a SkyFrame,
                # try permuting that frame (in place, so the connected mappings are
                # correctly updated). The Transform constructor should undo the permutation,
                # (via SpherePointEndpoint.normalizeFrame) in its internal copy of frameSet,
                # forcing the axes of the SkyFrame into standard (longitude, latitude) order
                for permFrameSet, isBaseSkyFrame, isCurrSkyFrame, \
                        isBasePermuted, isCurrPermuted in self.permFrameSetIter(frameSet):
                    # sanity check the data
                    if isBasePermuted:
                        self.assertTrue(isBaseSkyFrame)
                    if isCurrPermuted:
                        self.assertTrue(isCurrSkyFrame)
                    if isBaseSkyFrame:
                        baseFrame = permFrameSet.getFrame(astshim.FrameSet.BASE)
                        desBaseLonAxis = 2 if isBasePermuted else 1
                        self.assertEqual(baseFrame.getLonAxis(), desBaseLonAxis)
                    if isCurrSkyFrame:
                        currFrame = permFrameSet.getFrame(astshim.FrameSet.CURRENT)
                        desCurrLonAxis = 2 if isCurrPermuted else 1
                        self.assertEqual(currFrame.getLonAxis(), desCurrLonAxis)

                    permTransform = transformClass(permFrameSet)
                    # If the base and/or current frame is a SkyFrame then make sure the frame
                    # in the *Transform* has axes in standard (longitude, latitude) order
                    unpermFrameSet = permTransform.getFrameSet()
                    if isBaseSkyFrame:
                        self.assertEqual(unpermFrameSet.getFrame(astshim.FrameSet.BASE).getLonAxis(), 1)
                    if isCurrSkyFrame:
                        self.assertEqual(unpermFrameSet.getFrame(astshim.FrameSet.CURRENT).getLonAxis(), 1)

                    self.checkTransformation(permTransform, mapping=polyMap, msg=msg)

    def checkGetInverse(self, fromName, toName):
        """Test Transform<fromName>To<toName>.getInverse

        Parameters
        ----------
        fromName, toName : string
            the prefixes of the transform's endpoints (e.g., "Point2" for a
            Point2Endpoint)
        """
        transformClassName = "Transform{}To{}".format(fromName, toName)
        transformClass = getattr(afwGeom, transformClassName)
        baseMsg = "transformClass={}".format(transformClass.__name__)
        for nIn in self.goodNaxes[fromName]:
            for nOut in self.goodNaxes[toName]:
                msg = "{}, nIn={}, nOut={}".format(baseMsg, nIn, nOut)
                self.checkInverseMapping(
                    transformClass,
                    makeTwoWayPolyMap(nIn, nOut),
                    "{}, Map={}".format(msg, "TwoWay"))
                self.checkInverseMapping(
                    transformClass,
                    makeForwardPolyMap(nIn, nOut),
                    "{}, Map={}".format(msg, "Forward"))
                self.checkInverseMapping(
                    transformClass,
                    makeForwardPolyMap(nOut, nIn).getInverse(),
                    "{}, Map={}".format(msg, "Inverse"))

                self.checkInverseFrameSet(transformClass,
                                          self.makeGoodFrame(fromName, nIn),
                                          self.makeGoodFrame(toName, nOut))

    def checkInverseMapping(self, clsTransform, mapping, msg):
        """Test Transform<fromName>To<toName>.getInverse for a specific mapping.

        Parameters
        ----------
        clsTransform : type
            the transform to test
        mapping : Mapping
            the map to test `clsTransform` with
        msg : string
            a suffix for error messages, distinguishing this test from others
        """
        transform = clsTransform(mapping)
        inverse = transform.getInverse()
        inverseInverse = inverse.getInverse()

        self.checkInverseTransformation(transform, inverse, msg=msg)
        self.checkInverseTransformation(inverse, inverseInverse, msg=msg)
        self.checkTransformation(inverseInverse, mapping, msg=msg)

    def checkInverseFrameSet(self, clsTransform, frameIn, frameOut):
        """Test whether inverting a Transform preserves all information
           in its FrameSet.

        Parameters
        ----------
        clsTransform : type
            the transform to test
        frameIn, frameOut : Frame
            the frames to between which `clsTransform` shall convert. Must be
            compatible with `clsTransform`.
        """
        frameSet = self.makeFrameSet(frameIn, frameOut)
        self.assertEqual(frameSet.getNframe(), 4)

        baseMsg = "transformClass={}, nIn={}, nOut={}".format(
            clsTransform.__name__, frameIn.getNaxes(), frameOut.getNaxes())
        transform = clsTransform(frameSet)
        forwardFrames = transform.getFrameSet()
        self.assertFalse(forwardFrames.isInverted())
        self.assertEqual(forwardFrames.getNframe(), frameSet.getNframe(),
                         msg=baseMsg)
        self.assertEqual(forwardFrames.getFrame(1).getIdent(), "baseFrame",
                         msg=baseMsg)
        self.assertEqual(forwardFrames.getFrame(2).getIdent(), "frame2",
                         msg=baseMsg)
        self.assertEqual(forwardFrames.getFrame(3).getIdent(), "frame3",
                         msg=baseMsg)
        self.assertEqual(forwardFrames.getFrame(4).getIdent(), "currFrame",
                         msg=baseMsg)

        reverseFrames = transform.getInverse().getFrameSet()
        self.assertTrue(reverseFrames.isInverted())
        self.assertEqual(reverseFrames.getNframe(), frameSet.getNframe(),
                         msg=baseMsg)
        self.assertEqual(reverseFrames.getFrame(1).getIdent(), "baseFrame",
                         msg=baseMsg)
        self.assertEqual(reverseFrames.getFrame(2).getIdent(), "frame2",
                         msg=baseMsg)
        self.assertEqual(reverseFrames.getFrame(3).getIdent(), "frame3",
                         msg=baseMsg)
        self.assertEqual(reverseFrames.getFrame(4).getIdent(), "currFrame",
                         msg=baseMsg)

    def checkGetJacobian(self, fromName, toName):
        """Test Transform<fromName>To<toName>.getJacobian

        Parameters
        ----------
        fromName, toName : string
            the prefixes of the transform's endpoints (e.g., "Point2" for a
            Point2Endpoint)
        """
        transformClassName = "Transform{}To{}".format(fromName, toName)
        transformClass = getattr(afwGeom, transformClassName)
        baseMsg = "transformClass={}".format(transformClass.__name__)
        for nIn in self.goodNaxes[fromName]:
            for nOut in self.goodNaxes[toName]:
                msg = "{}, nIn={}, nOut={}".format(baseMsg, nIn, nOut)
                polyMap = makeForwardPolyMap(nIn, nOut)
                transform = transformClass(polyMap)
                fromEndpoint = transform.getFromEndpoint()

                # Test multiple points to ensure correct functional form
                rawInPoint = self.makeRawPointData(nIn)
                inPoint = fromEndpoint.pointFromData(rawInPoint)
                jacobian = transform.getJacobian(inPoint)
                assert_allclose(jacobian, self.makeJacobian(nIn, nOut, rawInPoint),
                                err_msg=msg)

                rawInPoint = self.makeRawPointData(nIn, 0.111)
                inPoint = fromEndpoint.pointFromData(rawInPoint)
                jacobian = transform.getJacobian(inPoint)
                assert_allclose(jacobian, self.makeJacobian(nIn, nOut, rawInPoint),
                                err_msg=msg)

    def checkOf(self, fromName, midName, toName):
        """Test Transform<midName>To<toName>.of(Transform<fromName>To<midName>)

        Parameters
        ----------
        fromName : string
            the prefix of the starting endpoint (e.g., "Point2" for a
            Point2Endpoint) for the final, concatenated Transform
        midName : string
            the prefix for the shared endpoint where two Transforms will be
            concatenated
        toName : string
            the prefix of the ending endpoint for the final, concatenated
            Transform
        """
        transform1Class = getattr(afwGeom,
                                  "Transform{}To{}".format(fromName, midName))
        transform2Class = getattr(afwGeom,
                                  "Transform{}To{}".format(midName, toName))
        baseMsg = "{}.of({})".format(transform2Class.__name__,
                                     transform1Class.__name__)
        for nIn in self.goodNaxes[fromName]:
            for nMid in self.goodNaxes[midName]:
                for nOut in self.goodNaxes[toName]:
                    msg = "{}, nIn={}, nMid={}, nOut={}".format(
                        baseMsg, nIn, nMid, nOut)
                    polyMap = makeTwoWayPolyMap(nIn, nMid)
                    transform1 = transform1Class(polyMap)
                    polyMap = makeTwoWayPolyMap(nMid, nOut)
                    transform2 = transform2Class(polyMap)
                    transform = transform2.of(transform1)

                    fromEndpoint = transform1.getFromEndpoint()
                    toEndpoint = transform2.getToEndpoint()

                    inPoint = fromEndpoint.pointFromData(self.makeRawPointData(nIn))
                    outPointMerged = transform.tranForward(inPoint)
                    outPointSeparate = transform2.tranForward(
                        transform1.tranForward(inPoint))
                    assert_allclose(toEndpoint.dataFromPoint(outPointMerged),
                                    toEndpoint.dataFromPoint(outPointSeparate),
                                    err_msg=msg)

                    outPoint = toEndpoint.pointFromData(self.makeRawPointData(nOut))
                    inPointMerged = transform.tranInverse(outPoint)
                    inPointSeparate = transform1.tranInverse(
                        transform2.tranInverse(outPoint))
                    assert_allclose(
                        fromEndpoint.dataFromPoint(inPointMerged),
                        fromEndpoint.dataFromPoint(inPointSeparate),
                        err_msg=msg)

        # Mismatched number of axes should fail
        if midName == "Generic":
            nIn = self.goodNaxes[fromName][0]
            nOut = self.goodNaxes[toName][0]
            polyMap = makeTwoWayPolyMap(nIn, 3)
            transform1 = transform1Class(polyMap)
            polyMap = makeTwoWayPolyMap(2, nOut)
            transform2 = transform2Class(polyMap)
            with self.assertRaises(InvalidParameterError):
                transform = transform2.of(transform1)

        # Mismatched types of endpoints should fail
        if fromName != midName:
            # Use transform1Class for both args to keep test logic simple
            outName = midName
            joinNaxes = set(self.goodNaxes[fromName]).intersection(
                self.goodNaxes[outName])
            for nIn in self.goodNaxes[fromName]:
                for nMid in joinNaxes:
                    for nOut in self.goodNaxes[outName]:
                        polyMap = makeTwoWayPolyMap(nIn, nMid)
                        transform1 = transform1Class(polyMap)
                        polyMap = makeTwoWayPolyMap(nMid, nOut)
                        transform2 = transform1Class(polyMap)
                        with self.assertRaises(InvalidParameterError):
                            transform = transform2.of(transform1)

    def checkOfChaining(self):
        """Test that both conventions for chaining Transform*To*.of give
        the same result
        """
        transform1 = afwGeom.TransformGenericToGeneric(
            makeForwardPolyMap(2, 3))
        transform2 = afwGeom.TransformGenericToGeneric(
            makeForwardPolyMap(3, 4))
        transform3 = afwGeom.TransformGenericToGeneric(
            makeForwardPolyMap(4, 1))

        merged1 = transform3.of(transform2).of(transform1)
        merged2 = transform3.of(transform2.of(transform1))

        fromEndpoint = transform1.getFromEndpoint()
        toEndpoint = transform3.getToEndpoint()

        inPoint = fromEndpoint.pointFromData(self.makeRawPointData(2))
        assert_allclose(toEndpoint.dataFromPoint(merged1.tranForward(inPoint)),
                        toEndpoint.dataFromPoint(merged2.tranForward(inPoint)))

    def checkPersistence(self, transform):
        """Check persistence of a transform
        """
        className = type(transform).__name__
        fileName = "persisted_{}.dat".format(className)
        filePath = os.path.join(self.getTestDir(), fileName)
        transform.toFile(filePath)
        transformRoundTrip = afwGeom.readTransform(filePath)
        self.assertEqual(type(transform), type(transformRoundTrip))
        self.assertEqual(transform.getFrameSet().show(),
                         transformRoundTrip.getFrameSet().show())

        fromEndpoint = transform.getFromEndpoint()
        toEndpoint = transform.getToEndpoint()
        frameSet = transform.getFrameSet()
        nIn = frameSet.getNin()
        nOut = frameSet.getNout()

        if frameSet.hasForward():
            nPoints = 7  # arbitrary
            rawInArray = self.makeRawArrayData(nPoints, nIn)
            inArray = fromEndpoint.arrayFromData(rawInArray)
            outArray = transform.tranForward(inArray)
            outData = toEndpoint.dataFromArray(outArray)
            outArrayRoundTrip = transformRoundTrip.tranForward(inArray)
            outDataRoundTrip = toEndpoint.dataFromArray(outArrayRoundTrip)
            assert_allclose(outData, outDataRoundTrip)

        if frameSet.hasInverse():
            nPoints = 7  # arbitrary
            rawOutArray = self.makeRawArrayData(nPoints, nOut)
            outArray = toEndpoint.arrayFromData(rawOutArray)
            inArray = transform.tranInverse(outArray)
            inData = fromEndpoint.dataFromArray(inArray)
            inArrayRoundTrip = transformRoundTrip.tranInverse(outArray)
            inDataRoundTrip = fromEndpoint.dataFromArray(inArrayRoundTrip)
            assert_allclose(inData, inDataRoundTrip)
        os.remove(filePath)
