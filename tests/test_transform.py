"""
LSST Data Management System
See COPYRIGHT file at the top of the source tree.

This product includes software developed by the
LSST Project (http://www.lsst.org/).

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the LSST License Statement and
the GNU General Public License along with this program. If not,
see <http://www.lsstcorp.org/LegalNotices/>.
"""
from __future__ import absolute_import, division, print_function
import unittest

import numpy as np
from numpy.testing import assert_allclose
import astshim

import lsst.utils.tests
import lsst.afw.geom as afwGeom
from lsst.pex.exceptions import InvalidParameterError

# names of endpoints
NameList = ("Generic", "Point2", "Point3", "SpherePoint")


def makeRawArrayData(nPoints, nAxes, delta=0.123):
    return np.array([[i + j*delta for j in range(nAxes)] for i in range(nPoints)])


def makeRawPointData(nAxes, delta=0.123):
    return [i*delta for i in range(nAxes)]


def makeEndpoint(name, nAxes):
    """Make an endpoint

    @param[in] name  one of "Generic", "Point2", "Point3" or "SpherePoint"
    @param[in] nAxes  number of axes; ignored if the name is not "Generic"
    """
    endpointClassName = name + "Endpoint"
    endpointClass = getattr(afwGeom, endpointClassName)
    if name == "Generic":
        return endpointClass(nAxes)
    return endpointClass()


def makeGoodFrame(name, nAxes):
    """Return the appropriate frame for the given name and nAxes

    @param[in] name  one of "Generic", "Point2", "Point3" or "SpherePoint"
    @param[in] nAxes  number of axes; ignored if the name is not "Generic"
    """
    return makeEndpoint(name, nAxes).makeFrame()


def makeBadFrames(name):
    """Return a list of 0 or more frames that are not a valid match for the named endpoint

    @param[in] name  one of "Generic", "Point2", "Point3" or "SpherePoint"
    @param[in] nAxes  number of input axes
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

    @param[in] baseFrame  base frame
    @param[in] currFrame  current frame
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


def makeCoeffs(nIn, nOut):
    """Make an array of coefficients for astshim.PolyMap for the following equation:

    fj(x) = C0j x0 + C1j x1 + C2j x2...
    where:
    * i ranges from 0 to nIn-1
    * j ranges from 0 to nOut-1,
    * Cij = 0.001 i (j+1)
    """
    baseCoeff = 0.001
    forwardCoeffs = []
    for out_ind in range(nOut):
        coeffOffset = baseCoeff * out_ind
        for in_ind in range(nIn):
            coeff = baseCoeff * (in_ind + 1) + coeffOffset
            coeffArr = [coeff, out_ind + 1] + [1 if i == in_ind else 0 for i in range(nIn)]
            forwardCoeffs.append(coeffArr)
    return np.array(forwardCoeffs, dtype=float)


def makeTwoWayPolyMap(nIn, nOut):
    """Make an astShim.PolyMap suitable for testing

    The forward transform is as follows:
    fj(x) = C0j x0 + C1j x1 + C2j x2... where Cij = 0.001 i (j+1)

    The reverse transform is the same equation with i and j reversed
    thus it is NOT the inverse of the forward direction,
    but is something that can be easily evaluated.

    The equation is chosen for the following reasons:
    - It is well defined for any value of nIn, nOut
    - It stays small for small x, to avoid wraparound of angles for SpherePoint endpoints
    """
    forwardCoeffs = makeCoeffs(nIn, nOut)
    reverseCoeffs = makeCoeffs(nOut, nIn)
    polyMap = astshim.PolyMap(forwardCoeffs, reverseCoeffs)
    assert polyMap.getNin() == nIn
    assert polyMap.getNout() == nOut
    assert polyMap.hasForward()
    assert polyMap.hasInverse()
    return polyMap


def makeForwardPolyMap(nIn, nOut):
    """Make an astShim.PolyMap suitable for testing

    The forward transform is as follows:
    fj(x) = C0j x0 + C1j x1 + C2j x2... where Cij = 0.001 i (j+1)

    This map does not have a reverse transform.

    The equation is chosen for the following reasons:
    - It is well defined for any value of nIn, nOut
    - It stays small for small x, to avoid wraparound of angles for SpherePoint endpoints
    """
    forwardCoeffs = makeCoeffs(nIn, nOut)
    polyMap = astshim.PolyMap(forwardCoeffs, nOut, "IterInverse=0")
    assert polyMap.getNin() == nIn
    assert polyMap.getNout() == nOut
    assert polyMap.hasForward()
    assert not polyMap.hasInverse()
    return polyMap


class TransformTestCase(lsst.utils.tests.TestCase):

    def setUp(self):
        self.longMessage = True

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

    def checkTransformation(self, transform, mapping, msg=""):
        """Check tranForward and tranInverse for a transform

        @param[in] mapping  The mapping the transform should use. This mapping
                            must contain valid forward or inverse transformations,
                            but they need not match if both present. Hence the
                            mappings returned by make*PolyMap are acceptable.
        @param[in] transform  The transform to check
        @param[in] msg  Error message suffix describing test parameters
        """
        fromEndpoint = transform.getFromEndpoint()
        toEndpoint = transform.getToEndpoint()
        frameSet = transform.getFrameSet()

        nIn = mapping.getNin()
        nOut = mapping.getNout()
        self.assertEqual(nIn, fromEndpoint.getNAxes(), msg=msg)
        self.assertEqual(nOut, toEndpoint.getNAxes(), msg=msg)

        # forward transformation of one point
        rawInPoint = makeRawPointData(nIn)
        inPoint = fromEndpoint.pointFromData(rawInPoint)

        # forward transformation of an array of points
        nPoints = 7  # arbitrary
        rawInArray = makeRawArrayData(nPoints, nIn)
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
            rawOutPoint = makeRawPointData(nOut)
            outPoint = toEndpoint.pointFromData(rawOutPoint)
            rawOutArray = makeRawArrayData(nPoints, nOut)
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
        rawInPoint = makeRawPointData(fromEndpoint.getNAxes())
        inPoint = fromEndpoint.pointFromData(rawInPoint)
        rawOutPoint = makeRawPointData(toEndpoint.getNAxes())
        outPoint = toEndpoint.pointFromData(rawOutPoint)

        # transformations of arrays of points
        nPoints = 7  # arbitrary
        rawInArray = makeRawArrayData(nPoints, fromEndpoint.getNAxes())
        inArray = fromEndpoint.arrayFromData(rawInArray)
        rawOutArray = makeRawArrayData(nPoints, toEndpoint.getNAxes())
        outArray = toEndpoint.arrayFromData(rawOutArray)

        if forward.hasForward():
            self.assertEqual(forward.tranForward(inPoint),
                             inverse.tranInverse(inPoint), msg=msg)
            self.assertEqual(frameSet.tranForward(rawInPoint),
                             invFrameSet.tranInverse(rawInPoint), msg=msg)
            # Assertions must work with both lists and numpy arrays
            np.testing.assert_array_equal(forward.tranForward(inArray),
                                          inverse.tranInverse(inArray),
                                          err_msg=msg)
            np.testing.assert_array_equal(frameSet.tranForward(rawInArray),
                                          invFrameSet.tranInverse(rawInArray),
                                          err_msg=msg)

        if forward.hasInverse():
            self.assertEqual(forward.tranInverse(outPoint),
                             inverse.tranForward(outPoint), msg=msg)
            self.assertEqual(frameSet.tranInverse(rawOutPoint),
                             invFrameSet.tranForward(rawOutPoint), msg=msg)
            np.testing.assert_array_equal(forward.tranInverse(outArray),
                                          inverse.tranForward(outArray),
                                          err_msg=msg)
            np.testing.assert_array_equal(frameSet.tranInverse(rawOutArray),
                                          invFrameSet.tranForward(rawOutArray),
                                          err_msg=msg)

    def checkTransformFromMapping(self, fromName, toName):
        """Check a Transform_<fromName>_<toName> using the Mapping constructor

        fromName: one of Namelist
        toName  one of NameList
        fromAxes  number of axes in fromFrame
        toAxes  number of axes in toFrame
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

                baseFrame = makeGoodFrame(fromName, nIn)
                currFrame = makeGoodFrame(toName, nOut)
                frameSet = makeFrameSet(baseFrame, currFrame)
                self.assertEqual(frameSet.getNframe(), 4)

                # construct 0 or more frame sets that are invalid for this transform class
                for badBaseFrame in makeBadFrames(fromName):
                    badFrameSet = makeFrameSet(badBaseFrame, currFrame)
                    with self.assertRaises(InvalidParameterError):
                        transformClass(badFrameSet)
                    for badCurrFrame in makeBadFrames(toName):
                        reallyBadFrameSet = makeFrameSet(badBaseFrame, badCurrFrame)
                        with self.assertRaises(InvalidParameterError):
                            transformClass(reallyBadFrameSet)
                for badCurrFrame in makeBadFrames(toName):
                    badFrameSet = makeFrameSet(baseFrame, badCurrFrame)
                    with self.assertRaises(InvalidParameterError):
                        transformClass(badFrameSet)

                transform = transformClass(frameSet)

                desStr = "{}[{}->{}]".format(transformClassName, nIn, nOut)
                self.assertEqual("{}".format(transform), desStr)
                self.assertEqual(repr(transform), "lsst.afw.geom." + desStr)

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
                        isBasePermuted, isCurrPermuted in permFrameSetIter(frameSet):
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
                                          makeGoodFrame(fromName, nIn),
                                          makeGoodFrame(toName, nOut))

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
        frameSet = makeFrameSet(frameIn, frameOut)
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

    def testTransforms(self):
        for fromName in NameList:
            for toName in NameList:
                self.checkTransformFromMapping(fromName, toName)
                self.checkTransformFromFrameSet(fromName, toName)
                self.checkGetInverse(fromName, toName)


class MemoryTester(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
