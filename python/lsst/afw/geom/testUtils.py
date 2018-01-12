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
from __future__ import absolute_import, division, print_function

__all__ = ["BoxGrid", "makeFitsHeaderFromMetadata", "makeSipIwcToPixel", "makeSipPixelToIwc"]

from builtins import range
from builtins import object

import itertools
import math
import os
import pickle

import astshim as ast
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from astshim.test import makeForwardPolyMap, makeTwoWayPolyMap
import lsst.afw.coord  # required to use IcrsCoordEndpoint
from lsst.afw.geom.wcsUtils import getCdMatrixFromMetadata

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
            raise RuntimeError(
                "numColRow=%r; must be a sequence of two integers" % (numColRow,))
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
        end = self.pointClass(
            *[self._divList[i][indXY[i] + 1] - self.stopDelta for i in range(2)])
        return self.boxClass(beg, end)

    def __len__(self):
        return self.shape[0]*self.shape[1]

    def __iter__(self):
        """!Return an iterator over all boxes, where column varies most quickly
        """
        for row in range(self.numColRow[1]):
            for col in range(self.numColRow[0]):
                yield self[col, row]


class FrameSetInfo(object):
    """Information about a FrameSet

    Attributes
    ----------
    baseInd : `int`
        Index of base frame
    currInd : `int`
        Index of current frame
    isBaseSkyFrame : `bool`
        Is the base frame an `ast.SkyFrame`?
    isCurrSkyFrame : `bool`
        Is the current frame an `ast.SkyFrame`?
    """
    def __init__(self, frameSet):
        """Construct a FrameSetInfo

        Parameters
        ----------
        frameSet : `ast.FrameSet`
            The FrameSet about which you want information
        """
        self.baseInd = frameSet.base
        self.currInd = frameSet.current
        self.isBaseSkyFrame = frameSet.getFrame(self.baseInd).className == "SkyFrame"
        self.isCurrSkyFrame = frameSet.getFrame(self.currInd).className == "SkyFrame"


def makeFitsHeaderFromMetadata(metadata):
    """Make a FITS header string from metadata
    """
    strList = []
    for name in metadata.names(False):
        value = metadata.get(name)
        if len(name) > 8:
            raise RuntimeError("Name %r too long" % (name,))
        if isinstance(value, float):
            # keep astropy.wcs from warning about invalid format for floats
            nameValStr = "%-8s= %0.25f" % (name, value)
        else:
            nameValStr = "%-8s= %r" % (name, value)
        strList.append("%-80s" % (nameValStr,))
    return "".join(strList)


def makeSipPolyMapCoeffs(metadata, name):
    """Return a list of ast.PolyMap coefficients for the specified SIP matrix

    The returned list of coefficients for an ast.PolyMap
    that computes the following function:

        f(dxy) = dxy + sipPolynomial(dxy))
        where dxy = pixelPosition - pixelOrigin
        and sipPolynomial is a polynomial with terms `<name>n_m for x^n y^m`
            (e.g. A2_0 is the coefficient for x^2 y^0)

    Parameters
    ----------
    metadata : lsst.daf.base.PropertySet
        FITS metadata describing a WCS with the specified SIP coefficients
    name : str
        The desired SIP terms: one of A, B, AP, BP

    Returns
    -------
    list
        A list of coefficients for an ast.PolyMap that computes
        the specified SIP polynomial, including a term for out = in

    Note
    ----
    This is an internal function for use by makeSipIwcToPixel
    and makeSipPixelToIwc
    """
    outAxisDict = dict(A=1, B=2, AP=1, BP=2)
    outAxis = outAxisDict.get(name)
    if outAxis is None:
        raise RuntimeError("%s not a supported SIP name" % (name,))
    width = metadata.getAsInt("%s_ORDER" % (name,)) + 1
    found = False
    # start with a term for out = in
    coeffs = []
    if outAxis == 1:
        coeffs.append([1.0, outAxis, 1, 0])
    else:
        coeffs.append([1.0, outAxis, 0, 1])
    # add SIP distortion terms
    for xPower in range(width):
        for yPower in range(width):
            coeffName = "%s_%s_%s" % (name, xPower, yPower)
            if not metadata.exists(coeffName):
                continue
            found = True
            coeff = metadata.getAsDouble(coeffName)
            coeffs.append([coeff, outAxis, xPower, yPower])
    if not found:
        raise RuntimeError("No %s coefficients found" % (name,))
    return coeffs


def makeSipIwcToPixel(metadata):
    """Make an IWC to pixel transform with SIP distortion from FITS-WCS metadata

    This function is primarily intended for unit tests.
    IWC is intermediate world coordinates, as described in the FITS papers.

    Parameters
    ----------
    metadata : lsst.daf.base.PropertySet
        FITS metadata describing a WCS with inverse SIP coefficients

    Returns
    -------
    lsst.afw.geom.TransformPoint2ToPoint2
        Transform from IWC position to pixel position (zero-based)
        in the forward direction. The inverse direction is not defined.

    Notes
    -----

    The inverse SIP terms APn_m, BPn_m are polynomial coefficients x^n y^m
    for computing transformed x, y respectively. If we call the resulting
    polynomial inverseSipPolynomial, the returned transformation is:

        pixelPosition = pixel origin + uv + inverseSipPolynomial(uv)
        where uv = inverseCdMatrix * iwcPosition
    """
    crpix = (metadata.get("CRPIX1") - 1, metadata.get("CRPIX2") - 1)
    pixelRelativeToAbsoluteMap = ast.ShiftMap(crpix)
    cdMatrix = getCdMatrixFromMetadata(metadata)
    cdMatrixMap = ast.MatrixMap(cdMatrix.copy())
    coeffList = makeSipPolyMapCoeffs(metadata, "AP") + makeSipPolyMapCoeffs(metadata, "BP")
    coeffArr = np.array(coeffList, dtype=float)
    sipPolyMap = ast.PolyMap(coeffArr, 2, "IterInverse=0")

    iwcToPixelMap = cdMatrixMap.getInverse().then(sipPolyMap).then(pixelRelativeToAbsoluteMap)
    return afwGeom.TransformPoint2ToPoint2(iwcToPixelMap)


def makeSipPixelToIwc(metadata):
    """Make a pixel to IWC transform with SIP distortion from FITS-WCS metadata

    This function is primarily intended for unit tests.
    IWC is intermediate world coordinates, as described in the FITS papers.

    Parameters
    ----------
    metadata : lsst.daf.base.PropertySet
        FITS metadata describing a WCS with forward SIP coefficients

    Returns
    -------
    lsst.afw.geom.TransformPoint2ToPoint2
        Transform from pixel position (zero-based) to IWC position
        in the forward direction. The inverse direction is not defined.

    Notes
    -----

    The forward SIP terms An_m, Bn_m are polynomial coefficients x^n y^m
    for computing transformed x, y respectively. If we call the resulting
    polynomial sipPolynomial, the returned transformation is:

        iwcPosition = cdMatrix * (dxy + sipPolynomial(dxy))
        where dxy = pixelPosition - pixelOrigin
    """
    crpix = (metadata.get("CRPIX1") - 1, metadata.get("CRPIX2") - 1)
    pixelAbsoluteToRelativeMap = ast.ShiftMap(crpix).getInverse()
    cdMatrix = getCdMatrixFromMetadata(metadata)
    cdMatrixMap = ast.MatrixMap(cdMatrix.copy())
    coeffList = makeSipPolyMapCoeffs(metadata, "A") + makeSipPolyMapCoeffs(metadata, "B")
    coeffArr = np.array(coeffList, dtype=float)
    sipPolyMap = ast.PolyMap(coeffArr, 2, "IterInverse=0")
    pixelToIwcMap = pixelAbsoluteToRelativeMap.then(sipPolyMap).then(cdMatrixMap)
    return afwGeom.TransformPoint2ToPoint2(pixelToIwcMap)


class PermutedFrameSet(object):
    """A FrameSet with base or current frame possibly permuted, with associated
    information

    Only two-axis frames will be permuted.

    Attributes
    ----------
    frameSet : `ast.FrameSet`
        The FrameSet that may be permuted. A local copy is made.
    isBaseSkyFrame : `bool`
        Is the base frame an `ast.SkyFrame`?
    isCurrSkyFrame : `bool`
        Is the current frame an `ast.SkyFrame`?
    isBasePermuted : `bool`
        Are the base frame axes permuted?
    isCurrPermuted : `bool`
        Are the current frame axes permuted?
    """
    def __init__(self, frameSet, permuteBase, permuteCurr):
        """Construct a PermutedFrameSet

        Make a copy of a FrameSet and permute the base and/or current frames if
        requested

        Parameters
        ----------
        frameSet : `ast.FrameSet`
            The FrameSet you wish to permute. A deep copy is made.
        permuteBase : `bool`
            Permute the base frame's axes?
        permuteCurr : `bool`
            Permute the current frame's axes?

        Raises
        ------
        `RuntimeError`
            If you try to permute a frame that does not have 2 axes
        """
        self.frameSet = frameSet.copy()
        fsInfo = FrameSetInfo(self.frameSet)
        self.isBaseSkyFrame = fsInfo.isBaseSkyFrame
        self.isCurrSkyFrame = fsInfo.isCurrSkyFrame
        if permuteBase:
            baseNAxes = self.frameSet.getFrame(fsInfo.baseInd).nAxes
            if baseNAxes != 2:
                raise RuntimeError("Base frame has {} axes; 2 required to permute".format(baseNAxes))
            self.frameSet.current = fsInfo.baseInd
            self.frameSet.permAxes([2, 1])
            self.frameSet.current = fsInfo.currInd
        if permuteCurr:
            currNAxes = self.frameSet.getFrame(fsInfo.currInd).nAxes
            if currNAxes != 2:
                raise RuntimeError("Current frame has {} axes; 2 required to permute".format(currNAxes))
            assert self.frameSet.getFrame(fsInfo.currInd).nAxes == 2
            self.frameSet.permAxes([2, 1])
        self.isBasePermuted = permuteBase
        self.isCurrPermuted = permuteCurr


class TransformTestBaseClass(lsst.utils.tests.TestCase):
    """Base class for unit tests of Transform<X>To<Y>

    Subclasses must call `TransformTestBaseClass.setUp(self)`
    if they provide their own version.

    If a package other than afw uses this class then it must
    override the `getTestDir` method to avoid writing into
    afw's test directory.
    """

    def getTestDir(self):
        """Return a directory where temporary test files can be written

        The default implementation returns the test directory of the `afw`
        package.

        If this class is used by a test in a package other than `afw`
        then the subclass must override this method.
        """
        return os.path.join(lsst.utils.getPackageDir("afw"), "tests")

    def setUp(self):
        """Set up a test

        Subclasses should call this method if they override setUp.
        """
        # tell unittest to use the msg argument of asserts as a supplement
        # to the error message, rather than as the whole error message
        self.longMessage = True

        # list of endpoint class name prefixes; the full name is prefix + "Endpoint"
        self.endpointPrefixes = ("Generic", "Point2", "IcrsCoord")

        # GoodNAxes is dict of endpoint class name prefix:
        #    tuple containing 0 or more valid numbers of axes
        self.goodNAxes = {
            "Generic": (1, 2, 3, 4),  # all numbers of axes are valid for GenericEndpoint
            "Point2": (2,),
            "IcrsCoord": (2,),
        }

        # BadAxes is dict of endpoint class name prefix:
        #    tuple containing 0 or more invalid numbers of axes
        self.badNAxes = {
            "Generic": (),  # all numbers of axes are valid for GenericEndpoint
            "Point2": (1, 3, 4),
            "IcrsCoord": (1, 3, 4),
        }

        # Dict of frame index: identity name for frames created by makeFrameSet
        self.frameIdentDict = {
            1: "baseFrame",
            2: "frame2",
            3: "frame3",
            4: "currFrame",
        }

    @staticmethod
    def makeRawArrayData(nPoints, nAxes, delta=0.123):
        """Make an array of generic point data

        The data will be suitable for spherical points

        Parameters
        ----------
        nPoints : `int`
            Number of points in the array
        nAxes : `int`
            Number of axes in the point

        Returns
        -------
        np.array of floats with shape (nAxes, nPoints)
            The values are as follows; if nAxes != 2:
                The first point has values `[0, delta, 2*delta, ..., (nAxes-1)*delta]`
                The Nth point has those values + N
            if nAxes == 2 then the data is scaled so that the max value of axis 1
                is a bit less than pi/2
        """
        delta = 0.123
        # oneAxis = [0, 1, 2, ...nPoints-1]
        oneAxis = np.arange(nPoints, dtype=float)  # [0, 1, 2...]
        # rawData = [oneAxis, oneAxis + delta, oneAxis + 2 delta, ...]
        rawData = np.array([j * delta + oneAxis for j in range(nAxes)], dtype=float)
        if nAxes == 2:
            # scale rawData so that max value of 2nd axis is a bit less than pi/2,
            # thus making the data safe for IcrsCoord
            maxLatitude = np.max(rawData[1])
            rawData *= math.pi * 0.4999 / maxLatitude
        return rawData

    @staticmethod
    def makeRawPointData(nAxes, delta=0.123):
        """Make one generic point

        Parameters
        ----------
        nAxes : `int`
            Number of axes in the point
        delta : `float`
            Increment between axis values

        Returns
        -------
        A list of `nAxes` floats with values `[0, delta, ..., (nAxes-1)*delta]
        """
        return [i*delta for i in range(nAxes)]

    @staticmethod
    def makeEndpoint(name, nAxes=None):
        """Make an endpoint

        Parameters
        ----------
        name : `str`
            Endpoint class name prefix; the full class name is name + "Endpoint"
        nAxes : `int` or `None`, optional
            number of axes; an int is required if `name` == "Generic";
            otherwise ignored

        Returns
        -------
        subclass of `lsst.afw.geom.BaseEndpoint`
            The constructed endpoint

        Raises
        ------
        `TypeError`
            If `name` == "Generic" and `nAxes` is None or <= 0
        """
        EndpointClassName = name + "Endpoint"
        EndpointClass = getattr(afwGeom, EndpointClassName)
        if name == "Generic":
            if nAxes is None:
                raise TypeError("nAxes must be an integer for GenericEndpoint")
            return EndpointClass(nAxes)
        return EndpointClass()

    @classmethod
    def makeGoodFrame(cls, name, nAxes=None):
        """Return the appropriate frame for the given name and nAxes

        Parameters
        ----------
        name : `str`
            Endpoint class name prefix; the full class name is name + "Endpoint"
        nAxes : `int` or `None`, optional
            number of axes; an int is required if `name` == "Generic";
            otherwise ignored

        Returns
        -------
        `ast.Frame`
            The constructed frame

        Raises
        ------
        `TypeError`
            If `name` == "Generic" and `nAxes` is `None` or <= 0
        """
        return cls.makeEndpoint(name, nAxes).makeFrame()

    @staticmethod
    def makeBadFrames(name):
        """Return a list of 0 or more frames that are not a valid match for the
        named endpoint

        Parameters
        ----------
        name : `str`
            Endpoint class name prefix; the full class name is name + "Endpoint"

        Returns
        -------
        Collection of `ast.Frame`
            A collection of 0 or more frames
        """
        return {
            "Generic": [],
            "Point2": [
                ast.SkyFrame(),
                ast.Frame(1),
                ast.Frame(3),
            ],
            "IcrsCoord": [
                ast.Frame(1),
                ast.Frame(2),
                ast.Frame(3),
            ],
        }[name]

    def makeFrameSet(self, baseFrame, currFrame):
        """Make a FrameSet

        The FrameSet will contain 4 frames and three transforms connecting them.
        The idenity of each frame is provided by self.frameIdentDict

        Frame       Index   Mapping from this frame to the next
        `baseFrame`   1     `ast.UnitMap(nIn)`
        Frame(nIn)    2     `polyMap`
        Frame(nOut)   3     `ast.UnitMap(nOut)`
        `currFrame`   4

        where:
        - `nIn` = `baseFrame.nAxes`
        - `nOut` = `currFrame.nAxes`
        - `polyMap` = `makeTwoWayPolyMap(nIn, nOut)`

        Return
        ------
        `ast.FrameSet`
            The FrameSet as described above

        Parameters
        ----------
        baseFrame : `ast.Frame`
            base frame
        currFrame : `ast.Frame`
            current frame
        """
        nIn = baseFrame.nAxes
        nOut = currFrame.nAxes
        polyMap = makeTwoWayPolyMap(nIn, nOut)

        # The only way to set the Ident of a frame in a FrameSet is to set it in advance,
        # and I don't want to modify the inputs, so replace the input frames with copies
        baseFrame = baseFrame.copy()
        baseFrame.ident = self.frameIdentDict[1]
        currFrame = currFrame.copy()
        currFrame.ident = self.frameIdentDict[4]

        frameSet = ast.FrameSet(baseFrame)
        frame2 = ast.Frame(nIn)
        frame2.ident = self.frameIdentDict[2]
        frameSet.addFrame(ast.FrameSet.CURRENT, ast.UnitMap(nIn), frame2)
        frame3 = ast.Frame(nOut)
        frame3.ident = self.frameIdentDict[3]
        frameSet.addFrame(ast.FrameSet.CURRENT, polyMap, frame3)
        frameSet.addFrame(ast.FrameSet.CURRENT, ast.UnitMap(nOut), currFrame)
        return frameSet

    @staticmethod
    def permuteFrameSetIter(frameSet):
        """Iterator over 0 or more frameSets with SkyFrames axes permuted

        Only base and current SkyFrames are permuted. If neither the base nor
        the current frame is a SkyFrame then no frames are returned.

        Returns
        -------
        iterator over `PermutedFrameSet`
        """

        fsInfo = FrameSetInfo(frameSet)
        if not (fsInfo.isBaseSkyFrame or fsInfo.isCurrSkyFrame):
            return

        permuteBaseList = [False, True] if fsInfo.isBaseSkyFrame else [False]
        permuteCurrList = [False, True] if fsInfo.isCurrSkyFrame else [False]
        for permuteBase in permuteBaseList:
            for permuteCurr in permuteCurrList:
                yield PermutedFrameSet(frameSet, permuteBase, permuteCurr)

    @staticmethod
    def makeJacobian(nIn, nOut, inPoint):
        """Make a Jacobian matrix for the equation described by
        `makeTwoWayPolyMap`.

        Parameters
        ----------
        nIn, nOut : `int`
            the dimensions of the input and output data; see makeTwoWayPolyMap
        inPoint : `numpy.ndarray`
            an array of size `nIn` representing the point at which the Jacobian
            is measured

        Returns
        -------
        J : `numpy.ndarray`
            an `nOut` x `nIn` array of first derivatives
        """
        basePolyMapCoeff = 0.001  # see makeTwoWayPolyMap
        baseCoeff = 2.0 * basePolyMapCoeff
        coeffs = np.empty((nOut, nIn))
        for iOut in range(nOut):
            coeffOffset = baseCoeff * iOut
            for iIn in range(nIn):
                coeffs[iOut, iIn] = baseCoeff * (iIn + 1) + coeffOffset
                coeffs[iOut, iIn] *= inPoint[iIn]
        assert coeffs.ndim == 2
        # Avoid spurious errors when comparing to a simplified array
        assert coeffs.shape == (nOut, nIn)
        return coeffs

    def checkTransformation(self, transform, mapping, msg=""):
        """Check applyForward and applyInverse for a transform

        Parameters
        ----------
        transform : `lsst.afw.geom.Transform`
            The transform to check
        mapping : `ast.Mapping`
            The mapping the transform should use. This mapping
            must contain valid forward or inverse transformations,
            but they need not match if both present. Hence the
            mappings returned by make*PolyMap are acceptable.
        msg : `str`
            Error message suffix describing test parameters
        """
        fromEndpoint = transform.fromEndpoint
        toEndpoint = transform.toEndpoint
        frameSet = transform.getFrameSet()

        nIn = mapping.nIn
        nOut = mapping.nOut
        self.assertEqual(nIn, fromEndpoint.nAxes, msg=msg)
        self.assertEqual(nOut, toEndpoint.nAxes, msg=msg)

        # forward transformation of one point
        rawInPoint = self.makeRawPointData(nIn)
        inPoint = fromEndpoint.pointFromData(rawInPoint)

        # forward transformation of an array of points
        nPoints = 7  # arbitrary
        rawInArray = self.makeRawArrayData(nPoints, nIn)
        inArray = fromEndpoint.arrayFromData(rawInArray)

        if mapping.hasForward:
            self.assertTrue(transform.hasForward)
            outPoint = transform.applyForward(inPoint)
            rawOutPoint = toEndpoint.dataFromPoint(outPoint)
            assert_allclose(rawOutPoint, mapping.applyForward(rawInPoint), err_msg=msg)
            assert_allclose(rawOutPoint, frameSet.applyForward(rawInPoint), err_msg=msg)

            outArray = transform.applyForward(inArray)
            rawOutArray = toEndpoint.dataFromArray(outArray)
            self.assertFloatsAlmostEqual(rawOutArray, mapping.applyForward(rawInArray), msg=msg)
            self.assertFloatsAlmostEqual(rawOutArray, frameSet.applyForward(rawInArray), msg=msg)
        else:
            # Need outPoint, but don't need it to be consistent with inPoint
            rawOutPoint = self.makeRawPointData(nOut)
            outPoint = toEndpoint.pointFromData(rawOutPoint)
            rawOutArray = self.makeRawArrayData(nPoints, nOut)
            outArray = toEndpoint.arrayFromData(rawOutArray)

            self.assertFalse(transform.hasForward)

        if mapping.hasInverse:
            self.assertTrue(transform.hasInverse)
            # inverse transformation of one point;
            # remember that the inverse need not give the original values
            # (see the description of the `mapping` parameter)
            inversePoint = transform.applyInverse(outPoint)
            rawInversePoint = fromEndpoint.dataFromPoint(inversePoint)
            assert_allclose(rawInversePoint, mapping.applyInverse(rawOutPoint), err_msg=msg)
            assert_allclose(rawInversePoint, frameSet.applyInverse(rawOutPoint), err_msg=msg)

            # inverse transformation of an array of points;
            # remember that the inverse will not give the original values
            # (see the description of the `mapping` parameter)
            inverseArray = transform.applyInverse(outArray)
            rawInverseArray = fromEndpoint.dataFromArray(inverseArray)
            self.assertFloatsAlmostEqual(rawInverseArray, mapping.applyInverse(rawOutArray), msg=msg)
            self.assertFloatsAlmostEqual(rawInverseArray, frameSet.applyInverse(rawOutArray), msg=msg)
        else:
            self.assertFalse(transform.hasInverse)

    def checkInverseTransformation(self, forward, inverse, msg=""):
        """Check that two Transforms are each others' inverses.

        Parameters
        ----------
        forward : `lsst.afw.geom.Transform`
            the reference Transform to test
        inverse : `lsst.afw.geom.Transform`
            the transform that should be the inverse of `forward`
        msg : `str`
            error message suffix describing test parameters
        """
        fromEndpoint = forward.fromEndpoint
        toEndpoint = forward.toEndpoint
        frameSet = forward.getFrameSet()
        invFrameSet = inverse.getFrameSet()

        # properties
        self.assertEqual(forward.fromEndpoint,
                         inverse.toEndpoint, msg=msg)
        self.assertEqual(forward.toEndpoint,
                         inverse.fromEndpoint, msg=msg)
        self.assertEqual(forward.hasForward, inverse.hasInverse, msg=msg)
        self.assertEqual(forward.hasInverse, inverse.hasForward, msg=msg)

        # transformations of one point
        # we don't care about whether the transformation itself is correct
        # (see checkTransformation), so inPoint/outPoint need not be related
        rawInPoint = self.makeRawPointData(fromEndpoint.nAxes)
        inPoint = fromEndpoint.pointFromData(rawInPoint)
        rawOutPoint = self.makeRawPointData(toEndpoint.nAxes)
        outPoint = toEndpoint.pointFromData(rawOutPoint)

        # transformations of arrays of points
        nPoints = 7  # arbitrary
        rawInArray = self.makeRawArrayData(nPoints, fromEndpoint.nAxes)
        inArray = fromEndpoint.arrayFromData(rawInArray)
        rawOutArray = self.makeRawArrayData(nPoints, toEndpoint.nAxes)
        outArray = toEndpoint.arrayFromData(rawOutArray)

        if forward.hasForward:
            self.assertEqual(forward.applyForward(inPoint),
                             inverse.applyInverse(inPoint), msg=msg)
            self.assertEqual(frameSet.applyForward(rawInPoint),
                             invFrameSet.applyInverse(rawInPoint), msg=msg)
            # Assertions must work with both lists and numpy arrays
            assert_array_equal(forward.applyForward(inArray),
                               inverse.applyInverse(inArray),
                               err_msg=msg)
            assert_array_equal(frameSet.applyForward(rawInArray),
                               invFrameSet.applyInverse(rawInArray),
                               err_msg=msg)

        if forward.hasInverse:
            self.assertEqual(forward.applyInverse(outPoint),
                             inverse.applyForward(outPoint), msg=msg)
            self.assertEqual(frameSet.applyInverse(rawOutPoint),
                             invFrameSet.applyForward(rawOutPoint), msg=msg)
            assert_array_equal(forward.applyInverse(outArray),
                               inverse.applyForward(outArray),
                               err_msg=msg)
            assert_array_equal(frameSet.applyInverse(rawOutArray),
                               invFrameSet.applyForward(rawOutArray),
                               err_msg=msg)

    def checkTransformFromMapping(self, fromName, toName):
        """Check Transform_<fromName>_<toName> using the Mapping constructor

        Parameters
        ----------
        fromName, toName : `str`
            Endpoint name prefix for "from" and "to" endpoints, respectively,
            e.g. "Point2" for `lsst.afw.geom.Point2Endpoint`
        fromAxes, toAxes : `int`
            number of axes in fromFrame and toFrame, respectively
        """
        transformClassName = "Transform{}To{}".format(fromName, toName)
        TransformClass = getattr(afwGeom, transformClassName)
        baseMsg = "TransformClass={}".format(TransformClass.__name__)

        # check valid numbers of inputs and outputs
        for nIn, nOut in itertools.product(self.goodNAxes[fromName],
                                           self.goodNAxes[toName]):
            msg = "{}, nIn={}, nOut={}".format(baseMsg, nIn, nOut)
            polyMap = makeTwoWayPolyMap(nIn, nOut)
            transform = TransformClass(polyMap)

            # desired output from `str(transform)`
            desStr = "{}[{}->{}]".format(transformClassName, nIn, nOut)
            self.assertEqual("{}".format(transform), desStr)
            self.assertEqual(repr(transform), "lsst.afw.geom." + desStr)

            self.checkTransformation(transform, polyMap, msg=msg)

            # Forward transform but no inverse
            polyMap = makeForwardPolyMap(nIn, nOut)
            transform = TransformClass(polyMap)
            self.checkTransformation(transform, polyMap, msg=msg)

            # Inverse transform but no forward
            polyMap = makeForwardPolyMap(nOut, nIn).getInverse()
            transform = TransformClass(polyMap)
            self.checkTransformation(transform, polyMap, msg=msg)

        # check invalid # of output against valid # of inputs
        for nIn, badNOut in itertools.product(self.goodNAxes[fromName],
                                              self.badNAxes[toName]):
            badPolyMap = makeTwoWayPolyMap(nIn, badNOut)
            msg = "{}, nIn={}, badNOut={}".format(baseMsg, nIn, badNOut)
            with self.assertRaises(InvalidParameterError, msg=msg):
                TransformClass(badPolyMap)

        # check invalid # of inputs against valid and invalid # of outputs
        for badNIn, nOut in itertools.product(self.badNAxes[fromName],
                                              self.goodNAxes[toName] + self.badNAxes[toName]):
                badPolyMap = makeTwoWayPolyMap(badNIn, nOut)
                msg = "{}, badNIn={}, nOut={}".format(baseMsg, nIn, nOut)
                with self.assertRaises(InvalidParameterError, msg=msg):
                    TransformClass(badPolyMap)

    def checkTransformFromFrameSet(self, fromName, toName):
        """Check Transform_<fromName>_<toName> using the FrameSet constructor

        Parameters
        ----------
        fromName, toName : `str`
            Endpoint name prefix for "from" and "to" endpoints, respectively,
            e.g. "Point2" for `lsst.afw.geom.Point2Endpoint`
        """
        transformClassName = "Transform{}To{}".format(fromName, toName)
        TransformClass = getattr(afwGeom, transformClassName)
        baseMsg = "TransformClass={}".format(TransformClass.__name__)
        for nIn, nOut in itertools.product(self.goodNAxes[fromName],
                                           self.goodNAxes[toName]):
            msg = "{}, nIn={}, nOut={}".format(baseMsg, nIn, nOut)

            baseFrame = self.makeGoodFrame(fromName, nIn)
            currFrame = self.makeGoodFrame(toName, nOut)
            frameSet = self.makeFrameSet(baseFrame, currFrame)
            self.assertEqual(frameSet.nFrame, 4)

            # construct 0 or more frame sets that are invalid for this transform class
            for badBaseFrame in self.makeBadFrames(fromName):
                badFrameSet = self.makeFrameSet(badBaseFrame, currFrame)
                with self.assertRaises(InvalidParameterError):
                    TransformClass(badFrameSet)
                for badCurrFrame in self.makeBadFrames(toName):
                    reallyBadFrameSet = self.makeFrameSet(badBaseFrame, badCurrFrame)
                    with self.assertRaises(InvalidParameterError):
                        TransformClass(reallyBadFrameSet)
            for badCurrFrame in self.makeBadFrames(toName):
                badFrameSet = self.makeFrameSet(baseFrame, badCurrFrame)
                with self.assertRaises(InvalidParameterError):
                    TransformClass(badFrameSet)

            transform = TransformClass(frameSet)

            desStr = "{}[{}->{}]".format(transformClassName, nIn, nOut)
            self.assertEqual("{}".format(transform), desStr)
            self.assertEqual(repr(transform), "lsst.afw.geom." + desStr)

            self.checkPersistence(transform)

            frameSetCopy = transform.getFrameSet()
            transformCopy = TransformClass(frameSetCopy)
            self.assertEqual(type(transform), type(transformCopy))
            self.assertEqual(transform.getFrameSet(), transformCopy.getFrameSet())

            desNFrame = 4  # desired number of frames
            self.assertEqual(frameSet.nFrame, desNFrame)
            self.assertEqual(frameSetCopy.nFrame, desNFrame)
            for frameInd in range(1, 1 + desNFrame):
                self.assertEqual(frameSet.getFrame(frameInd).ident,
                                 self.frameIdentDict[frameInd])

            polyMap = makeTwoWayPolyMap(nIn, nOut)

            self.checkTransformation(transform, mapping=polyMap, msg=msg)

            # If the base and/or current frame of frameSet is a SkyFrame,
            # try permuting that frame (in place, so the connected mappings are
            # correctly updated). The Transform constructor should undo the permutation,
            # (via IcrsCoordEndpoint.normalizeFrame) in its internal copy of frameSet,
            # forcing the axes of the SkyFrame into standard (longitude, latitude) order
            for permutedFS in self.permuteFrameSetIter(frameSet):
                if permutedFS.isBaseSkyFrame:
                    baseFrame = permutedFS.frameSet.getFrame(ast.FrameSet.BASE)
                    # desired base longitude axis
                    desBaseLonAxis = 2 if permutedFS.isBasePermuted else 1
                    self.assertEqual(baseFrame.lonAxis, desBaseLonAxis)
                if permutedFS.isCurrSkyFrame:
                    currFrame = permutedFS.frameSet.getFrame(ast.FrameSet.CURRENT)
                    # desired current base longitude axis
                    desCurrLonAxis = 2 if permutedFS.isCurrPermuted else 1
                    self.assertEqual(currFrame.lonAxis, desCurrLonAxis)

                permTransform = TransformClass(permutedFS.frameSet)
                # If the base and/or current frame is a SkyFrame then make sure the frame
                # in the *Transform* has axes in standard (longitude, latitude) order
                unpermFrameSet = permTransform.getFrameSet()
                if permutedFS.isBaseSkyFrame:
                    self.assertEqual(unpermFrameSet.getFrame(ast.FrameSet.BASE).lonAxis, 1)
                if permutedFS.isCurrSkyFrame:
                    self.assertEqual(unpermFrameSet.getFrame(ast.FrameSet.CURRENT).lonAxis, 1)

                self.checkTransformation(permTransform, mapping=polyMap, msg=msg)

    def checkGetInverse(self, fromName, toName):
        """Test Transform<fromName>To<toName>.getInverse

        Parameters
        ----------
        fromName, toName : `str`
            Endpoint name prefix for "from" and "to" endpoints, respectively,
            e.g. "Point2" for `lsst.afw.geom.Point2Endpoint`
        """
        transformClassName = "Transform{}To{}".format(fromName, toName)
        TransformClass = getattr(afwGeom, transformClassName)
        baseMsg = "TransformClass={}".format(TransformClass.__name__)
        for nIn, nOut in itertools.product(self.goodNAxes[fromName],
                                           self.goodNAxes[toName]):
            msg = "{}, nIn={}, nOut={}".format(baseMsg, nIn, nOut)
            self.checkInverseMapping(
                TransformClass,
                makeTwoWayPolyMap(nIn, nOut),
                "{}, Map={}".format(msg, "TwoWay"))
            self.checkInverseMapping(
                TransformClass,
                makeForwardPolyMap(nIn, nOut),
                "{}, Map={}".format(msg, "Forward"))
            self.checkInverseMapping(
                TransformClass,
                makeForwardPolyMap(nOut, nIn).getInverse(),
                "{}, Map={}".format(msg, "Inverse"))

            self.checkInverseFrameSet(TransformClass,
                                      self.makeGoodFrame(fromName, nIn),
                                      self.makeGoodFrame(toName, nOut))

    def checkInverseMapping(self, TransformClass, mapping, msg):
        """Test Transform<fromName>To<toName>.getInverse for a specific mapping.

        Parameters
        ----------
        TransformClass : `type`
            The class of transform to test, such as TransformPoint2ToIcrsCoord
        mapping : `ast.Mapping`
            The mapping to use for the transform
        msg : `str`
            Error message suffix
        """
        transform = TransformClass(mapping)
        inverse = transform.getInverse()
        inverseInverse = inverse.getInverse()

        self.checkInverseTransformation(transform, inverse, msg=msg)
        self.checkInverseTransformation(inverse, inverseInverse, msg=msg)
        self.checkTransformation(inverseInverse, mapping, msg=msg)

    def checkInverseFrameSet(self, TransformClass, frameIn, frameOut):
        """Test whether inverting a Transform preserves all information
           in its FrameSet.

        Parameters
        ----------
        TransformClass : `type`
            the transform to test
        frameIn, frameOut : `ast.Frame`
            the frames to between which `TransformClass` shall convert. Must be
            compatible with `TransformClass`.
        """
        desNFrame = 4  # desired number of frames
        frameSet = self.makeFrameSet(frameIn, frameOut)
        self.assertEqual(frameSet.nFrame, desNFrame)

        baseMsg = "TransformClass={}, nIn={}, nOut={}".format(
            TransformClass.__name__, frameIn.nAxes, frameOut.nAxes)
        transform = TransformClass(frameSet)
        forwardFrames = transform.getFrameSet()
        self.assertFalse(forwardFrames.isInverted)
        self.assertEqual(forwardFrames.base, 1)
        self.assertEqual(forwardFrames.current, desNFrame)

        self.assertEqual(forwardFrames.nFrame, desNFrame, msg=baseMsg)
        for frameInd in range(1, 1 + desNFrame):
            self.assertEqual(forwardFrames.getFrame(frameInd).ident,
                             self.frameIdentDict[frameInd], msg=baseMsg)

        reverseFrames = transform.getInverse().getFrameSet()
        self.assertTrue(reverseFrames.isInverted)
        self.assertEqual(reverseFrames.base, desNFrame)
        self.assertEqual(reverseFrames.current, 1)
        self.assertEqual(reverseFrames.nFrame, desNFrame, msg=baseMsg)
        for frameInd in range(1, 1 + desNFrame):
            self.assertEqual(reverseFrames.getFrame(frameInd).ident,
                             self.frameIdentDict[frameInd], msg=baseMsg)

    def checkGetJacobian(self, fromName, toName):
        """Test Transform<fromName>To<toName>.getJacobian

        Parameters
        ----------
        fromName, toName : `str`
            Endpoint name prefix for "from" and "to" endpoints, respectively,
            e.g. "Point2" for `lsst.afw.geom.Point2Endpoint`
        """
        transformClassName = "Transform{}To{}".format(fromName, toName)
        TransformClass = getattr(afwGeom, transformClassName)
        baseMsg = "TransformClass={}".format(TransformClass.__name__)
        for nIn, nOut in itertools.product(self.goodNAxes[fromName],
                                           self.goodNAxes[toName]):
            msg = "{}, nIn={}, nOut={}".format(baseMsg, nIn, nOut)
            polyMap = makeForwardPolyMap(nIn, nOut)
            transform = TransformClass(polyMap)
            fromEndpoint = transform.fromEndpoint

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

    def checkThen(self, fromName, midName, toName):
        """Test Transform<fromName>To<midName>.then(Transform<midName>To<toName>)

        Parameters
        ----------
        fromName : `str`
            the prefix of the starting endpoint (e.g., "Point2" for a
            Point2Endpoint) for the final, concatenated Transform
        midName : `str`
            the prefix for the shared endpoint where two Transforms will be
            concatenated
        toName : `str`
            the prefix of the ending endpoint for the final, concatenated
            Transform
        """
        TransformClass1 = getattr(afwGeom,
                                  "Transform{}To{}".format(fromName, midName))
        TransformClass2 = getattr(afwGeom,
                                  "Transform{}To{}".format(midName, toName))
        baseMsg = "{}.then({})".format(TransformClass1.__name__,
                                       TransformClass2.__name__)
        for nIn, nMid, nOut in itertools.product(self.goodNAxes[fromName],
                                                 self.goodNAxes[midName],
                                                 self.goodNAxes[toName]):
            msg = "{}, nIn={}, nMid={}, nOut={}".format(
                baseMsg, nIn, nMid, nOut)
            polyMap1 = makeTwoWayPolyMap(nIn, nMid)
            transform1 = TransformClass1(polyMap1)
            polyMap2 = makeTwoWayPolyMap(nMid, nOut)
            transform2 = TransformClass2(polyMap2)
            transform = transform1.then(transform2)

            fromEndpoint = transform1.fromEndpoint
            toEndpoint = transform2.toEndpoint

            inPoint = fromEndpoint.pointFromData(self.makeRawPointData(nIn))
            outPointMerged = transform.applyForward(inPoint)
            outPointSeparate = transform2.applyForward(
                transform1.applyForward(inPoint))
            assert_allclose(toEndpoint.dataFromPoint(outPointMerged),
                            toEndpoint.dataFromPoint(outPointSeparate),
                            err_msg=msg)

            outPoint = toEndpoint.pointFromData(self.makeRawPointData(nOut))
            inPointMerged = transform.applyInverse(outPoint)
            inPointSeparate = transform1.applyInverse(
                transform2.applyInverse(outPoint))
            assert_allclose(
                fromEndpoint.dataFromPoint(inPointMerged),
                fromEndpoint.dataFromPoint(inPointSeparate),
                err_msg=msg)

        # Mismatched number of axes should fail
        if midName == "Generic":
            nIn = self.goodNAxes[fromName][0]
            nOut = self.goodNAxes[toName][0]
            polyMap = makeTwoWayPolyMap(nIn, 3)
            transform1 = TransformClass1(polyMap)
            polyMap = makeTwoWayPolyMap(2, nOut)
            transform2 = TransformClass2(polyMap)
            with self.assertRaises(InvalidParameterError):
                transform = transform1.then(transform2)

        # Mismatched types of endpoints should fail
        if fromName != midName:
            # Use TransformClass1 for both args to keep test logic simple
            outName = midName
            joinNAxes = set(self.goodNAxes[fromName]).intersection(
                self.goodNAxes[outName])

            for nIn, nMid, nOut in itertools.product(self.goodNAxes[fromName],
                                                     joinNAxes,
                                                     self.goodNAxes[outName]):
                polyMap = makeTwoWayPolyMap(nIn, nMid)
                transform1 = TransformClass1(polyMap)
                polyMap = makeTwoWayPolyMap(nMid, nOut)
                transform2 = TransformClass1(polyMap)
                with self.assertRaises(InvalidParameterError):
                    transform = transform1.then(transform2)

    def assertTransformsEqual(self, transform1, transform2):
        """Assert that two transforms are equal"""
        self.assertEqual(type(transform1), type(transform2))
        self.assertEqual(transform1.fromEndpoint, transform2.fromEndpoint)
        self.assertEqual(transform1.toEndpoint, transform2.toEndpoint)
        self.assertEqual(transform1.getFrameSet(), transform2.getFrameSet())

        fromEndpoint = transform1.fromEndpoint
        toEndpoint = transform1.toEndpoint
        frameSet = transform1.getFrameSet()
        nIn = frameSet.nIn
        nOut = frameSet.nOut

        if frameSet.hasForward:
            nPoints = 7  # arbitrary
            rawInArray = self.makeRawArrayData(nPoints, nIn)
            inArray = fromEndpoint.arrayFromData(rawInArray)
            outArray = transform1.applyForward(inArray)
            outData = toEndpoint.dataFromArray(outArray)
            outArrayRoundTrip = transform2.applyForward(inArray)
            outDataRoundTrip = toEndpoint.dataFromArray(outArrayRoundTrip)
            assert_allclose(outData, outDataRoundTrip)

        if frameSet.hasInverse:
            nPoints = 7  # arbitrary
            rawOutArray = self.makeRawArrayData(nPoints, nOut)
            outArray = toEndpoint.arrayFromData(rawOutArray)
            inArray = transform1.applyInverse(outArray)
            inData = fromEndpoint.dataFromArray(inArray)
            inArrayRoundTrip = transform2.applyInverse(outArray)
            inDataRoundTrip = fromEndpoint.dataFromArray(inArrayRoundTrip)
            assert_allclose(inData, inDataRoundTrip)

    def checkPersistence(self, transform):
        """Check persistence of a transform
        """
        className = type(transform).__name__

        # check writeString and readString
        transformStr = transform.writeString()
        serialVersion, serialClassName, serialRest = transformStr.split(" ", 2)
        self.assertEqual(int(serialVersion), 1)
        self.assertEqual(serialClassName, className)
        badStr1 = " ".join(["2", serialClassName, serialRest])
        with self.assertRaises(lsst.pex.exceptions.InvalidParameterError):
            transform.readString(badStr1)
        badClassName = "x" + serialClassName
        badStr2 = " ".join(["1", badClassName, serialRest])
        with self.assertRaises(lsst.pex.exceptions.InvalidParameterError):
            transform.readString(badStr2)
        transformFromStr1 = transform.readString(transformStr)
        self.assertTransformsEqual(transform, transformFromStr1)

        # check transformFromString
        transformFromStr2 = afwGeom.transformFromString(transformStr)
        self.assertTransformsEqual(transform, transformFromStr2)

        # Check pickling
        self.assertTransformsEqual(transform, pickle.loads(pickle.dumps(transform)))

        # Check afw::table::io persistence round-trip
        with lsst.utils.tests.getTempFilePath(".fits") as filename:
            transform.writeFits(filename)
            self.assertTransformsEqual(transform, type(transform).readFits(filename))
