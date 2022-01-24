# This file is part of afw.
#
# Developed for the LSST Data Management System.
# This product includes software developed by the LSST Project
# (https://www.lsst.org).
# See the COPYRIGHT file at the top-level directory of this distribution
# for details of code ownership.
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
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

__all__ = ["AmplifierGeometryComparison", "ReadoutCornerValNameDict", "ReadoutCornerNameValDict"]

import enum

from lsst.geom import Extent2I
from lsst.utils import continueClass, inClass
from ._cameraGeom import Amplifier, ReadoutCorner


ReadoutCornerValNameDict = {
    ReadoutCorner.LL: "LL",
    ReadoutCorner.LR: "LR",
    ReadoutCorner.UR: "UR",
    ReadoutCorner.UL: "UL",
}
ReadoutCornerNameValDict = {val: key for key, val in
                            ReadoutCornerValNameDict.items()}


class AmplifierGeometryComparison(enum.Flag):
    """Flags used to report geometric differences between amplifier"""

    EQUAL = 0
    """All tested properties of the two amplifiers are equal."""

    SHIFTED_X = enum.auto()
    """Amplifiers have different X offsets relative to assembled raw."""

    SHIFTED_Y = enum.auto()
    """Amplifiers have different Y offsets relative to assembled raw."""

    SHIFTED = SHIFTED_X | SHIFTED_Y
    """Amplifiers are different offsets relative to assembled raw."""

    FLIPPED_X = enum.auto()
    """Amplifiers differ by (at least) an X-coordinate flip."""

    FLIPPED_Y = enum.auto()
    """Amplifiers differ by (at least) a Y-coordinate flip."""

    FLIPPED = FLIPPED_X | FLIPPED_Y
    """Amplifiers differ by (at least) a coordinate flip."""

    ASSEMBLY_DIFFERS = SHIFTED | FLIPPED
    """Amplifiers differ in offsets relative to raw, indicating at least a
    difference in assembly state.
    """

    REGIONS_DIFFER = enum.auto()
    """Amplifiers have different full/data/overscan/prescan regions.

    If ``assembly=True`` was passed to `Amplifier.compare`, this will only be
    set if regions differ even after applying flips and offsets to make the
    assembly states the same.  If ``assembly=False`` was passed to
    `Amplifier.compare`, regions will be compared while assuming that assembly
    state is the same.
    """


@continueClass
class Amplifier:  # noqa: F811

    def compareGeometry(self, other, *, assembly=True, regions=True):
        """Compare the geometry of this amplifier with another.

        Parameters
        ----------
        assembly : `bool`, optional
            If `True` (default) test whether flips and offsets relative to
            assembled raw are the same, and account for those when testing
            whether regions are the same.
        regions : `bool`, optional
            If `True` (default) test whether full/data/overscan/prescan regions
            are the same.

        Returns
        -------
        comparison : `AmplifierGeometryComparison`
            Flags representing the result of the comparison.
        """
        result = AmplifierGeometryComparison.EQUAL
        if assembly:
            if self.getRawXYOffset().getX() != other.getRawXYOffset().getX():
                result |= AmplifierGeometryComparison.SHIFTED_X
            if self.getRawXYOffset().getY() != other.getRawXYOffset().getY():
                result |= AmplifierGeometryComparison.SHIFTED_Y
            if self.getRawFlipX() != other.getRawFlipX():
                result |= AmplifierGeometryComparison.FLIPPED_X
            if self.getRawFlipY() != other.getRawFlipY():
                result |= AmplifierGeometryComparison.FLIPPED_Y
        if regions:
            if result & AmplifierGeometryComparison.ASSEMBLY_DIFFERS:
                # Transform (a copy of) other to the same assembly state as
                # self.
                other = other.rebuild().transform(
                    outOffset=self.getRawXYOffset(),
                    outFlipX=self.getRawFlipX(),
                    outFlipY=self.getRawFlipY(),
                ).finish()
            for bboxName in ("",
                             "HorizontalOverscan",
                             "Data",
                             "VerticalOverscan",
                             "Prescan"):
                if getattr(self, f"getRaw{bboxName}BBox")() != getattr(other, f"getRaw{bboxName}BBox")():
                    result |= AmplifierGeometryComparison.REGIONS_DIFFER
        return result


@inClass(Amplifier.Builder)
def transform(self, *, outOffset=None, outFlipX=False, outFlipY=False):
    """Transform an amplifier builder (in-place) by applying shifts and
    flips.

    Parameters
    ----------
    outOffset : `lsst.geom.Extent2I`, optional
        Post-transformation return value for ``self.getRawXYOffset()``.
        The default is ``(0, 0)``, which shifts the amplifier to its
        position in the assembled (but still untrimmed) raw image.
    outFlipX : `bool`, optional
        Post-transformation return value for ``self.getRawFlipX()``.  The
        default is `False`, which flips the amplifier to its correct
        X orientation in the assembled raw image.
    outFlipX : `bool`, optional
        Post-transformation return value for ``self.getRawFlipY()``.  The
        default is `False`, which flips the amplifier to its correct
        Y orientation in the assembled raw image.

    Returns
    -------
    self : `AmplifierBuilder`
        Returned to enable method chaining, e.g.
        ``amplifier.rebuild().transform().finish()``.
    """
    if outOffset is None:
        outOffset = Extent2I(0, 0)
    bbox = self.getRawBBox()
    awidth, aheight = bbox.getDimensions()
    #
    # Figure out how far flipping the amp LR and/or TB offsets the bboxes
    #
    boxMin0 = bbox.getMin()     # initial position of rawBBox's LLC corner
    if self.getRawFlipX() != outFlipX:
        bbox.flipLR(awidth)
    if self.getRawFlipY() != outFlipY:
        bbox.flipTB(aheight)
    shift = boxMin0 - bbox.getMin()

    for bboxName in ("",
                     "HorizontalOverscan",
                     "Data",
                     "VerticalOverscan",
                     "Prescan"):
        bbox = getattr(self, f"getRaw{bboxName}BBox")()
        if self.getRawFlipX() != outFlipX:
            bbox.flipLR(awidth)
        if self.getRawFlipY() != outFlipY:
            bbox.flipTB(aheight)
        bbox.shift(self.getRawXYOffset() + shift - outOffset)

        getattr(self, f"setRaw{bboxName}BBox")(bbox)

    # Update the Readout Corner if we've flipped anything.
    outReadoutCorner = self.getReadoutCorner()
    if self.getRawFlipX() != outFlipX:
        xFlipMapping = {ReadoutCorner.LL: ReadoutCorner.LR, ReadoutCorner.LR: ReadoutCorner.LL,
                        ReadoutCorner.UR: ReadoutCorner.UL, ReadoutCorner.UL: ReadoutCorner.UR}
        outReadoutCorner = xFlipMapping[outReadoutCorner]
    if self.getRawFlipY() != outFlipY:
        yFlipMapping = {ReadoutCorner.LL: ReadoutCorner.UL, ReadoutCorner.LR: ReadoutCorner.UR,
                        ReadoutCorner.UR: ReadoutCorner.LR, ReadoutCorner.UL: ReadoutCorner.LL}
        outReadoutCorner = yFlipMapping[outReadoutCorner]
    if outReadoutCorner != self.getReadoutCorner():
        self.setReadoutCorner(outReadoutCorner)

    #
    # All of these have now been transferred to the amp geometry
    #
    self.setRawXYOffset(outOffset)
    self.setRawFlipX(outFlipX)
    self.setRawFlipY(outFlipY)
    return self
