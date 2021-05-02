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

__all__ = ["ReadoutCornerValNameDict", "ReadoutCornerNameValDict"]


from lsst.geom import Extent2I
from lsst.utils import inClass
from ._cameraGeom import Amplifier, ReadoutCorner


ReadoutCornerValNameDict = {
    ReadoutCorner.LL: "LL",
    ReadoutCorner.LR: "LR",
    ReadoutCorner.UR: "UR",
    ReadoutCorner.UL: "UL",
}
ReadoutCornerNameValDict = {val: key for key, val in
                            ReadoutCornerValNameDict.items()}


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
    #
    # All of these have now been transferred to the amp geometry
    #
    self.setRawXYOffset(outOffset)
    self.setRawFlipX(outFlipX)
    self.setRawFlipY(outFlipY)
    return self
