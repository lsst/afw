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

"""
Examples of using Footprints
"""

import lsst.geom
import lsst.afw.image as afwImage
import lsst.afw.detection as afwDetect
import lsst.afw.display as afwDisplay

afwDisplay.setDefaultMaskTransparency(75)


def showPeaks(im=None, fs=None, frame=0):
    """Show the image and peaks"""
    if frame is None:
        return

    disp = afwDisplay.Display(frame=frame)
    if im:
        disp.mtv(im, title="Image and peaks")

    if fs:
        with disp.Buffering():           # turn on buffering of display's slow "region" writes
            for foot in fs.getFootprints():
                for p in foot.getPeaks():
                    disp.dot("+", p.getIx(), p.getIy(), size=0.4, ctype=afwDisplay.RED)


def run(frame=6):
    im = afwImage.MaskedImageF(lsst.geom.Extent2I(14, 10))
    #
    # Populate the image with objects that we should detect
    #
    objects = []
    objects.append([(4, 1, 10), (3, 2, 10), (4, 2, 20),
                    (5, 2, 10), (4, 3, 10), ])
    objects.append([(9, 7, 30), (10, 7, 29), (12, 7, 28),
                    (10, 8, 27), (11, 8, 26), (10, 4, -5)])
    objects.append([(3, 8, 10), (4, 8, 10), ])

    for obj in objects:
        for x, y, I in obj:
            im.getImage()[x, y, afwImage.LOCAL] = I

    im.getVariance().set(1)
    im.getVariance()[10, 4, afwImage.LOCAL] = 0.5**2
    #
    # Detect the objects at 10 counts or above
    #
    level = 10
    fs = afwDetect.FootprintSet(im, afwDetect.Threshold(level), "DETECTED")

    showPeaks(im, fs, frame=frame)
    #
    # Detect the objects at -10 counts or below.  N.b. the peak's at -5, so it isn't detected
    #
    polarity = False                     # look for objects below background
    threshold = afwDetect.Threshold(level, afwDetect.Threshold.VALUE, polarity)
    fs2 = afwDetect.FootprintSet(im, threshold, "DETECTED_NEGATIVE")
    print("Detected %d objects below background" % len(fs2.getFootprints()))
    #
    # Search in S/N (n.b. the peak's -10sigma)
    #
    threshold = afwDetect.Threshold(
        level, afwDetect.Threshold.PIXEL_STDEV, polarity)
    fs2 = afwDetect.FootprintSet(im, threshold)
    #
    # Here's another way to set a mask plane (we chose not to do so in the FootprintSet call)
    #
    msk = im.getMask()
    afwDetect.setMaskFromFootprintList(
        msk, fs2.getFootprints(), msk.getPlaneBitMask("DETECTED_NEGATIVE"))

    if frame is not None:
        frame += 1
        afwDisplay.Display(frame=frame).mtv(msk, title="Image Mask")
    #
    # Merge the positive and negative detections, growing both sets by 1 pixel
    #
    fs.merge(fs2, 1, 1)
    #
    # Set EDGE so we can see the grown Footprints
    #
    afwDetect.setMaskFromFootprintList(
        msk, fs.getFootprints(), msk.getPlaneBitMask("EDGE"))

    if frame is not None:
        frame += 1
        afwDisplay.Display(frame=frame).mtv(msk, title="Grown Mask")

    showPeaks(fs=fs, frame=frame + 1)


if __name__ == "__main__":
    run(frame=None)
