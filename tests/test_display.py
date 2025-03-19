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
Tests for displaying devices

Run with:
   python test_display.py [backend]
"""
import os
import subprocess
import sys
import tempfile
import unittest

import lsst.utils.tests
import lsst.afw.image as afwImage
import lsst.afw.display as afwDisplay
import lsst.geom
from lsst.daf.base import PropertyList

try:
    type(backend)
except NameError:
    backend = "virtualDevice"
    oldBackend = None

TESTDIR = os.path.abspath(os.path.dirname(__file__))


class DisplayTestCase(unittest.TestCase):
    """A test case for Display"""

    def setUp(self):
        global oldBackend
        if backend != oldBackend:
            afwDisplay.setDefaultBackend(backend)
            afwDisplay.delAllDisplays()  # as some may use the old backend

            oldBackend = backend

        self.fileName = os.path.join(TESTDIR, "data", "HSC-0908120-056-small.fits")
        self.display0 = afwDisplay.Display(frame=0, verbose=True)

    def testMtv(self):
        """Test basic image display"""
        exp = afwImage.ExposureF(self.fileName)
        self.display0.mtv(exp, title="parent")

    def testMaskPlanes(self):
        """Test basic image display"""
        self.display0.setMaskTransparency(50)
        self.display0.setMaskPlaneColor("CROSSTALK", "orange")

    def testWith(self):
        """Test using displays with with statement"""
        with afwDisplay.Display(0) as disp:
            self.assertIsNotNone(disp)

    def testTwoDisplays(self):
        """Test that we can do things with two frames"""

        exp = afwImage.ExposureF(self.fileName)

        for frame in (0, 1):
            with afwDisplay.Display(frame, verbose=False) as disp:
                disp.setMaskTransparency(50)

                if frame == 1:
                    disp.setMaskPlaneColor("CROSSTALK", "ignore")
                disp.mtv(exp, title="parent")

                disp.erase()
                disp.dot('o', 205, 180, size=6, ctype=afwDisplay.RED)

    def testZoomPan(self):
        self.display0.pan(205, 180)
        self.display0.zoom(4)

        afwDisplay.Display(1).zoom(4, 205, 180)

    def testStackingOrder(self):
        """ Un-iconise and raise the display to the top of the stacking order if appropriate"""
        self.display0.show()

    def testDrawing(self):
        """Test drawing lines and glyphs"""
        self.display0.erase()

        exp = afwImage.ExposureF(self.fileName)
        # tells display0 about the image's xy0
        self.display0.mtv(exp, title="parent")

        with self.display0.Buffering():
            self.display0.dot('o', 200, 220)
            vertices = [(200, 220), (210, 230), (224, 230),
                        (214, 220), (200, 220)]
            self.display0.line(vertices, ctype=afwDisplay.CYAN)
            self.display0.line(vertices[:-1], symbs="+x+x", size=3)

    def testStretch(self):
        """Test playing with the lookup table"""
        self.display0.show()

        self.display0.scale("linear", "zscale")

    def testMaskColorGeneration(self):
        """Demonstrate the utility routine to generate mask plane colours
        (used by e.g. the ds9 implementation of _mtv)"""

        colorGenerator = self.display0.maskColorGenerator(omitBW=True)
        for i in range(10):
            print(i, next(colorGenerator), end=' ')
        print()

    def testImageTypes(self):
        """Check that we can display a range of types of image"""
        with afwDisplay.Display("dummy", "virtualDevice") as dummy:
            for imageType in [afwImage.DecoratedImageF,
                              afwImage.ExposureF,
                              afwImage.ImageF,
                              afwImage.MaskedImageF,
                              ]:
                im = imageType(self.fileName)
                dummy.mtv(im)

            for imageType in [afwImage.ImageU, afwImage.ImageI]:
                im = imageType(self.fileName, hdu=2, allowUnsafe=True)
                dummy.mtv(im)

            im = afwImage.Mask(self.fileName, hdu=2)
            dummy.mtv(im)

    def testInteract(self):
        r"""Check that interact exits when a q, \c CR, or \c ESC is pressed, or if a callback function
        returns a ``True`` value.
        If this is run using the virtualDevice a "q" is automatically triggered.
        If running the tests using ds9 you will be expected to do this manually.
        """
        print("Hit q to exit interactive mode")
        self.display0.interact()

    def testGetMaskPlaneColor(self):
        """Test that we can return mask colours either as a dict or maskplane by maskplane
        """
        mpc = self.display0.getMaskPlaneColor()

        maskPlane = 'DETECTED'
        self.assertEqual(mpc[maskPlane], self.display0.getMaskPlaneColor(maskPlane))

    def testSetDefaultImageColormap(self):
        """Test that we can set the default colourmap
        """
        self.display0.setDefaultImageColormap("gray")

    def testSetImageColormap(self):
        """Test that we can set a colourmap
        """
        self.display0.setImageColormap("gray")

    def testClose(self):
        """Test that we can close devices."""
        self.display0.close()

    def tearDown(self):
        for d in self.display0._displays.values():
            d.verbose = False           # ensure that display9.close() call is quiet

        del self.display0
        afwDisplay.delAllDisplays()


class TestFitsWriting(lsst.utils.tests.TestCase):
    """Test the FITS file writing used internally by afwDisplay."""

    def setUp(self):
        self.fileName = os.path.join(TESTDIR, "data", "HSC-0908120-056-small.fits")
        self.exposure = afwImage.ExposureF(self.fileName)
        self.unit = "nJy"
        self.exposure.metadata["BUNIT"] = self.unit

    def read_image(self, filename) -> tuple[afwImage.Image, PropertyList]:
        reader = afwImage.ImageFitsReader(filename)
        return reader.read(), reader.readMetadata()

    def read_mask(self, filename) -> tuple[afwImage.Mask, PropertyList]:
        reader = afwImage.MaskFitsReader(filename)
        return reader.read(), reader.readMetadata()

    def assertFitsEqual(
        self,
        fits_file: str,
        data: afwImage.Image | afwImage.Mask,
        wcs: lsst.afw.geom.SkyWcs | None,
        title: str | None,
        metadata: lsst.daf.base.PropertyList | None,
        unit: str | None,
    ):
        """Compare FITS file with parameters given to writeFitsImage."""
        if isinstance(data, afwImage.Image):
            new_data, new_metadata = self.read_image(fits_file)
        else:
            new_data, new_metadata = self.read_mask(fits_file)
        self.assertImagesEqual(new_data, data)
        if metadata and "BUNIT" in metadata:
            self.assertEqual(new_metadata["BUNIT"], metadata["BUNIT"])
        if metadata and unit:
            self.assertEqual(new_metadata["BUNIT"], unit)
        if title:
            self.assertEqual(new_metadata["OBJECT"], title)
        if wcs:
            # WCS needs to be shifted back to same reference.
            bbox = lsst.geom.Box2D(lsst.geom.Point2D(-100, -100), lsst.geom.Extent2D(300, 300))
            new_wcs = lsst.afw.geom.makeSkyWcs(new_metadata, strip=False)
            shift = lsst.geom.Extent2D(data.getX0(), data.getY0())
            unshifted_wcs = new_wcs.copyAtShiftedPixelOrigin(shift)
            self.assertWcsAlmostEqualOverBBox(unshifted_wcs, wcs, bbox)

    def test_named_file(self):
        """Write to a named file."""
        with tempfile.TemporaryDirectory() as tempdir:
            filename = os.path.join(tempdir, "test.fits")
            for data, wcs, title, metadata in (
                (self.exposure.image, self.exposure.wcs, "Write to file", self.exposure.metadata),
                (self.exposure.mask, self.exposure.wcs, "Mask to file", self.exposure.metadata),
                (self.exposure.image, None, "Image to file", self.exposure.metadata),
                (self.exposure.image, None, None, self.exposure.metadata),
                (self.exposure.image, None, None, None),
            ):
                afwDisplay.writeFitsImage(filename, data, wcs, title, metadata)
                self.assertFitsEqual(filename, data, wcs, title, metadata, self.unit)

    def test_file_handle(self):
        """Write to a file handle.

        This is how firefly uses afwDisplay.
        """
        with tempfile.NamedTemporaryFile(suffix=".fits") as tmp:
            afwDisplay.writeFitsImage(
                tmp, self.exposure.image, self.exposure.wcs, "filehdl", self.exposure.metadata
            )
            tmp.flush()
            tmp.seek(0)
            self.assertFitsEqual(
                tmp.name, self.exposure.image, self.exposure.wcs, "filehdl", self.exposure.metadata, self.unit
            )

    def test_fileno(self):
        """Write to a file descriptor.

        This is the way that the C++ interface worked.
        """
        with tempfile.TemporaryDirectory() as tempdir:
            filename = os.path.join(tempdir, "test.fits")
            with open(filename, "wb") as fh:
                afwDisplay.writeFitsImage(
                    fh.fileno(), self.exposure.image, self.exposure.wcs, "fileno", self.exposure.metadata
                )
            self.assertFitsEqual(
                filename, self.exposure.image, self.exposure.wcs, "fileno", self.exposure.metadata, self.unit
            )

    def test_subprocess(self):
        """Write through a pipe.

        This is how display_ds9 works.
        """
        with tempfile.TemporaryDirectory() as tempdir:
            filename = os.path.join(tempdir, "test.fits")
            with subprocess.Popen(
                [
                    sys.executable,
                    "-c",
                    # Minimal command that reads from stdin and writes to the
                    # named file.
                    "import sys; fh = open(sys.argv[1], 'wb'); fh.write(sys.stdin.buffer.read()); fh.close()",
                    filename,
                ],
                stdin=subprocess.PIPE
            ) as pipe:
                afwDisplay.writeFitsImage(
                    pipe, self.exposure.image, self.exposure.wcs, "pipe", self.exposure.metadata
                )
            self.assertFitsEqual(
                filename, self.exposure.image, self.exposure.wcs, "pipe", self.exposure.metadata, self.unit
            )


class MemoryTester(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run the image display test suite")

    parser.add_argument('backend', type=str, nargs="?", default="virtualDevice",
                        help="The backend to use, e.g. ds9.  You may need to have the device setup")
    args = parser.parse_args()

    # check that that backend is valid
    with afwDisplay.Display("test", backend=args.backend) as disp:
        pass

    backend = args.backend              # backend is just a variable in this file
    lsst.utils.tests.init()
    del sys.argv[1:]
    unittest.main()
