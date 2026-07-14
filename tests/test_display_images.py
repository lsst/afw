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

"""Tests for displaying lsst.images objects via lsst.afw.display."""
import unittest

import numpy as np

import lsst.utils.tests
import lsst.geom
import lsst.afw.display as afwDisplay
import lsst.afw.geom as afwGeom
import lsst.afw.image as afwImage
from lsst.afw.display._images_compat import normalize_lsst_images

try:
    import lsst.images
    import astropy.units as u
    HAVE_LSST_IMAGES = True
except ImportError:
    HAVE_LSST_IMAGES = False


if HAVE_LSST_IMAGES:
    class _UnsupportedGeneralizedImage(lsst.images.GeneralizedImage):
        """A GeneralizedImage subclass that display does not support."""

        @property
        def bbox(self):
            return lsst.images.Box.from_shape((1, 1))

        @property
        def sky_projection(self):
            return None

        def __getitem__(self, bbox):
            return self

        def copy(self):
            return self


def _make_legacy_wcs():
    """Return a simple lsst.afw.geom.SkyWcs."""
    return afwGeom.makeSkyWcs(
        lsst.geom.Point2D(5, 5),
        lsst.geom.SpherePoint(30.0, -10.0, lsst.geom.degrees),
        afwGeom.makeCdMatrix(scale=0.2 * lsst.geom.arcseconds),
    )


def _make_sky_projection(bbox):
    """Return an lsst.images.SkyProjection covering bbox.

    Built with from_legacy because projections created by from_fits_wcs
    cannot currently be converted back to SkyWcs (PIXEL vs PIXELS AST
    domain mismatch in lsst.images).
    """
    frame = lsst.images.DetectorFrame(instrument="TestCam", detector=0, bbox=bbox)
    return lsst.images.SkyProjection.from_legacy(_make_legacy_wcs(), frame, pixel_bounds=bbox)


def _make_image(with_projection=False):
    """Return an lsst.images.Image with a non-zero origin."""
    bbox = lsst.images.Box.from_shape((10, 20), start=(100, 200))
    sky_projection = _make_sky_projection(bbox) if with_projection else None
    return lsst.images.Image(
        np.arange(200, dtype=np.float32).reshape(10, 20),
        bbox=bbox,
        unit=u.nJy,
        sky_projection=sky_projection,
    )


def _make_mask():
    """Return an lsst.images.Mask with the COSMIC_RAY plane set everywhere."""
    schema = lsst.images.MaskSchema(
        [
            lsst.images.MaskPlane("SATURATED", "Saturated pixel."),
            lsst.images.MaskPlane("COSMIC_RAY", "Cosmic ray hit."),
        ]
    )
    mask = lsst.images.Mask(0, schema=schema, yx0=(100, 200), shape=(10, 20))
    mask.set("COSMIC_RAY", np.ones((10, 20), dtype=bool))
    return mask


def _make_masked_image(with_projection=False):
    """Return an lsst.images.MaskedImage, optionally with a sky projection."""
    image = _make_image()
    sky_projection = _make_sky_projection(image.bbox) if with_projection else None
    return lsst.images.MaskedImage(image, mask=_make_mask(), sky_projection=sky_projection)


@unittest.skipUnless(HAVE_LSST_IMAGES, "lsst.images is not available")
class NormalizeLsstImagesTestCase(lsst.utils.tests.TestCase):
    """Direct tests of normalize_lsst_images."""

    def test_afw_passthrough(self):
        """afw objects and their wcs must pass through untouched."""
        image = afwImage.ImageF(3, 4)
        wcs = _make_legacy_wcs()
        data, outwcs = normalize_lsst_images(image, wcs)
        self.assertIs(data, image)
        self.assertIs(outwcs, wcs)

    def test_image(self):
        image = _make_image()
        data, wcs = normalize_lsst_images(image, None)
        self.assertIsInstance(data, afwImage.ImageF)
        self.assertEqual(data.getXY0(), lsst.geom.Point2I(200, 100))
        np.testing.assert_array_equal(data.array, image.array)
        self.assertIsNone(wcs)

    def test_image_with_projection(self):
        image = _make_image(with_projection=True)
        data, wcs = normalize_lsst_images(image, None)
        self.assertIsInstance(data, afwImage.ImageF)
        self.assertIsInstance(wcs, afwGeom.SkyWcs)
        # The converted WCS must agree with the source projection.
        expected = _make_legacy_wcs()
        self.assertSpherePointsAlmostEqual(
            wcs.pixelToSky(205.0, 105.0), expected.pixelToSky(205.0, 105.0)
        )

    def test_image_wcs_kwarg(self):
        """A caller-supplied wcs is kept when the image has no projection."""
        image = _make_image()
        wcs = _make_legacy_wcs()
        data, outwcs = normalize_lsst_images(image, wcs)
        self.assertIs(outwcs, wcs)

    def test_wcs_conflict(self):
        image = _make_image(with_projection=True)
        with self.assertRaises(RuntimeError):
            normalize_lsst_images(image, _make_legacy_wcs())

    def test_mask(self):
        mask = _make_mask()
        data, wcs = normalize_lsst_images(mask, None)
        self.assertIsInstance(data, afwImage.Mask)
        self.assertEqual(data.getXY0(), lsst.geom.Point2I(200, 100))
        self.assertIsNone(wcs)
        planes = data.getMaskPlaneDict()
        self.assertIn("COSMIC_RAY", planes)
        self.assertIn("SATURATED", planes)
        cr = data.getPlaneBitMask("COSMIC_RAY")
        self.assertTrue(((data.array & cr) != 0).all())
        sat = data.getPlaneBitMask("SATURATED")
        self.assertTrue(((data.array & sat) == 0).all())

    def test_masked_image(self):
        masked_image = _make_masked_image(with_projection=True)
        data, wcs = normalize_lsst_images(masked_image, None)
        self.assertIsInstance(data, afwImage.MaskedImageF)
        self.assertEqual(data.getXY0(), lsst.geom.Point2I(200, 100))
        np.testing.assert_array_equal(data.image.array, masked_image.image.array)
        np.testing.assert_array_equal(data.variance.array, masked_image.variance.array)
        self.assertIn("COSMIC_RAY", data.mask.getMaskPlaneDict())
        cr = data.mask.getPlaneBitMask("COSMIC_RAY")
        self.assertTrue(((data.mask.array & cr) != 0).all())
        self.assertIsInstance(wcs, afwGeom.SkyWcs)

    def test_masked_image_without_projection(self):
        masked_image = _make_masked_image()
        data, wcs = normalize_lsst_images(masked_image, None)
        self.assertIsInstance(data, afwImage.MaskedImageF)
        self.assertIsNone(wcs)

    def test_unsupported_generalized_image(self):
        """Unsupported subclasses pass through for Display to reject."""
        unsupported = _UnsupportedGeneralizedImage()
        data, wcs = normalize_lsst_images(unsupported, None)
        self.assertIs(data, unsupported)
        self.assertIsNone(wcs)


@unittest.skipUnless(HAVE_LSST_IMAGES, "lsst.images is not available")
class DisplayLsstImagesTestCase(lsst.utils.tests.TestCase):
    """End-to-end mtv tests through the virtualDevice backend."""

    def setUp(self):
        afwDisplay.setDefaultBackend("virtualDevice")
        afwDisplay.delAllDisplays()
        self.display = afwDisplay.Display(frame=0, verbose=True)

    def tearDown(self):
        afwDisplay.delAllDisplays()

    def test_mtv_image(self):
        self.display.mtv(_make_image(), title="image")
        self.assertEqual(self.display._xy0, lsst.geom.Point2I(200, 100))

    def test_mtv_mask(self):
        self.display.mtv(_make_mask(), title="mask")
        self.assertEqual(self.display._xy0, lsst.geom.Point2I(200, 100))

    def test_mtv_masked_image(self):
        self.display.mtv(_make_masked_image(with_projection=True), title="masked image")
        self.assertEqual(self.display._xy0, lsst.geom.Point2I(200, 100))

    def test_mtv_wcs_conflict(self):
        masked_image = _make_masked_image(with_projection=True)
        with self.assertRaises(RuntimeError):
            self.display.mtv(masked_image, wcs=_make_legacy_wcs())

    def test_mtv_unsupported(self):
        with self.assertRaises(TypeError):
            self.display.mtv(_UnsupportedGeneralizedImage())

    def test_default_mask_plane_colors(self):
        """Renamed planes keep their traditional colors."""
        self.assertEqual(self.display.getMaskPlaneColor("COSMIC_RAY"), afwDisplay.MAGENTA)
        self.assertEqual(self.display.getMaskPlaneColor("DETECTION_EDGE"), afwDisplay.YELLOW)


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
