#
# LSST Data Management System
# Copyright 2021 LSST Corporation.
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
import unittest
import hpgeom as hpg
import numpy as np
from astropy.wcs import WCS

import lsst.utils.tests
import lsst.afw.geom as afwGeom


class HpxUtilsTestCase(lsst.utils.tests.TestCase):
    def test_hpx_wcs(self):
        """Test various computations of HPX wcs.
        """
        # The default is 2**9=512 pixels on a side.  We also
        # test smaller and larger tiles.
        shift_order_tests = [8, 9, 10]
        # Choose arbitrary positions that are north, south,
        # equatorial, east, and west.  41.8 degrees is special.
        pos_tests = [(0.0, 0.0),
                     (43.0, 3.0),
                     (127.0, -23.0),
                     (194.0, 36.0),
                     (207.0, -41.5),
                     (256.0, 42.0),
                     (302.0, -67.0),
                     (332.0, 75.0),
                     (348.0, -83.0),
                     (358.0, 89.0)]

        # Default order of 11 (nside=2048) gives 0.2" pixels for 512
        # subpixels
        hips_order_tests = [12, 11, 10]

        for shift_order, hips_order in zip(shift_order_tests, hips_order_tests):
            nside = 2**hips_order
            nsubpix = 2**shift_order

            for pos in pos_tests:
                pixel = hpg.angle_to_pixel(nside, pos[0], pos[1])

                wcs = afwGeom.makeHpxWcs(hips_order, pixel, shift_order=shift_order)

                y, x = np.meshgrid(np.arange(nsubpix), np.arange(nsubpix))
                x = x.ravel().astype(np.float64)
                y = y.ravel().astype(np.float64)
                ra, dec = wcs.pixelToSkyArray(x, y, degrees=True)
                ra[ra == 360.0] = 0.0

                # Check that all these positions are truly inside the pixel
                radec_pixels = hpg.angle_to_pixel(nside, ra, dec)
                np.testing.assert_array_equal(radec_pixels, pixel)

                # Check that the orientation is correct.
                # pixel (0, 0) should be E
                # pixel (nsubpix - 1, 0) should be N
                # pixel (nsubpix - 1, nsubpix - 1) should be W
                # pixel (0, nsubpix - 1) should be S
                xx = np.array([0, nsubpix - 1, nsubpix - 1, 0], dtype=np.float64)
                yy = np.array([0, 0, nsubpix - 1, nsubpix - 1], dtype=np.float64)
                ra_cornerpix, dec_cornerpix = wcs.pixelToSkyArray(xx, yy, degrees=True)

                # Generate all the sub-pixels
                bit_shift = 2*int(np.round(np.log2(nsubpix)))
                sub_pixels = np.left_shift(pixel, bit_shift) + np.arange(nsubpix*nsubpix)
                ra_sub, dec_sub = hpg.pixel_to_angle(nside*nsubpix, sub_pixels)
                # Deal with RA = 0 for testing...
                if ra_sub.max() > 350.0 and ra_sub.min() < 10.0:
                    hi, = np.where(ra_sub > 180.0)
                    ra_sub[hi] -= 360.0
                    hi_corner, = np.where(ra_cornerpix > 180.0)
                    ra_cornerpix[hi_corner] -= 360.0

                cos_dec = np.cos(np.deg2rad(np.median(dec)))

                easternmost = np.argmax(ra_sub)
                self.assertFloatsAlmostEqual(ra_cornerpix[0], ra_sub[easternmost], atol=1e-13/cos_dec)
                self.assertFloatsAlmostEqual(dec_cornerpix[0], dec_sub[easternmost], atol=5e-13)
                northernmost = np.argmax(dec_sub)
                self.assertFloatsAlmostEqual(ra_cornerpix[1], ra_sub[northernmost], atol=1e-13/cos_dec)
                self.assertFloatsAlmostEqual(dec_cornerpix[1], dec_sub[northernmost], atol=5e-13)
                westernmost = np.argmin(ra_sub)
                self.assertFloatsAlmostEqual(ra_cornerpix[2], ra_sub[westernmost], atol=1e-13/cos_dec)
                self.assertFloatsAlmostEqual(dec_cornerpix[2], dec_sub[westernmost], atol=5e-13)
                southernmost = np.argmin(dec_sub)
                self.assertFloatsAlmostEqual(ra_cornerpix[3], ra_sub[southernmost], atol=1e-13/cos_dec)
                self.assertFloatsAlmostEqual(dec_cornerpix[3], dec_sub[southernmost], atol=5e-13)

                # Confirm that the transformation also works with astropy WCS
                astropy_wcs = WCS(header=wcs.getFitsMetadata().toDict())
                astropy_coords = astropy_wcs.pixel_to_world(x, y)

                astropy_ra = astropy_coords.ra.degree
                astropy_dec = astropy_coords.dec.degree

                # The astropy warping is only consistent at the 5e-11 level.
                self.assertFloatsAlmostEqual(astropy_ra, ra, atol=5e-11/cos_dec)
                self.assertFloatsAlmostEqual(astropy_dec, dec, atol=5e-11)

    def test_hpx_wcs_bad_inputs(self):
        """Test assertions for bad inputs to makeHpxWcs()
        """
        # order must be positive.
        self.assertRaises(ValueError, afwGeom.makeHpxWcs, 0, 100)
        self.assertRaises(ValueError, afwGeom.makeHpxWcs, -10, 100)

        # pixel number must be in range.
        self.assertRaises(ValueError, afwGeom.makeHpxWcs, 5, -1)
        self.assertRaises(ValueError, afwGeom.makeHpxWcs, 5, 12*2**10 + 1)

        # tilepix must be a positive power of 2.
        # shift_order must be positive
        self.assertRaises(ValueError, afwGeom.makeHpxWcs, 5, 0, shift_order=0)
        self.assertRaises(ValueError, afwGeom.makeHpxWcs, 5, 0, shift_order=-10)


class MemoryTester(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
