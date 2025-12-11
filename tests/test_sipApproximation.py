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
#

import os
import unittest

import astropy.coordinates
import astropy.units as u
import astropy.wcs
import numpy as np

import lsst.utils.tests
from lsst.daf.base import PropertyList
from lsst.geom import Point2D, Point2I, Extent2I, Box2D, Box2I, SpherePoint, arcseconds, degrees
from lsst.afw.geom import SipApproximation, makeSkyWcs, makeCdMatrix, SkyWcs


class SipApproximationTestCases(lsst.utils.tests.TestCase):
    @staticmethod
    def make_wcs(md):
        return makeSkyWcs(PropertyList.from_mapping(md))

    def compare_to_astropy(self, wcs, bbox, deltaSkyMax, deltaPixelMax):
        """Compare a WCS to Astropy's WCS via its FITS representation."""
        x, y = np.meshgrid(np.linspace(bbox.x.min, bbox.x.max, 50), np.linspace(bbox.y.min, bbox.y.max, 50))
        x = x.flatten()
        y = y.flatten()
        astropy_wcs = astropy.wcs.WCS(wcs.getFitsMetadata(bbox=bbox))
        astropy_coords = astropy_wcs.pixel_to_world(x, y)
        afw_ra, afw_dec = wcs.pixelToSkyArray(x, y)
        afw_coords = astropy.coordinates.SkyCoord(afw_ra * u.radian, afw_dec * u.radian)
        sky_separation = astropy_coords.separation(afw_coords).to(u.arcsec)
        self.assertLess(sky_separation.max(), deltaSkyMax)
        astropy_x, astropy_y = astropy_wcs.world_to_pixel(afw_coords)
        pixel_separation = ((astropy_x - x) ** 2 + (astropy_y - y) ** 2) ** 0.5
        self.assertLess(pixel_separation.max(), deltaPixelMax)

    def test_pure_tan(self):
        """Test 'approximating' a pure TAN WCS that we should be able to
        represent exactly with no SIP.
        """
        pixelOrigin = Point2D(500.0, 3000.09)
        skyOrigin = SpherePoint(60.0, 40.0, degrees)
        cd = makeCdMatrix(0.2 * arcseconds, orientation=30 * degrees)
        target = makeSkyWcs(pixelOrigin, skyOrigin, cd)
        bbox = Box2I(Point2I(0, 0), Extent2I(2048, 4176))
        approx = SipApproximation(target, Box2D(bbox), Extent2I(5, 5), 0, pixelOrigin=pixelOrigin)
        self.assertAlmostEqual(approx.getPixelOrigin(), pixelOrigin)
        self.assertLess(approx.getSkyOrigin().separation(skyOrigin), 1e-9 * arcseconds)
        self.assertFloatsAlmostEqual(approx.getCdMatrix(), cd, rtol=1e-15)
        deltaSky, deltaPixel = approx.computeDeltas()
        self.assertLess(deltaSky, 1e-9 * arcseconds)
        self.assertLess(deltaPixel, 1e-9)
        approxWcs = approx.getWcs()
        fits = dict(approxWcs.getFitsMetadata(bbox=bbox))
        self.assertFloatsAlmostEqual(fits["CRPIX1"], pixelOrigin.x + 1, rtol=1e-15)
        self.assertFloatsAlmostEqual(fits["CRPIX2"], pixelOrigin.y + 1, rtol=1e-15)
        self.assertFloatsAlmostEqual(fits["CRVAL1"], skyOrigin.getRa().asDegrees(), rtol=1e-15)
        self.assertFloatsAlmostEqual(fits["CRVAL2"], skyOrigin.getDec().asDegrees(), rtol=1e-15)
        self.assertFloatsAlmostEqual(fits["CD1_1"], cd[0, 0], rtol=1e-15)
        self.assertFloatsAlmostEqual(fits["CD1_2"], cd[0, 1], rtol=1e-15)
        self.assertFloatsAlmostEqual(fits["CD2_1"], cd[1, 0], rtol=1e-15)
        self.assertFloatsAlmostEqual(fits["CD2_2"], cd[1, 1], rtol=1e-15)
        self.assertEqual(fits["CTYPE1"], "RA---TAN")
        self.assertEqual(fits["CTYPE2"], "DEC--TAN")
        self.compare_to_astropy(approxWcs, bbox, deltaSkyMax=1e-9 * u.arcsec, deltaPixelMax=1e-8)
        self.compare_to_astropy(
            target.copyWithFitsApproximation(approxWcs),
            bbox,
            deltaSkyMax=1e-9 * u.arcsec,
            deltaPixelMax=1e-8,
        )

    def test_linear(self):
        """Test 'approximating' a TAN-SIP WCS with only linear terms that we
        should be able to represent exactly with no SIP, by nulling the linear
        terms.
        """
        pixelOrigin = Point2D(500.0, 3000.09)
        skyOrigin = SpherePoint(60.0, 40.0, degrees)
        cd = makeCdMatrix(0.2 * arcseconds, orientation=30 * degrees)
        bbox = Box2I(Point2I(0, 0), Extent2I(2048, 4176))
        target = self.make_wcs(
            {
                "CD1_1": cd[0, 0],
                "CD1_2": cd[0, 1],
                "CD2_1": cd[1, 0],
                "CD2_2": cd[1, 1],
                "CRPIX1": pixelOrigin.x + 1,
                "CRPIX2": pixelOrigin.y + 1,
                "CRVAL1": skyOrigin.getRa().asDegrees(),
                "CRVAL2": skyOrigin.getDec().asDegrees(),
                "CTYPE1": "RA---TAN-SIP",
                "CTYPE2": "DEC--TAN-SIP",
                "CUNIT1": "deg",
                "CUNIT2": "deg",
                "NAXES1": bbox.width,
                "NAXES2": bbox.height,
                "NAXIS": 2,
                "RADESYS": "ICRS",
                "A_ORDER": 1,
                "B_ORDER": 1,
                "AP_ORDER": 1,
                "BP_ORDER": 1,
                "A_1_0": 1.0,
                "A_0_1": 2.0,
                "B_1_0": 2.0,
                "B_0_1": 3.0,
                "AP_1_0": 0.0,
                "AP_0_1": -0.5,
                "BP_1_0": -0.5,
                "BP_0_1": -0.5,
            }
        )
        approx = SipApproximation(target, Box2D(bbox), Extent2I(5, 5), 0, pixelOrigin=pixelOrigin)
        self.assertAlmostEqual(approx.getPixelOrigin(), pixelOrigin)
        self.assertLess(approx.getSkyOrigin().separation(skyOrigin), 1e-9 * arcseconds)
        deltaSky, deltaPixel = approx.computeDeltas()
        self.assertLess(deltaSky, 1e-9 * arcseconds)
        self.assertLess(deltaPixel, 1e-9)
        approxWcs = approx.getWcs()
        fits = dict(approxWcs.getFitsMetadata(bbox=bbox))
        self.assertFloatsAlmostEqual(fits["CRPIX1"], pixelOrigin.x + 1)
        self.assertFloatsAlmostEqual(fits["CRPIX2"], pixelOrigin.y + 1)
        self.assertFloatsAlmostEqual(fits["CRVAL1"], skyOrigin.getRa().asDegrees(), rtol=1e-15)
        self.assertFloatsAlmostEqual(fits["CRVAL2"], skyOrigin.getDec().asDegrees(), rtol=1e-15)
        self.assertEqual(fits["CTYPE1"], "RA---TAN")
        self.assertEqual(fits["CTYPE2"], "DEC--TAN")
        self.compare_to_astropy(approxWcs, bbox, deltaSkyMax=1e-9 * u.arcsec, deltaPixelMax=1e-8)
        self.compare_to_astropy(
            target.copyWithFitsApproximation(approxWcs),
            bbox,
            deltaSkyMax=1e-9 * u.arcsec,
            deltaPixelMax=1e-8,
        )

    def test_calexp03(self):
        """Test on a real WCS, from the calexp of HSC visit=1228 ccd=3,
        from run RC/w_2017_50/DM-12929.  It has realistic distortion with CRPIX
        within the CCD image.
        """
        pixelOrigin = Point2D(1096.8935760454308, 2262.9403834197587)
        skyOrigin = SpherePoint(149.81129315622917, 1.527518593302043, degrees)
        cd = np.array(
            [
                [-1.178015203669291e-06, 4.498766772429112e-05],
                [
                    4.29304691824093e-05,
                    -1.0762546100726416e-06,
                ],
            ]
        )
        bbox = Box2I(Point2I(0, 0), Extent2I(2048, 4176))
        target = self.make_wcs(
            {
                "AP_0_0": -0.00042766492723573385,
                "AP_0_1": -1.421294634690333e-06,
                "AP_0_2": -2.249070237131596e-06,
                "AP_0_3": 7.614693711846379e-11,
                "AP_0_4": -2.1350939718171946e-15,
                "AP_1_0": -2.9070491393671955e-06,
                "AP_1_1": -2.600191883647906e-06,
                "AP_1_2": 3.209251813253422e-10,
                "AP_1_3": -6.377163911922792e-15,
                "AP_2_0": -6.917400901963954e-06,
                "AP_2_1": 1.441645068817284e-10,
                "AP_2_2": -1.4384914067292817e-14,
                "AP_3_0": 5.179922357600516e-10,
                "AP_3_1": -9.618846794061896e-15,
                "AP_4_0": -1.6455748303909694e-14,
                "AP_ORDER": 4,
                "A_0_2": 2.2496092550091135e-06,
                "A_0_3": -5.7226624826215967e-11,
                "A_1_1": 2.601332947490435e-06,
                "A_1_2": -2.548161981578503e-10,
                "A_2_0": 6.918453783853945e-06,
                "A_2_1": -7.070288496574992e-11,
                "A_3_0": -4.1515075181628575e-10,
                "A_ORDER": 3,
                "BP_0_0": -0.00027519363188290014,
                "BP_0_1": -1.339922915231262e-06,
                "BP_0_2": -2.7508000129989704e-06,
                "BP_0_3": 1.6166435243727447e-10,
                "BP_0_4": -2.6159444529413283e-15,
                "BP_1_0": -7.170533032300998e-07,
                "BP_1_1": -4.157202213924163e-06,
                "BP_1_2": 1.0655925530147004e-10,
                "BP_1_3": -5.8777807071581165e-15,
                "BP_2_0": -1.4037210918092256e-06,
                "BP_2_1": 1.9156184107700897e-10,
                "BP_2_2": -5.251709685663605e-15,
                "BP_3_0": -7.09409582069615e-11,
                "BP_3_1": -5.02267443301453e-15,
                "BP_4_0": 5.642051133209865e-16,
                "BP_ORDER": 4,
                "B_0_2": 2.751296696053813e-06,
                "B_0_3": -1.363848403847289e-10,
                "B_1_1": 4.157792860966118e-06,
                "B_1_2": -5.4037152183616375e-11,
                "B_2_0": 1.4039800636228212e-06,
                "B_2_1": -1.2897941316430668e-10,
                "B_3_0": 9.63138959016586e-11,
                "B_ORDER": 3,
                "CD1_1": cd[0, 0],
                "CD1_2": cd[0, 1],
                "CD2_1": cd[1, 0],
                "CD2_2": cd[1, 1],
                "CRPIX1": pixelOrigin.x + 1,
                "CRPIX2": pixelOrigin.y + 1,
                "CRVAL1": skyOrigin.getRa().asDegrees(),
                "CRVAL2": skyOrigin.getDec().asDegrees(),
                "CTYPE1": "RA---TAN-SIP",
                "CTYPE2": "DEC--TAN-SIP",
                "CUNIT1": "deg",
                "CUNIT2": "deg",
                "NAXES1": bbox.width,
                "NAXES2": bbox.height,
                "NAXIS": 2,
                "RADESYS": "ICRS",
            }
        )
        approx = SipApproximation(target, Box2D(bbox), Extent2I(6, 6), 4, pixelOrigin=pixelOrigin)
        self.assertAlmostEqual(approx.getPixelOrigin(), pixelOrigin)
        self.assertLess(approx.getSkyOrigin().separation(skyOrigin), 1e-9 * arcseconds)
        self.assertFloatsAlmostEqual(approx.getCdMatrix(), cd, rtol=1e-7)
        deltaSky, deltaPixel = approx.computeDeltas()
        self.assertLess(deltaSky, 1e-7 * arcseconds)
        self.assertLess(deltaPixel, 1e-6)
        approxWcs = approx.getWcs()
        fits = dict(approxWcs.getFitsMetadata(bbox=bbox))
        self.assertFloatsAlmostEqual(fits["CRPIX1"], pixelOrigin.x + 1, rtol=1e-8)
        self.assertFloatsAlmostEqual(fits["CRPIX2"], pixelOrigin.y + 1, rtol=1e-8)
        self.assertFloatsAlmostEqual(fits["CRVAL1"], skyOrigin.getRa().asDegrees(), rtol=1e-8)
        self.assertFloatsAlmostEqual(fits["CRVAL2"], skyOrigin.getDec().asDegrees(), rtol=1e-8)
        self.assertEqual(fits["CTYPE1"], "RA---TAN-SIP")
        self.assertEqual(fits["CTYPE2"], "DEC--TAN-SIP")
        self.compare_to_astropy(approxWcs, bbox, deltaSkyMax=1e-7 * u.arcsec, deltaPixelMax=1e-5)
        self.compare_to_astropy(
            target.copyWithFitsApproximation(approxWcs),
            bbox,
            deltaSkyMax=1e-6 * u.arcsec,
            deltaPixelMax=1e-5,
        )

    def test_wcs22(self):
        """Test on real WCS, from the [meas_mosaic] wcs of HSC visit=1228
        ccd=22 tract=9813, from rerun RC/w_2017_50/DM-12929.  This has
        realistic distortion with CRPIX near the center of the focal plane (off
        the edge of the CCD), which means we cannot represent it exactly with
        SipApproximation, since that requires CRPIX within its bbox.
        """
        bbox = Box2I(Point2I(0, 0), Extent2I(2048, 4176))
        target = self.make_wcs(
            {
                "AP_0_1": -1.4877747209973921e-05,
                "AP_0_2": 2.595831425638402e-09,
                "AP_0_3": -3.538473304544541e-13,
                "AP_0_4": -1.7364255070987416e-17,
                "AP_0_5": 4.158354613624071e-21,
                "AP_0_6": 6.031504872540217e-26,
                "AP_0_7": -1.8843528197634618e-29,
                "AP_0_8": -4.411702853182809e-35,
                "AP_0_9": 2.998016558839638e-38,
                "AP_1_0": 2.1597763415259763e-05,
                "AP_1_1": -1.7037264968756363e-08,
                "AP_1_2": 1.0563437512655601e-10,
                "AP_1_3": 5.513447366302908e-17,
                "AP_1_4": 4.1396908622107236e-20,
                "AP_1_5": -5.300207420437093e-25,
                "AP_1_6": 1.1459136839582329e-28,
                "AP_1_7": 1.0114481756477905e-33,
                "AP_1_8": 5.688355826686257e-38,
                "AP_2_0": 2.9292904845913305e-09,
                "AP_2_1": -2.31720223718549e-13,
                "AP_2_2": -7.483299584011452e-17,
                "AP_2_3": -1.0888257447773327e-21,
                "AP_2_4": 5.387989056591254e-25,
                "AP_2_5": 1.644007231402398e-29,
                "AP_2_6": -1.0830432046814581e-33,
                "AP_2_7": -3.804272720395786e-38,
                "AP_3_0": 1.0448304714804088e-10,
                "AP_3_1": 2.6347274534827444e-17,
                "AP_3_2": 9.623865277012687e-20,
                "AP_3_3": -2.8543130699430308e-25,
                "AP_3_4": 3.0929382240412716e-28,
                "AP_3_5": 8.14501749342599e-34,
                "AP_3_6": 2.2629352809700625e-37,
                "AP_4_0": -1.2664081335530406e-17,
                "AP_4_1": 8.338516492121341e-22,
                "AP_4_2": 4.683815141227516e-25,
                "AP_4_3": 3.8297278086295593e-29,
                "AP_4_4": -1.0848140289090295e-33,
                "AP_4_5": -1.6055236700259204e-37,
                "AP_5_0": 5.923796465250516e-20,
                "AP_5_1": -3.0109528249623673e-25,
                "AP_5_2": 2.07651550777521e-28,
                "AP_5_3": -3.3488672921196585e-34,
                "AP_5_4": 5.371228109055884e-37,
                "AP_6_0": 3.0606077273393017e-26,
                "AP_6_1": -7.470520624005137e-30,
                "AP_6_2": -7.924054929029662e-34,
                "AP_6_3": -8.993856894381261e-38,
                "AP_7_0": 2.8965736955513715e-29,
                "AP_7_1": 6.805599512576653e-34,
                "AP_7_2": 5.5526197352423534e-37,
                "AP_8_0": 4.496329614076484e-35,
                "AP_8_1": 2.915048166495553e-38,
                "AP_9_0": 1.9036813587133663e-37,
                "AP_ORDER": 9,
                "A_0_2": -1.6811953691767574e-09,
                "A_0_3": 1.0433743662634e-12,
                "A_0_4": -6.8711691147178074e-18,
                "A_0_5": -1.2339761900405834e-20,
                "A_0_6": 9.361497711139168e-26,
                "A_0_7": 5.447712238518265e-29,
                "A_0_8": -2.283455665486541e-34,
                "A_0_9": -8.148729256556932e-38,
                "A_1_1": 1.3292850389050882e-09,
                "A_1_2": -1.0511955221299713e-10,
                "A_1_3": 2.0198006554663002e-16,
                "A_1_4": -2.2187859071190358e-20,
                "A_1_5": -9.096711215990621e-25,
                "A_1_6": -1.0779518715344257e-29,
                "A_1_7": 1.3704000006676665e-33,
                "A_1_8": -7.427824514345201e-38,
                "A_2_0": 6.368561442632054e-09,
                "A_2_1": -2.4600357643424362e-12,
                "A_2_2": -2.435654620248234e-17,
                "A_2_3": 3.547267533946869e-20,
                "A_2_4": -6.487365679719972e-26,
                "A_2_5": -1.521351230094737e-28,
                "A_2_6": 2.1227383264900538e-34,
                "A_2_7": 2.0108583869008413e-37,
                "A_3_0": -1.043925630542753e-10,
                "A_3_1": 1.75976184042504e-16,
                "A_3_2": -3.2500931028178183e-20,
                "A_3_3": -1.921035718735496e-24,
                "A_3_4": -1.6197076806807766e-28,
                "A_3_5": 4.657442220122837e-33,
                "A_3_6": -4.1544398563283404e-38,
                "A_4_0": -1.572172681976455e-16,
                "A_4_1": 4.101331132363016e-20,
                "A_4_2": 6.1547414425568145e-25,
                "A_4_3": -3.9664912422773684e-28,
                "A_4_4": -9.219688587239831e-34,
                "A_4_5": 8.961182174237393e-37,
                "A_5_0": -3.1263190379608905e-20,
                "A_5_1": -6.744938189499249e-25,
                "A_5_2": -1.6066918861635898e-28,
                "A_5_3": 4.500410209244365e-33,
                "A_5_4": 1.6023740299725829e-37,
                "A_6_0": 9.442291591849256e-25,
                "A_6_1": -1.9578930866414858e-28,
                "A_6_2": -1.9703616513681726e-33,
                "A_6_3": 9.254059979828214e-37,
                "A_7_0": 2.857311310929313e-29,
                "A_7_1": 8.203568899883087e-34,
                "A_7_2": 8.581372417066432e-39,
                "A_8_0": -1.8039031898052798e-33,
                "A_8_1": 2.898481891491671e-37,
                "A_9_0": -1.4089135319319912e-37,
                "A_ORDER": 9,
                "BP_0_1": 1.3269406378579873e-05,
                "BP_0_2": -2.4020130376277656e-08,
                "BP_0_3": 1.0578604850716356e-10,
                "BP_0_4": 1.2705099582590507e-17,
                "BP_0_5": 4.374702590256722e-20,
                "BP_0_6": -1.1904361273665637e-25,
                "BP_0_7": 9.365376813229858e-29,
                "BP_0_8": 1.0249348542687937e-35,
                "BP_0_9": 9.913174239204098e-38,
                "BP_1_0": 5.3154970941211545e-06,
                "BP_1_1": 1.338585221404198e-09,
                "BP_1_2": -4.403323203838703e-13,
                "BP_1_3": -4.927675412191317e-17,
                "BP_1_4": 4.2056599337502155e-21,
                "BP_1_5": 4.347494937639946e-25,
                "BP_1_6": -1.7016369393589545e-29,
                "BP_1_7": -9.250712710300761e-34,
                "BP_1_8": 2.903862291589102e-38,
                "BP_2_0": -8.361037592783e-09,
                "BP_2_1": 1.0557933487998814e-10,
                "BP_2_2": 6.851578459014938e-17,
                "BP_2_3": 9.166056602172931e-20,
                "BP_2_4": -5.588794654793661e-25,
                "BP_2_5": 2.799146126748001e-28,
                "BP_2_6": 4.736465967290488e-34,
                "BP_2_7": 3.479972936276621e-37,
                "BP_3_0": 1.9579873417948577e-13,
                "BP_3_1": -3.4635677527711934e-18,
                "BP_3_2": 7.39087279804992e-21,
                "BP_3_3": 1.7492745475725066e-25,
                "BP_3_4": -4.035333126173646e-29,
                "BP_3_5": -1.3361684693427232e-33,
                "BP_3_6": 4.955950912353302e-38,
                "BP_4_0": 8.209637950392852e-18,
                "BP_4_1": 5.1024702870142e-20,
                "BP_4_2": -6.8032763060891895e-25,
                "BP_4_3": 2.1044072099103315e-28,
                "BP_4_4": 1.6036540498846812e-33,
                "BP_4_5": 7.0605046394838895e-37,
                "BP_5_0": -1.2718164521958228e-21,
                "BP_5_1": 3.861294906642736e-26,
                "BP_5_2": -3.728708833310905e-29,
                "BP_5_3": 7.585835353752152e-34,
                "BP_5_4": 1.0442735673821763e-37,
                "BP_6_0": 1.315728787073803e-26,
                "BP_6_1": 5.121275456285683e-29,
                "BP_6_2": 1.3292445159690183e-33,
                "BP_6_3": 5.9934345037665075e-37,
                "BP_7_0": -2.0388122227546495e-30,
                "BP_7_1": -2.75093255434579e-34,
                "BP_7_2": 6.416444202667438e-38,
                "BP_8_0": -1.4895913066402677e-34,
                "BP_8_1": 1.6803705066965438e-37,
                "BP_9_0": 1.8813475916412184e-38,
                "BP_ORDER": 9,
                "B_0_2": 2.42424716719325e-08,
                "B_0_3": -1.0625210421541499e-10,
                "B_0_4": -3.734011267534742e-17,
                "B_0_5": -4.4320952447253e-21,
                "B_0_6": 2.136396320364695e-25,
                "B_0_7": -1.0261671476558508e-28,
                "B_0_8": -2.8687265716879407e-34,
                "B_0_9": 7.34451334399625e-38,
                "B_1_1": -2.4662102612928633e-09,
                "B_1_2": 4.6033704144968226e-14,
                "B_1_3": 6.575057893720466e-17,
                "B_1_4": 5.132957383456082e-22,
                "B_1_5": -4.354806940180001e-25,
                "B_1_6": -9.86162124079778e-30,
                "B_1_7": 7.692996166964473e-34,
                "B_1_8": 2.4662028685683337e-38,
                "B_2_0": 9.397728194428527e-09,
                "B_2_1": -1.0671833625668622e-10,
                "B_2_2": -5.398605316354907e-17,
                "B_2_3": -1.862909181494854e-21,
                "B_2_4": 4.0889634152600245e-25,
                "B_2_5": -3.6402639280175727e-28,
                "B_2_6": -6.9963627184091545e-34,
                "B_2_7": 4.295158728944756e-37,
                "B_3_0": -1.8786163554348966e-13,
                "B_3_1": 1.972512581046491e-17,
                "B_3_2": 3.6729496252874392e-22,
                "B_3_3": -3.3840481271540886e-25,
                "B_3_4": -6.797729495720485e-30,
                "B_3_5": 1.4493402427155075e-33,
                "B_3_6": 3.998788271456866e-38,
                "B_4_0": -3.7050392480384804e-17,
                "B_4_1": -3.1322715658144286e-22,
                "B_4_2": 4.180620258608441e-25,
                "B_4_3": -3.4512946689359176e-28,
                "B_4_4": -1.0034951748320315e-33,
                "B_4_5": 5.8604050242447385e-37,
                "B_5_0": -2.236401243933395e-22,
                "B_5_1": -9.814956405914982e-26,
                "B_5_2": -2.921874946735589e-30,
                "B_5_3": -2.8112248331124586e-34,
                "B_5_4": 1.1980683642446688e-38,
                "B_6_0": 1.6355097123858816e-25,
                "B_6_1": -1.2020157950352392e-28,
                "B_6_2": -8.155965356039375e-34,
                "B_6_3": 3.5708930267652927e-37,
                "B_7_0": 1.1594106652283982e-29,
                "B_7_1": 3.168975065824031e-34,
                "B_7_2": 1.1734898293897363e-39,
                "B_8_0": -2.184412963604129e-34,
                "B_8_1": 1.019604698867224e-37,
                "B_9_0": -3.408083556979094e-38,
                "B_ORDER": 9,
                "CD1_1": 4.034435112527688e-08,
                "CD1_2": -4.688657441390689e-05,
                "CD2_1": -4.687791920020537e-05,
                "CD2_2": -5.0493192130940185e-08,
                "CRPIX1": -5340.870233465987,
                "CRPIX2": 17440.391355863132,
                "CRVAL1": 150.11251452060122,
                "CRVAL2": 2.2004651419030177,
                "CTYPE1": "RA---TAN-SIP",
                "CTYPE2": "DEC--TAN-SIP",
                "CUNIT1": "deg",
                "CUNIT2": "deg",
                "NAXES1": bbox.width,
                "NAXES2": bbox.height,
                "NAXIS": 2,
                "RADESYS": "ICRS",
            }
        )
        approx = SipApproximation(target, Box2D(bbox), Extent2I(25, 25), 5)
        deltaSky, deltaPixel = approx.computeDeltas()
        self.assertLess(deltaSky, 1e-4 * arcseconds)
        self.assertLess(deltaPixel, 1e-3)
        approxWcs = approx.getWcs()
        self.compare_to_astropy(approxWcs, bbox, deltaSkyMax=1e-7 * u.arcsec, deltaPixelMax=1e-5)
        self.compare_to_astropy(
            target.copyWithFitsApproximation(approxWcs),
            bbox,
            deltaSkyMax=1e-4 * u.arcsec,
            deltaPixelMax=1e-3,
        )

    def test_jointcal(self):
        """'jointcal' data are from the jointcal solution of HSC visit=30600
        ccd=88 tract=16973. This doesn't have a direct TAN-SIP representation.
        """
        bbox = Box2I(Point2I(0, 0), Extent2I(2048, 4176))
        target = SkyWcs.readFits(
            os.path.join(os.path.abspath(os.path.dirname(__file__)), "data", "jointcal_wcs-0030600-088.fits")
        )
        approx = SipApproximation(target, Box2D(bbox), Extent2I(25, 25), 5)
        deltaSky, deltaPixel = approx.computeDeltas()
        self.assertLess(deltaSky, 1e-4 * arcseconds)
        self.assertLess(deltaPixel, 1e-3)
        approxWcs = approx.getWcs()
        self.compare_to_astropy(approxWcs, bbox, deltaSkyMax=1e-7 * u.arcsec, deltaPixelMax=1e-5)
        self.compare_to_astropy(
            target.copyWithFitsApproximation(approxWcs),
            bbox,
            deltaSkyMax=1e-4 * u.arcsec,
            deltaPixelMax=1e-3,
        )


class MemoryTester(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
