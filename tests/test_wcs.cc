// -*- lsst-c++ -*-

/*
 * LSST Data Management System
 * Copyright 2008, 2009, 2010 LSST Corporation.
 *
 * This product includes software developed by the
 * LSST Project (http://www.lsst.org/).
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the LSST License Statement and
 * the GNU General Public License along with this program.  If not,
 * see <http://www.lsstcorp.org/LegalNotices/>.
 */

//
//##====----------------                                ----------------====##/
// Does Wcs class in Wcs.cc correctly transform pixel coords <--> ra/dec coords
//
//##====----------------                                ----------------====##/

#include <iostream>
#include <cmath>
#include <vector>

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE Wcs test

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-variable"
#include "boost/test/unit_test.hpp"
#pragma clang diagnostic pop
#include "boost/test/floating_point_comparison.hpp"

#include "Eigen/Core"

#include "lsst/afw/image/Image.h"
#include "lsst/afw/image/Wcs.h"
#include "lsst/afw/image/TanWcs.h"
#include "lsst/afw/math/Interpolate.h"
#include "lsst/afw/math/Background.h"

#include "lsst/daf/base/PropertySet.h"

namespace image = lsst::afw::image;
namespace geom = lsst::afw::geom;
namespace afwGeom = lsst::afw::geom;
typedef Eigen::Matrix2d matrixD;

BOOST_AUTO_TEST_CASE(
        constructors_test) { /* parasoft-suppress  LsstDm-3-2a LsstDm-3-4a LsstDm-4-6 LsstDm-5-25 "Boost
                                non-Std" */
    geom::Point2D crval = geom::Point2D(30.0, 80.9);
    geom::Point2D crpix = geom::Point2D(127, 127);
    matrixD CD(2, 2);

    // An identity matrix
    CD(0, 0) = CD(1, 1) = 1;
    CD(1, 0) = CD(0, 1) = 0;

    image::Wcs wcs2(crval, crpix, CD);

    // Create a Wcs with sip polynomials.
    image::TanWcs wcs3(crval, crpix, CD, CD, CD, CD, CD);
}

BOOST_AUTO_TEST_CASE(
        comparison) { /* parasoft-suppress  LsstDm-3-2a LsstDm-3-4a LsstDm-4-6 LsstDm-5-25 "Boost non-Std" */
    geom::Point2D crval = geom::Point2D(30.0, 80.9);
    geom::Point2D crpix = geom::Point2D(127, 127);
    matrixD CD(2, 2);

    // An identity matrix
    CD(0, 0) = CD(1, 1) = 1;
    CD(1, 0) = CD(0, 1) = 0;

    image::Wcs plainWcs(crval, crpix, CD);
    image::TanWcs sipWcs(crval, crpix, CD);

    std::shared_ptr<image::Wcs> plainWcsCopy = plainWcs.clone();
    std::shared_ptr<image::Wcs> sipWcsCopy = sipWcs.clone();

    BOOST_CHECK(!sipWcs.hasDistortion());
    BOOST_CHECK(!sipWcsCopy->hasDistortion());

    BOOST_CHECK(plainWcs == *plainWcsCopy);
    BOOST_CHECK(*plainWcsCopy == plainWcs);
    BOOST_CHECK(sipWcs == *sipWcsCopy);
    BOOST_CHECK(*sipWcsCopy == sipWcs);

    image::TanWcs distortedWcs(crval, crpix, CD, CD, CD, CD, CD);
    std::shared_ptr<image::Wcs> distortedWcsCopy = distortedWcs.clone();

    BOOST_CHECK(distortedWcs.hasDistortion());
    BOOST_CHECK(distortedWcsCopy->hasDistortion());

    BOOST_CHECK(sipWcs != distortedWcs);
    BOOST_CHECK(distortedWcs != sipWcs);

    BOOST_CHECK(*sipWcsCopy != *distortedWcsCopy);
    BOOST_CHECK(*distortedWcsCopy != *sipWcsCopy);

    BOOST_CHECK(*distortedWcsCopy == distortedWcs);
    BOOST_CHECK(distortedWcs == *distortedWcsCopy);
}

// A trivially easy example of the linear constructor
BOOST_AUTO_TEST_CASE(
        linearConstructor) { /* parasoft-suppress  LsstDm-3-2a LsstDm-3-4a LsstDm-4-6 LsstDm-5-25 "Boost
                                non-Std" */
    geom::Point2D crval = geom::Point2D(0., 0.);
    geom::Point2D crpix = geom::Point2D(8., 8.);

    matrixD CD;
    CD << 1 / 3600., 0, 0, 1 / 3600.;

    image::Wcs wcs(crval, crpix, CD);

    double arcsecInDeg = 1 / 3600.;
    double tol = 1e-2;
    geom::Point2D ad = wcs.pixelToSky(9, 9)->getPosition();
    BOOST_CHECK_CLOSE(ad.getX(), arcsecInDeg, tol);
    BOOST_CHECK_CLOSE(ad.getY(), arcsecInDeg, tol);

    geom::Point2D xy = wcs.skyToPixel(1 * afwGeom::arcseconds, 1 * afwGeom::arcseconds);
    BOOST_CHECK_CLOSE(xy.getX(), 9., tol);
    BOOST_CHECK_CLOSE(xy.getY(), 9., tol);
}

// A more complicated example. These numbers are taken from a visual inspection
// of the field of the white dwarf GD66
BOOST_AUTO_TEST_CASE(
        radec_to_xy) { /* parasoft-suppress  LsstDm-3-2a LsstDm-3-4a LsstDm-4-6 LsstDm-5-25 "Boost non-Std" */
    geom::Point2D crval = geom::Point2D(80.159679, 30.806568);
    // geom::Point2D crpix = geom::Point2D(891.500000, 893.500000);
    geom::Point2D crpix = geom::Point2D(890.500000, 892.500000);
    matrixD CD(2, 2);

    CD(0, 0) = -0.0002802350;
    CD(0, 1) = -0.0000021800;
    CD(1, 0) = -0.0000022507;
    CD(1, 1) = 0.0002796878;

    image::Wcs wcs(crval, crpix, CD);

    // check the trivial case
    geom::Point2D xy = wcs.skyToPixel(80.159679 * afwGeom::degrees, 30.80656 * afwGeom::degrees);
    BOOST_CHECK_CLOSE(xy.getX(), 890.5, .1);
    BOOST_CHECK_CLOSE(xy.getY(), 892.5, .1);

    xy = wcs.skyToPixel(80.258354 * afwGeom::degrees, +30.810147 * afwGeom::degrees);
    BOOST_CHECK_CLOSE(xy.getX(), 588., .1);
    BOOST_CHECK_CLOSE(xy.getY(), 903., .1);

    xy = wcs.skyToPixel(80.382829 * afwGeom::degrees, +31.0287389 * afwGeom::degrees);
    BOOST_CHECK_CLOSE(xy.getX(), 202., .1);
    BOOST_CHECK_CLOSE(xy.getY(), 1682., .1);

    xy = wcs.skyToPixel(79.900717 * afwGeom::degrees, +31.0046556 * afwGeom::degrees);
    BOOST_CHECK_CLOSE(xy.getX(), 1677., .1);
    BOOST_CHECK_CLOSE(xy.getY(), 1608., .1);

    xy = wcs.skyToPixel(79.987550 * afwGeom::degrees, +30.6272333 * afwGeom::degrees);
    BOOST_CHECK_CLOSE(xy.getX(), 1424., .1);
    BOOST_CHECK_CLOSE(xy.getY(), 256., .1);
}

BOOST_AUTO_TEST_CASE(
        xy_to_radec) { /* parasoft-suppress  LsstDm-3-2a LsstDm-3-4a LsstDm-4-6 LsstDm-5-25 "Boost non-Std" */
    geom::Point2D crval = geom::Point2D(80.159679, 30.806568);
    geom::Point2D crpix = geom::Point2D(890.500000, 892.500000);
    matrixD CD(2, 2);

    CD(0, 0) = -0.0002802350;
    CD(0, 1) = -0.0000021800;
    CD(1, 0) = -0.0000022507;
    CD(1, 1) = 0.0002796878;

    image::Wcs wcs(crval, crpix, CD);

    // check the trivial case
    geom::Point2D ad = wcs.pixelToSky(890.5, 892.5)->getPosition();
    BOOST_CHECK_CLOSE(ad.getX(), 80.15967, 3e-5);  // 2e-5 is <0.01 arcsec in ra
    BOOST_CHECK_CLOSE(ad.getY(), 30.80656, 3e-5);  // 2e-5 is <0.1 arcsec in dec

    ad = wcs.pixelToSky(140., 116.)->getPosition();
    BOOST_CHECK_CLOSE(ad.getX(), 80.405963, 3e-5);
    BOOST_CHECK_CLOSE(ad.getY(), +30.5908500, 3e-5);

    ad = wcs.pixelToSky(396., 1481.)->getPosition();
    BOOST_CHECK_CLOSE(ad.getX(), 80.319804, 3e-5);
    BOOST_CHECK_CLOSE(ad.getY(), +30.9721778, 3e-5);

    ad = wcs.pixelToSky(1487., 1754.)->getPosition();
    BOOST_CHECK_CLOSE(ad.getX(), 79.962379, 3e-5);
    BOOST_CHECK_CLOSE(ad.getY(), +31.0460250, 3e-5);

    ad = wcs.pixelToSky(1714., 186.)->getPosition();
    BOOST_CHECK_CLOSE(ad.getX(), 79.893342, 3e-5);
    BOOST_CHECK_CLOSE(ad.getY(), +30.6068444, 3e-5);
}

BOOST_AUTO_TEST_CASE(test_closure) { /* parasoft-suppress  LsstDm-3-2a LsstDm-3-4a LsstDm-4-6 LsstDm-5-25
                                        "Boost non-Std" */
    geom::Point2D crval = geom::Point2D(80.159679, 30.806568);
    geom::Point2D crpix = geom::Point2D(890.500000, 892.500000);
    matrixD CD(2, 2);

    CD(0, 0) = -0.0002802350;
    CD(0, 1) = -0.0000021800;
    CD(1, 0) = -0.0000022507;
    CD(1, 1) = 0.0002796878;

    image::Wcs wcs(crval, crpix, CD);

    double x = 251;
    double y = 910;
    geom::Point2D xy = geom::Point2D(251., 910.);
    geom::Point2D ad = wcs.pixelToSky(xy)->getPosition();
    BOOST_CHECK_CLOSE(wcs.skyToPixel(ad[0] * afwGeom::degrees, ad[1] * afwGeom::degrees).getX(), x, 1e-6);
    BOOST_CHECK_CLOSE(wcs.skyToPixel(ad[0] * afwGeom::degrees, ad[1] * afwGeom::degrees).getY(), y, 1e-6);
}

BOOST_AUTO_TEST_CASE(linearMatrix) { /* parasoft-suppress  LsstDm-3-2a LsstDm-3-4a LsstDm-4-6 LsstDm-5-25
                                        "Boost non-Std" */

    geom::Point2D crval = geom::Point2D(80.159679, 30.806568);
    geom::Point2D crpix = geom::Point2D(891.500000, 893.500000);
    matrixD CD(2, 2);

    CD(0, 0) = -0.0002802350;
    CD(0, 1) = -0.0000021800;
    CD(1, 0) = -0.0000022507;
    CD(1, 1) = 0.0002796878;

    image::Wcs wcs(crval, crpix, CD);

    matrixD M = wcs.getCDMatrix();
    BOOST_CHECK_CLOSE(CD(0, 0), M(0, 0), 1e-6);
    BOOST_CHECK_CLOSE(CD(0, 1), M(0, 1), 1e-6);
    BOOST_CHECK_CLOSE(CD(1, 0), M(1, 0), 1e-6);
    BOOST_CHECK_CLOSE(CD(1, 1), M(1, 1), 1e-6);
}
