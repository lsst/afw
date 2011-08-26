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
 
/**
 * @file distortion.cc
 * @author Steve Bickerton
 * @brief Verify that distortion roundtrips correctly after undistort
 *
 */
#include <stdio.h>
#include <iostream>
#include <string>

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE Distort

#include "boost/test/unit_test.hpp"
#include "boost/test/floating_point_comparison.hpp"

#include "lsst/afw/cameraGeom/Distortion.h"
#include "lsst/afw/geom/Point.h"

using namespace std;
namespace cameraGeom = lsst::afw::cameraGeom;
namespace afwGeom  = lsst::afw::geom;


BOOST_AUTO_TEST_CASE(roundTrip) {
    
    double x = 1.0;
    double y = 1.0;

    afwGeom::Point2D p(x, y);
    afwGeom::Point2D pDist; // distorted
    afwGeom::Point2D pp;    // undistorted ... roundtrip

    
    // try NullDistortion
    cameraGeom::NullDistortion ndist;
    pDist = ndist.distort(p);
    pp    = ndist.undistort(pDist);

    printf("%.12f %.12f\n", p.getX(),     p.getY());
    printf("%.12f %.12f\n", pDist.getX(), pDist.getY());
    printf("%.12f %.12f\n", pp.getX(),    pp.getY());

    BOOST_CHECK_EQUAL(pp.getX(), p.getX());
    BOOST_CHECK_EQUAL(pp.getY(), p.getY());

    
    // try RadialPolyDistortion
    std::vector<double> coeffs;
    coeffs.push_back(1.0);
    coeffs.push_back(1.1e-3);
    coeffs.push_back(2.2e-6);
    coeffs.push_back(3.3e-9);
    coeffs.push_back(4.4e-12);
    coeffs.push_back(5.5e-15);
    
    cameraGeom::RadialPolyDistortion rdist(coeffs);
    pDist = rdist.distort(p);
    pp    = rdist.undistort(pDist);
    
    printf("%.12f %.12f\n", p.getX(),     p.getY());
    printf("%.12f %.12f\n", pDist.getX(), pDist.getY());
    printf("%.12f %.12f\n", pp.getX(),    pp.getY());

    BOOST_CHECK_CLOSE(pp.getX(), p.getX(), 1.0e-7);
    BOOST_CHECK_CLOSE(pp.getY(), p.getY(), 1.0e-7);
    
}

