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
#include <cmath>

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE Distort

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-variable"
#include "boost/test/unit_test.hpp"
#pragma clang diagnostic pop
#include "boost/test/floating_point_comparison.hpp"

#include "lsst/afw/cameraGeom.h"

#include "lsst/afw/geom/Point.h"
#include "lsst/afw/geom/ellipses/Quadrupole.h"
#include "lsst/afw/image.h"
#include "lsst/afw/geom.h"

using namespace std;
namespace cameraGeom = lsst::afw::cameraGeom;
namespace afwGeom    = lsst::afw::geom;
namespace afwImage   = lsst::afw::image;
namespace geomEllip  = lsst::afw::geom::ellipses;

BOOST_AUTO_TEST_CASE(roundTrip) {
    
    double x = 1.0;
    double y = 1.0;
    double ixx = 1.0;
    double iyy = 1.0;
    double ixy = 0.0;

    cameraGeom::Detector detector(cameraGeom::Id(1), false, 1.0);
    
    afwGeom::Point2D p(x, y);
    afwGeom::Point2D pDist; // distorted
    afwGeom::Point2D pp;    // undistorted ... roundtrip
    geomEllip::Quadrupole q(ixx, iyy, ixy);
    geomEllip::Quadrupole qDist;
    geomEllip::Quadrupole qq;
    
    // try NullDistortion
    cameraGeom::NullDistortion ndist;
    pDist = ndist.distort(p, detector);
    pp    = ndist.undistort(pDist, detector);

    printf("%.12f %.12f\n", p.getX(),     p.getY());
    printf("%.12f %.12f\n", pDist.getX(), pDist.getY());
    printf("%.12f %.12f\n", pp.getX(),    pp.getY());

    BOOST_CHECK_EQUAL(pp.getX(), p.getX());
    BOOST_CHECK_EQUAL(pp.getY(), p.getY());

    
    // try RadialPolyDistortion
    std::vector<double> coeffs;
    coeffs.push_back(0.0);
    coeffs.push_back(1.0);
    coeffs.push_back(1.1e-3);
    coeffs.push_back(2.2e-6);
    coeffs.push_back(3.3e-9);
    coeffs.push_back(4.4e-12);
    coeffs.push_back(5.5e-15);
    
    cameraGeom::RadialPolyDistortion rdist(coeffs);
    pDist = rdist.distort(p, detector);
    qDist = rdist.distort(p, q, detector);
    pp    = rdist.undistort(pDist, detector);
    qq    = rdist.undistort(pDist, qDist, detector);
    
    printf("r: %.12f %.12f\n", p.getX(),     p.getY());
    printf("r: %.12f %.12f\n", pDist.getX(), pDist.getY());
    printf("r: %.12f %.12f\n", pp.getX(),    pp.getY());

    printf("r: %.12f %.12f\n", q.getIxx(),     q.getIyy());
    printf("r: %.12f %.12f\n", qDist.getIxx(), qDist.getIyy());
    printf("r: %.12f %.12f\n", qq.getIxx(),    qq.getIyy());
    
    BOOST_CHECK_CLOSE(pp.getX(), p.getX(), 1.0e-7);
    BOOST_CHECK_CLOSE(pp.getY(), p.getY(), 1.0e-7);
    BOOST_CHECK_CLOSE(qq.getIxx(), q.getIxx(), 1.0e-7);
    BOOST_CHECK_CLOSE(qq.getIyy(), q.getIyy(), 1.0e-7);


    int nx = 31, ny = 31;
    float rad0 = 3.0;
    int x0 = 15, y0 = 15;
    afwGeom::Point2D p0(x0, y0);
    float cx0 = 300.0, cy0 = 500.0;
    afwGeom::Point2D cp0(cx0, cy0);

    afwImage::Image<float> img(nx, ny, 0);
    for (int i=0; i<ny; ++i) {
        for (int j=0; j<nx; ++j) {
            float ic = i - y0;
            float jc = j - x0;
            float r = sqrt(ic*ic + jc*jc);
            img(j, i) = 1.0*std::exp(-r*r/(2.0*rad0*rad0));

        }
    }
    
    afwImage::Image<float>::Ptr wimg = rdist.distort(cp0, img, detector);
    //wimg->writeFits("ccwimg.fits");
    
}

