// -*- lsst-c++ -*-
//
//##====----------------                                ----------------====##/
//
//! \file
//! \brief  Does Wcs class in Wcs.cc correctly transform pixel coords <--> ra/dec coords
//
//##====----------------                                ----------------====##/

#include <iostream>
#include <cmath>
#include <vector>

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE Wcs test

#include "boost/test/unit_test.hpp"
#include "boost/test/floating_point_comparison.hpp"

#include "Eigen/Core.h"

#include "lsst/afw/image/Image.h"
#include "lsst/afw/image/Wcs.h"
#include "lsst/afw/image/TanWcs.h"
#include "lsst/afw/math/Interpolate.h"
#include "lsst/afw/math/Background.h"

#include "lsst/daf/base/PropertySet.h"

namespace image = lsst::afw::image;
namespace geom = lsst::afw::geom;
typedef Eigen::Matrix2d matrixD;


BOOST_AUTO_TEST_CASE(constructors_test) { /* parasoft-suppress  LsstDm-3-2a LsstDm-3-4a LsstDm-4-6 LsstDm-5-25 "Boost non-Std" */
    geom::PointD crval = geom::makePointD(30.0, 80.9);
    geom::PointD crpix = geom::makePointD(127,127);
    matrixD CD(2,2);

    //An identity matrix
    CD(0,0) = CD(1,1) = 1;
    CD(1,0) = CD(0,1) = 0;


    image::Wcs wcs();

    image::Wcs wcs2(crval, crpix, CD);

    //Create a Wcs with sip polynomials.
    image::TanWcs wcs3(crval, crpix, CD, CD, CD, CD, CD);
}

//A trivially easy example of the linear constructor
BOOST_AUTO_TEST_CASE(linearConstructor) { /* parasoft-suppress  LsstDm-3-2a LsstDm-3-4a LsstDm-4-6 LsstDm-5-25 "Boost non-Std" */
    geom::PointD crval = geom::makePointD(0.,0.);
    geom::PointD crpix = geom::makePointD(8.,8.);
    
    matrixD CD;
    CD  << 1,0,0,1; //Identity matrix
    
    image::Wcs wcs(crval, crpix, CD);

    //Remember, Wcs puts the origin of the chip at 1,1 while LSST
    //puts it at 0,0. This means that your coordinate transformation
    //will be one off what you expect it to be.
    //That said, I'm disturbed about how high my tolerance has to be
    //for what should be a simple computation
    double expect=2.0;
    geom::PointD ad = wcs.pixelToSky(9,9)->getPosition();
    BOOST_CHECK_CLOSE(ad.getX(), expect, .05);
    BOOST_CHECK_CLOSE(ad.getY(), 2., .11);    
    
    geom::PointD xy = wcs.skyToPixel(2,2);
    BOOST_CHECK_CLOSE(xy.getX(), 9., .05);
    BOOST_CHECK_CLOSE(xy.getY(), 9., .05);    
}


//A more complicated example. These numbers are taken from a visual inspection
//of the field of the white dwarf GD66
BOOST_AUTO_TEST_CASE(radec_to_xy) { /* parasoft-suppress  LsstDm-3-2a LsstDm-3-4a LsstDm-4-6 LsstDm-5-25 "Boost non-Std" */
    geom::PointD crval = geom::makePointD(80.159679, 30.806568);
    geom::PointD crpix = geom::makePointD(891.500000, 893.500000);
    matrixD CD(2,2);

    CD(0,0) = -0.0002802350;
    CD(0,1) = -0.0000021800;
    CD(1,0) = -0.0000022507;
    CD(1,1) = 0.0002796878;

    image::Wcs wcs(crval, crpix, CD);

    //check the trivial case
    geom::PointD xy = wcs.skyToPixel(80.159679, 30.80656);
    BOOST_CHECK_CLOSE(xy.getX(), 890.5, .1);
    BOOST_CHECK_CLOSE(xy.getY(), 892.5, .1);  
        
    xy = wcs.skyToPixel(80.258354, +30.810147);
    BOOST_CHECK_CLOSE(xy.getX(), 588., .1);
    BOOST_CHECK_CLOSE(xy.getY(), 903., .1);

    xy = wcs.skyToPixel(80.382829, +31.0287389);
    BOOST_CHECK_CLOSE(xy.getX(), 202., .1);
    BOOST_CHECK_CLOSE(xy.getY(), 1682., .1);

    xy = wcs.skyToPixel(79.900717, +31.0046556);
    BOOST_CHECK_CLOSE(xy.getX(), 1677., .1);
    BOOST_CHECK_CLOSE(xy.getY(), 1608., .1);

    xy = wcs.skyToPixel(79.987550, +30.6272333);
    BOOST_CHECK_CLOSE(xy.getX(), 1424., .1);
    BOOST_CHECK_CLOSE(xy.getY(), 256., .1);


}
    

BOOST_AUTO_TEST_CASE(xy_to_radec) { /* parasoft-suppress  LsstDm-3-2a LsstDm-3-4a LsstDm-4-6 LsstDm-5-25 "Boost non-Std" */
    geom::PointD crval = geom::makePointD(80.159679, 30.806568);
    geom::PointD crpix = geom::makePointD(891.500000, 893.500000);
    matrixD CD(2,2);

    CD(0,0) = -0.0002802350;
    CD(0,1) = -0.0000021800;
    CD(1,0) = -0.0000022507;
    CD(1,1) = 0.0002796878;

    image::Wcs wcs(crval, crpix, CD);

    //check the trivial case
    geom::PointD ad = wcs.pixelToSky(890.5, 892.5)->getPosition();
    BOOST_CHECK_CLOSE(ad.getX(), 80.15967 , 3e-5);  //2e-5 is <0.01 arcsec in ra
    BOOST_CHECK_CLOSE(ad.getY(), 30.80656 ,3e-5);  // 2e-5 is <0.1 arcsec in dec

    ad = wcs.pixelToSky(140., 116.)->getPosition();
    BOOST_CHECK_CLOSE(ad.getX(), 80.405963 , 3e-5);
    BOOST_CHECK_CLOSE(ad.getY(),  +30.5908500 , 3e-5);  

    ad = wcs.pixelToSky(396., 1481.)->getPosition();
    BOOST_CHECK_CLOSE(ad.getX(), 80.319804 , 3e-5);
    BOOST_CHECK_CLOSE(ad.getY(), +30.9721778 , 3e-5 );  

    ad = wcs.pixelToSky(1487., 1754.)->getPosition();
    BOOST_CHECK_CLOSE(ad.getX(), 79.962379 , 3e-5);
    BOOST_CHECK_CLOSE(ad.getY(), +31.0460250 , 3e-5);  

    ad = wcs.pixelToSky(1714., 186.)->getPosition();
    BOOST_CHECK_CLOSE(ad.getX(), 79.893342 , 3e-5);
    BOOST_CHECK_CLOSE(ad.getY(), +30.6068444 , 3e-5);  

    std::printf("T'end\n");
}

BOOST_AUTO_TEST_CASE(test_closure) { /* parasoft-suppress  LsstDm-3-2a LsstDm-3-4a LsstDm-4-6 LsstDm-5-25 "Boost non-Std" */
    geom::PointD crval = geom::makePointD(80.159679, 30.806568);
    geom::PointD crpix = geom::makePointD(890.500000, 892.500000);
    matrixD CD(2,2);

    CD(0,0) = -0.0002802350;
    CD(0,1) = -0.0000021800;
    CD(1,0) = -0.0000022507;
    CD(1,1) = 0.0002796878;

    image::Wcs wcs(crval, crpix, CD);

    double x = 251;
    double y = 910;
    geom::PointD xy = geom::makePointD(251., 910.);
    geom::PointD ad = wcs.pixelToSky(xy)->getPosition();
    BOOST_CHECK_CLOSE(wcs.skyToPixel(ad[0], ad[1]).getX(), x, 1e-6);
    BOOST_CHECK_CLOSE(wcs.skyToPixel(ad[0], ad[1]).getY(), y, 1e-6);
}


BOOST_AUTO_TEST_CASE(linearMatrix) { /* parasoft-suppress  LsstDm-3-2a LsstDm-3-4a LsstDm-4-6 LsstDm-5-25 "Boost non-Std" */
    
    geom::PointD crval = geom::makePointD(80.159679, 30.806568);
    geom::PointD crpix = geom::makePointD(891.500000, 893.500000);
    matrixD CD(2,2);

    CD(0,0) = -0.0002802350;
    CD(0,1) = -0.0000021800;
    CD(1,0) = -0.0000022507;
    CD(1,1) = 0.0002796878;

    image::Wcs wcs(crval, crpix, CD);
    
    matrixD M = wcs.getCDMatrix();
    BOOST_CHECK_CLOSE(CD(0,0), M(0,0), 1e-6);
    BOOST_CHECK_CLOSE(CD(0,1), M(0,1), 1e-6);
    BOOST_CHECK_CLOSE(CD(1,0), M(1,0), 1e-6);
    BOOST_CHECK_CLOSE(CD(1,1), M(1,1), 1e-6);
}


