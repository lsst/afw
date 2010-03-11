// -*- lsst-c++ -*-
/**
 * @file sex2dec.cc
 * @author Steve Bickerton
 * @brief An example executible which calls the example sex2dec code
 *
 */
#include <iostream>
#include <string>

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE Coord

#include "boost/test/unit_test.hpp"
#include "boost/test/floating_point_comparison.hpp"

#include "lsst/afw/coord/Coord.h"
#include "lsst/afw/geom/Point.h"

using namespace std;
namespace coord = lsst::afw::coord;
namespace geom  = lsst::afw::geom;

BOOST_AUTO_TEST_CASE(dmsToDecimal) {
    
    std::string ra = "10:00:00.00";
    std::string dec = "-02:30:00.00";
    double raDeg = coord::hmsStringToDegrees(ra);
    double decDeg = coord::dmsStringToDegrees(dec);
    
    BOOST_CHECK_EQUAL(raDeg, 10.0);
    BOOST_CHECK_EQUAL(decDeg, -2.5);
    
    // make sure the rounding issue works (ie. 59.998 rounds to 00, not 60 sec)
    raDeg -= 0.000001;
    std::string raStr = coord::degreesToHmsString(raDeg);
    BOOST_CHECK_EQUAL(raStr, ra);

    coord::Fk5Coord cel(raDeg, decDeg);
    cel.precess(2010.0);
    
}


BOOST_AUTO_TEST_CASE(eclipticConversion) {

    // Pollux
    std::string alpha = "07:45:18.946";
    std::string delta = "28:01:34.26";
    //double alpha = 116.328942;
    //double alpha = 28.026183;
    coord::Fk5Coord polluxEqu(alpha, delta);
    coord::EclipticCoord polluxEcl = polluxEqu.toEcliptic();
    coord::Fk5Coord fk5 = polluxEcl.toFk5();
    std::cout << "Pollux (ecl): " << polluxEcl.getLambdaDeg() << " " <<  polluxEcl.getBetaDeg() << std::endl;
    std::cout << "Pollux (equ): " << polluxEqu.getRaDeg() << " " <<  polluxEqu.getDecDeg() << std::endl;
    std::cout << "Pollux (fk5): " << fk5.getRaDeg() << " " <<  fk5.getDecDeg() << std::endl;

    geom::PointD p = polluxEqu.getPoint2D();
    std::cout << "PointD: " << p.getX() << " " << p.getY() << std::endl;
}
