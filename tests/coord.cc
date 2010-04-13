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

#include "lsst/afw/coord.h"
#include "lsst/afw/geom/Point.h"

using namespace std;
namespace afwCoord = lsst::afw::coord;
namespace afwGeom  = lsst::afw::geom;

BOOST_AUTO_TEST_CASE(dmsToDecimal) {
    
    std::string ra = "10:00:00.00";
    std::string dec = "-02:30:00.00";
    double raDeg = afwCoord::hmsStringToDegrees(ra);
    double decDeg = afwCoord::dmsStringToDegrees(dec);
    
    BOOST_CHECK_EQUAL(raDeg, 150.0);
    BOOST_CHECK_EQUAL(decDeg, -2.5);
    
    // make sure the rounding issue works (ie. 59.998 rounds to 00, not 60 sec)
    raDeg -= 0.000001;
    std::string raStr = afwCoord::degreesToHmsString(raDeg);
    BOOST_CHECK_EQUAL(raStr, ra);

}


BOOST_AUTO_TEST_CASE(eclipticConversion) {

    // Pollux
    std::string alpha = "07:45:18.946";
    std::string delta = "28:01:34.26";
    double lamb0 = afwCoord::degToRad*113.215629;
    double beta0 = afwCoord::degToRad*6.684170;
    
    afwCoord::Fk5Coord polluxEqu(alpha, delta);
    afwCoord::EclipticCoord polluxEcl = polluxEqu.toEcliptic();
    afwCoord::Fk5Coord fk5 = polluxEcl.toFk5();
    afwGeom::Point2D p     = polluxEcl.getPosition(afwCoord::RADIANS);
    double lamb = p.getX(), beta = p.getY();
    BOOST_CHECK_CLOSE(lamb, lamb0, 1.0e-6);
    BOOST_CHECK_CLOSE(beta, beta0, 1.0e-6);

    afwCoord::Coord::Ptr test = afwCoord::makeCoord(afwCoord::makeCoordEnum("FK5"), lamb0, beta0);

    BOOST_CHECK_EQUAL(test->getLongitude(afwCoord::RADIANS), lamb0);
}
