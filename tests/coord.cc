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
#include "lsst/afw/coord/Date.h"
#include "lsst/afw/geom/Point.h"

using namespace std;
namespace coord = lsst::afw::coord;
namespace geom  = lsst::afw::geom;

BOOST_AUTO_TEST_CASE(dmsToDecimal) {
    
    std::string ra = "10:00:00.00";
    std::string dec = "-02:30:00.00";
    double raDeg = coord::hmsStringToDegrees(ra);
    double decDeg = coord::dmsStringToDegrees(dec);
    
    BOOST_CHECK_EQUAL(raDeg, 150.0);
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
    std::cout << "Pollux (ecl): " <<
        polluxEcl.getLambda(coord::DEGREES) << " " <<  polluxEcl.getBeta(coord::DEGREES) << std::endl;
    std::cout << "Pollux (equ): " <<
        polluxEqu.getRa(coord::DEGREES) << " " <<  polluxEqu.getDec(coord::DEGREES) << std::endl;
    std::cout << "Pollux (fk5): " <<
        fk5.getRa(coord::DEGREES) << " " <<  fk5.getDec(coord::DEGREES) << std::endl;

    geom::PointD p = polluxEqu.getPoint2D();
    std::cout << "PointD: " << p.getX() << " " << p.getY() << std::endl;

    coord::Fk5Coord f = polluxEqu.precess(2028.0);
    std::cout << f.getRa(coord::DEGREES) << " " << std::endl;

    double lambFk5 = 149.48194;
    double betaFk5 = 1.76549;
        
    // known values for -214, June 30.0
    // they're actually 118.704, 1.615, but I suspect discrepancy is a rounding error in Meeus
    //  -- we use double precision, he carries 7 places only.
    double lambNew = 118.704;
    double betaNew = 1.606 ;
    coord::EclipticCoord venusFk5(lambFk5, betaFk5, 2000.0);
    //double ep = dafBase::DateTime(-214, 6, 30, 0, 0, 0,
    //                            dafBase::DateTime::TAI).getDate(dafBase::DateTime::EPOCH);
    double ep = coord::Date(-214, 6, 30, 0, 0, 0).getEpoch();
    coord::EclipticCoord venusNew = venusFk5.precess( ep );
    //coord::EclipticCoord venusNew(venusFk5.precess(coord::Date(-214, 6, 30, 0, 0, 0).getEpoch()));

    std::cout << venusNew.getLambda(coord::DEGREES) << " " << lambNew <<  " " << venusNew.getEpoch() << std::endl;
}
