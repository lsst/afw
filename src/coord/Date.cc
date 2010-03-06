// -*- lsst-c++ -*-
/**
 * @file Date.cc
 * @brief Provide functions to handle dates
 * @ingroup afw
 * @author Steve Bickerton
 *
 */
#include <sstream>
#include <cmath>

#include "lsst/pex/exceptions.h"
#include "boost/algorithm/string.hpp"
#include "boost/tuple/tuple.hpp"

#include "lsst/afw/coord/Date.h"

namespace coord = lsst::afw::coord;
namespace ex    = lsst::pex::exceptions;

/*
 * @brief A function to handle dates convert between formats
 *
 */
coord::Date::Date(double value, coord::Date::DateForm const dateform) {

    if (dateform == coord::Date::JD) {
        _jd = value;
    } else if (dateform == coord::Date::MJD) {
        _jd = _mjdToJd(value);
    } else if (dateform == coord::Date::EPOCH) {
        _jd = _epochToJd(value);
    } else {
        throw LSST_EXCEPT(ex::InvalidParameterException, "date form not defined.");
    }
}

double coord::Date::_calendarToJd(int Y, int M, int D, int H, int min, double S) {

    double const hPerDay = 24.0;
    double const minPerDay = hPerDay*60.0;
    double const sPerDay = minPerDay*60.0;

    if ( M <= 2 ) {
	Y -= 1;
	M += 12;
    }

    int A = int(Y/100);
    int B =  2 - A + int(A/4); 

    int y = 1582, m = 10; //, d = 4;
    if (Y < y || (Y == y && M < m) || (Y == y && M == m && D <= 4)) {
        B = 0;
    }

    double jd = static_cast<int>(365.25*(Y + 4716)) + static_cast<int>(30.6001*(M + 1)) + D + B - 1524.5;
    jd += H / hPerDay + min / minPerDay + S / sPerDay;

    return jd;
}

double coord::Date::_calendarToJd(coord::CalendarDate calDate) {

    int Y = calDate.get<0>();
    int M = calDate.get<1>();
    int D = calDate.get<2>();
    int H = calDate.get<3>();
    int min = calDate.get<4>();
    int S = calDate.get<5>();
    
    return _calendarToJd(Y, M, D, H, min, static_cast<double>(S));
}

coord::CalendarDate coord::Date::_jdToCalendar(double jd) {
    
    jd += 0.5;
    int z = static_cast<int>(jd);     // integer part
    double f = jd - z;                // decimal part

    int alpha = static_cast<int>( (z - 1867216.25)/36524.25 );
    int A = ( z < 2299161 ) ? z : z + 1 + alpha - static_cast<int>(alpha/4);

    int B = A + 1524;
    int C = static_cast<int>( (B - 122.1)/365.25 );
    int D = static_cast<int>( 365.25*C );
    int E = static_cast<int>( (B - D)/30.6001 );

    int mday  = B - D - int(30.6001*E) + f;
    int mon   = (E < 14) ? (E-1) : (E-13);
    int year  = (mon > 2)  ? (C-4716) : (C-4715);

    double hour = 24.0*f;
    int H = static_cast<int>(hour);
    double min = (hour - H)*60.0;
    int Min = static_cast<int>(min);
    double s = (min - Min)*60.0;

    return coord::CalendarDate(year, mon, mday, H, Min, s);
}

double coord::Date::_jdToEpoch(double jd) {
    return 2000.0 + (jd - coord::JD2000)/365.25;
}
double coord::Date::_epochToJd(double epoch) {
    return coord::JD2000 + (epoch - 2000.0)*365.25;
}

double coord::Date::_jdToMjd(double jd) {
    return jd - 2400000.5;
}
double coord::Date::_mjdToJd(double mjd) {
    return mjd + 2400000.5;
}


