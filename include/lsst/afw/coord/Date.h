// -*- lsst-c++ -*-
#if !defined(LSST_AFW_COORD_DATE_H)
#define LSST_AFW_COORD_DATE_H
/**
 * @file Date.h
 * @brief Functions to handle dates and times
 * @ingroup afw
 * @author Steve Bickerton
 *
 *
 */ 

#include "boost/tuple/tuple.hpp"

namespace lsst {
namespace afw {    
namespace coord {


typedef boost::tuple<int, int, int, int, int, double> CalendarDate;

/**
 * @class Date
 * @brief Handle dates and times
 *
 */
class Date {
public:
    
    enum DateForm { MJD, JD, EPOCH, NDATEFORM };

    Date(int Y, int M, int D, int H=0, int min=0, double S=0.0) {
        _jd = _calendarToJd(Y, M, D, H, min, S);
    }
    Date(CalendarDate const date) {
        _jd = _calendarToJd(date);
    }
    Date(double const date, DateForm const dateform);
    
    //std::string getDateStr(std::string fmt = "%Y-%m-%d %H-%M-%S") {}
    double getUnixSeconds() { return 1; }
    double getMjd()         { return _jdToMjd(_jd); }
    double getJd()          { return _jd; }
    double getEpoch()       { return _jdToEpoch(_jd); }

 private:

    double _calendarToJd(int Y, int M, int D, int H=0, int m=0, double S=0);
    double _calendarToJd(CalendarDate calDate);
    CalendarDate _jdToCalendar(double jd);
    double _epochToJd(double epoch);
    double _jdToEpoch(double jd);
    double _jdToMjd(double jd);
    double _mjdToJd(double mjd);
    
    double _jd;

};

double const JD2000 = 2451544.50;

}}}

#endif
