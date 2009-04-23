#ifndef RA_DEC_STR_H
#define RA_DEC_STR_H

#include <string>
#include <cmath>

#include "boost/format.hpp"

//namespace lsst { namespace utils

std::string raToStr(double ra);
std::string decToStr(double dec);

std::string raDecToStr(double ra, double dec);
//string raDecToStr(lsst::afw::image::PointD p);

//}}   //Close the namespace
                 

#endif

