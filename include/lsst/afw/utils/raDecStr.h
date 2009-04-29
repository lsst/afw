// -*- lsst-c++ -*-
#ifndef RA_DEC_STR_H
#define RA_DEC_STR_H

#include <cstring>
#include <cmath>

#include "boost/format.hpp"
#include "boost/regex.hpp"
#include "boost/lexical_cast.hpp"

#include "lsst/afw/image/Image.h"
#include "lsst/afw/image/Utils.h"
#include "lsst/pex/exceptions.h"

namespace lsst { 
    namespace afw { 
        namespace utils {

    
std::string raToStr(double ra);
std::string decToStr(double dec);

std::string raDecToStr(double ra, double dec);
std::string raDecToStr(lsst::afw::image::PointD p);

double strToRa(const std::string &str, const std::string &sep ="[: ]");
double strToDec(const std::string &str, const std::string &sep="[: ]");
            lsst::afw::image::PointD strToRaDec(const std::string &str, const std::string &sep="[: ]");
        

}}}   //lsst::afw::utils                 

#endif

