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
#include <cmath>
#include <string>
#include <boost/format.hpp>
#include "lsst/pex/exceptions.h"
#include "lsst/afw/detection/Threshold.h"


namespace lsst {
namespace afw {
namespace detection {

Threshold::ThresholdType Threshold::parseTypeString(std::string const & typeStr) {
    if (typeStr.compare("bitmask") == 0) {
        return Threshold::BITMASK;           
    } else if (typeStr.compare("value") == 0) {
        return Threshold::VALUE;           
    } else if (typeStr.compare("stdev") == 0) {
        return Threshold::STDEV;
    } else if (typeStr.compare("variance") == 0) {
        return Threshold::VARIANCE;
    } else {
        throw LSST_EXCEPT(
            lsst::pex::exceptions::InvalidParameterException,
            (boost::format("Unsupported Threshold type: %s") % typeStr).str()
        );
    }    
}

std::string Threshold::getTypeString(ThresholdType const & type) {
    if (type == VALUE) {
        return "value";
    } else if (type == STDEV) {
        return "stdev";
    } else if (type == VARIANCE) {
        return "variance";
    } else {
        throw LSST_EXCEPT(
            lsst::pex::exceptions::InvalidParameterException,
            (boost::format("Unsopported Threshold type: %d") % type).str()
        );
    }
}

/**
 * return value of threshold, to be interpreted via type
 * @param param value of variance/stdev if needed
 * @return value of threshold
 */
double Threshold::getValue(const double param) const {
    switch (_type) {
      case STDEV:
        if (param <= 0) {
            throw LSST_EXCEPT(
                lsst::pex::exceptions::InvalidParameterException,
                (boost::format("St. dev. must be > 0: %g") % param).str()
            );
        }
        return _value*param;
      case VALUE:
      case BITMASK:
        return _value;
      case VARIANCE:
        if (param <= 0) {
            throw LSST_EXCEPT(
                lsst::pex::exceptions::InvalidParameterException,
                (boost::format("Variance must be > 0: %g") % param).str()
            );
        }
        return _value*std::sqrt(param);
      default:
        throw LSST_EXCEPT(
            lsst::pex::exceptions::InvalidParameterException,
            (boost::format("Unsupported type: %d") % _type).str()
        );
    }
}


/**
 * \brief Factory method for creating Threshold objects
 *
 * @param value value of threshold
 * @param typeStr string representation of a ThresholdType. This parameter is 
 *                optional. Allowed values are: "variance", "value", "stdev"
 * @param polarity If true detect positive objects, false for negative
 *
 * @return desired Threshold
 */
Threshold createThreshold(
    double const value,                  
    std::string const typeStr,
    bool const polarity
) {
    return Threshold(value, Threshold::parseTypeString(typeStr), polarity);
}

}}}
