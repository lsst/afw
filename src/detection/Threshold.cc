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
#include <cstdint>
#include <cmath>
#include <string>
#include <boost/format.hpp>
#include "lsst/pex/exceptions.h"
#include "lsst/log/Log.h"
#include "lsst/afw/math/Statistics.h"
#include "lsst/afw/image/Image.h"
#include "lsst/afw/image/MaskedImage.h"
#include "lsst/afw/detection/Threshold.h"

namespace image = lsst::afw::image;
namespace math = lsst::afw::math;

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
    } else if (typeStr.compare("pixel_stdev") == 0) {
        return Threshold::PIXEL_STDEV;
    } else {
        throw LSST_EXCEPT(
            lsst::pex::exceptions::InvalidParameterError,
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
            lsst::pex::exceptions::InvalidParameterError,
            (boost::format("Unsopported Threshold type: %d") % type).str()
        );
    }
}

double Threshold::getValue(const double param) const {
    switch (_type) {
      case STDEV:
        if (param <= 0) {
            throw LSST_EXCEPT(
                lsst::pex::exceptions::InvalidParameterError,
                (boost::format("St. dev. must be > 0: %g") % param).str()
            );
        }
        return _value*param;
      case VALUE:
      case BITMASK:
      case PIXEL_STDEV:
        return _value;
      case VARIANCE:
        if (param <= 0) {
            throw LSST_EXCEPT(
                lsst::pex::exceptions::InvalidParameterError,
                (boost::format("Variance must be > 0: %g") % param).str()
            );
        }
        return _value*std::sqrt(param);
      default:
        throw LSST_EXCEPT(
            lsst::pex::exceptions::InvalidParameterError,
            (boost::format("Unsupported type: %d") % _type).str()
        );
    }
}

template<typename ImageT>
double Threshold::getValue(ImageT const& image) const {
    double param = -1;                  // Parameter for getValue()
    if (_type == STDEV ||
        _type == VARIANCE) {
        math::Statistics stats = math::makeStatistics(image, math::STDEVCLIP);
        double const sd = stats.getValue(math::STDEVCLIP);

        LOGL_DEBUG("afw.detection.threshold", "St. Dev = %g", sd);

        if (_type == VARIANCE) {
            param = sd*sd;
        } else {
            param = sd;
        }
    }
    return getValue(param);
}

Threshold createThreshold(
    double const value,
    std::string const typeStr,
    bool const polarity
) {
    return Threshold(value, Threshold::parseTypeString(typeStr), polarity);
}


//
// Explicit instantiations
//
#define INSTANTIATE(TYPE) \
template double Threshold::getValue(image::TYPE<unsigned short> const&) const; \
template double Threshold::getValue(image::TYPE<int> const&) const; \
template double Threshold::getValue(image::TYPE<float> const&) const; \
template double Threshold::getValue(image::TYPE<double> const&) const; \
template double Threshold::getValue(image::TYPE<std::uint64_t> const&) const;

#ifndef DOXYGEN
INSTANTIATE(Image);
INSTANTIATE(MaskedImage);
#endif

}}}
