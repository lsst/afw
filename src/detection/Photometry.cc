// -*- lsst-c++ -*-
#include "lsst/afw/detection/Photometry.h"

/// Output to stream os
std::ostream &lsst::afw::detection::Photometry::output(std::ostream &os ///< The stream to output to
                                ) const {
    os << getFlux();
    if (getFluxErr() >= 0) {
        os << "+-" << getFluxErr();
    }
    return os;
}
