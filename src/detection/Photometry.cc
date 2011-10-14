// -*- lsst-c++ -*-
#include "lsst/afw/detection/Photometry.h"
#include "lsst/afw/detection/Astrometry.h"
#include "lsst/afw/detection/Shape.h"

/// Output to stream os
std::ostream &lsst::afw::detection::Photometry::output(std::ostream &os ///< The stream to output to
                                ) const {
    os << getFlux();
    if (getFluxErr() >= 0) {
        os << "+-" << getFluxErr();
    }
    return os;
}

LSST_REGISTER_SERIALIZER(lsst::afw::detection::Photometry)
LSST_REGISTER_SERIALIZER(lsst::afw::detection::Astrometry)
LSST_REGISTER_SERIALIZER(lsst::afw::detection::Shape)
