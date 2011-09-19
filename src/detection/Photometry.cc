// -*- lsst-c++ -*-
#include "lsst/afw/detection/Photometry.h"

namespace lsst { namespace afw { namespace detection {

/// Output to stream os
std::ostream &Photometry::output(std::ostream &os ///< The stream to output to
                                ) const {
    os << getFlux();
    if (getFluxErr() >= 0) {
        os << "+-" << getFluxErr();
    }
    return os;
}


Photometry::Ptr Photometry::average() const {
    double sum = 0.0, sumWeight = 0.0;
    for (Photometry::const_iterator iter = begin(); iter != end(); ++iter) {
        Photometry::ConstPtr phot = *iter;
        if (!phot->empty()) {
            phot = phot->average();
        }
        double flux = phot->getFlux();
        double fluxErr = phot->getFluxErr();
        double weight = 1.0 / (fluxErr * fluxErr);
        sum += flux * weight;
        sumWeight += weight;
    }
    return Photometry::Ptr(new Photometry(sum / sumWeight, 1.0 / sumWeight));
}

}}} // namespace
