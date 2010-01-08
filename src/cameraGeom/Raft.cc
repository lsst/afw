/**
 * \file
 */
#include <algorithm>
#include "lsst/afw/cameraGeom/Raft.h"

namespace afwGeom = lsst::afw::geom;
namespace afwImage = lsst::afw::image;
namespace camGeom = lsst::afw::cameraGeom;

/// Return the pixel size, mm/pixel
double camGeom::Raft::getPixelSize() const {
    if (begin() == end()) {
        throw LSST_EXCEPT(lsst::pex::exceptions::OutOfRangeException,
                          (boost::format("DetectorMosaic with serial %|| has no Detectors") % getId()).str());
    }

    return (*begin())->getDetector()->getPixelSize();
}

