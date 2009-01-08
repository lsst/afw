// -*- lsst-c++ -*-
//
//##====----------------                                ----------------====##/
//!
//! \file
//! \brief Support for Sources
//!
//##====----------------                                ----------------====##/

#include "lsst/daf/base.h"
#include "lsst/afw/detection/Source.h"
#include "lsst/pex/exceptions/Runtime.h"

namespace detection = lsst::afw::detection;
namespace detail = lsst::afw::detection::source_detail;

bool detection::Source::operator==(Source const & d) const {
    return ( areEqual(_id, d._id) &&
        areEqual(_ampExposureId, d._ampExposureId, detail::AMP_EXPOSURE_ID) &&
        areEqual(_filterId, d._filterId) &&
        areEqual(_objectId, d._objectId, detail::OBJECT_ID) &&
        areEqual(_movingObjectId, d._movingObjectId, detail::MOVING_OBJECT_ID) &&
        areEqual(_procHistoryId, d._procHistoryId) &&
        areEqual(_ra, d._ra) &&
        areEqual(_dec, d._dec) &&
        areEqual(_raErr4wcs, d._raErr4wcs) &&
        areEqual(_decErr4wcs, d._decErr4wcs) &&
        areEqual(_raErr4detection, d._raErr4detection, detail::RA_ERR_4_DETECTION) &&
        areEqual(_decErr4detection, d._decErr4detection, detail::DEC_ERR_4_DETECTION) &&
        areEqual(_xFlux, d._xFlux, detail::X_FLUX) &&
        areEqual(_xFluxErr, d._xFluxErr, detail::X_FLUX_ERR) &&
        areEqual(_yFlux, d._yFlux, detail::Y_FLUX) &&
        areEqual(_yFluxErr, d._yFluxErr, detail::Y_FLUX_ERR) &&
        areEqual(_xPeak, d._xPeak, detail::X_PEAK) && 
        areEqual(_yPeak, d._yPeak, detail::Y_PEAK) && 
        areEqual(_raPeak, d._raPeak, detail::RA_PEAK) && 
        areEqual(_decPeak, d._decPeak, detail::DEC_PEAK) &&   
        areEqual(_xAstrom, d._xAstrom, detail::X_ASTROM) &&
        areEqual(_xAstromErr, d._xAstromErr, detail::X_ASTROM_ERR) &&                   
        areEqual(_yAstrom, d._yAstrom, detail::Y_ASTROM) &&
        areEqual(_yAstromErr, d._yAstromErr, detail::Y_ASTROM_ERR) &&                                   
        areEqual(_raAstrom, d._raAstrom, detail::RA_ASTROM) &&
        areEqual(_raAstromErr, d._raAstromErr, detail::RA_ASTROM_ERR) &&                   
        areEqual(_decAstrom, d._decAstrom, detail::DEC_ASTROM) &&
        areEqual(_decAstromErr, d._decAstromErr, detail::DEC_ASTROM_ERR) &&
        areEqual(_taiMidPoint, d._taiMidPoint) &&
        areEqual(_taiRange, d._taiRange, detail::TAI_RANGE) &&
        areEqual(_fwhmA, d._fwhmA) &&
        areEqual(_fwhmB, d._fwhmB) &&
        areEqual(_fwhmTheta, d._fwhmTheta) &&
        areEqual(_psfMag, d._psfMag) &&
        areEqual(_psfMagErr, d._psfMagErr) &&
        areEqual(_apMag, d._apMag) &&
        areEqual(_apMagErr, d._apMagErr) &&
        areEqual(_modelMag, d._modelMag) &&
        areEqual(_modelMagErr, d._modelMagErr) &&
        areEqual(_petroMag, d._petroMag, detail::PETRO_MAG) &&
        areEqual(_petroMagErr, d._petroMagErr, detail::PETRO_MAG_ERR) &&             
        areEqual(_instMag, d._instMag) &&
        areEqual(_instMagErr, d._instMagErr) &&
        areEqual(_nonGrayCorrMag, d._nonGrayCorrMag, detail::NON_GRAY_CORR_MAG) &&
        areEqual(_nonGrayCorrMagErr, d._nonGrayCorrMagErr, detail::NON_GRAY_CORR_MAG_ERR) &&
        areEqual(_atmCorrMag, d._atmCorrMag, detail::ATM_CORR_MAG) &&
        areEqual(_atmCorrMagErr, d._atmCorrMagErr, detail::ATM_CORR_MAG_ERR) &&
        areEqual(_apDia, d._apDia, detail::AP_DIA) &&
        areEqual(_snr, d._snr) &&
        areEqual(_chi2, d._chi2) &&
        areEqual(_sky, d._sky, detail::SKY) &&
        areEqual(_skyErr, d._skyErr, detail::SKY_ERR) &&
        areEqual(_flag4association, d._flag4association, detail::FLAG_4_ASSOCIATION) &&
        areEqual(_flag4detection, d._flag4detection, detail::FLAG_4_DETECTION) &&
        areEqual(_flag4wcs, d._flag4wcs, detail::FLAG_4_WCS));
}
