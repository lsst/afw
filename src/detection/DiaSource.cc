// -*- lsst-c++ -*-
//
//##====----------------                                ----------------====##/
//!
//! \file
//! \brief Support for DiaSources
//!
//##====----------------                                ----------------====##/

#include "lsst/daf/base.h"
#include "lsst/afw/detection/DiaSource.h"

namespace det = lsst::afw::detection;
namespace detail = lsst::afw::detection::diasource_detail;


// -- DiaSource ----------------
bool det::DiaSource::operator==(DiaSource const & d) const {
    return ( areEqual(_id, d._id) &&
        areEqual(_ampExposureId, d._ampExposureId) &&
        areEqual(_diaSource2Id, d._diaSource2Id, detail::DIA_SOURCE_2_ID) &&
        areEqual(_filterId, d._filterId) &&
        areEqual(_objectId, d._objectId, detail::OBJECT_ID) &&
        areEqual(_movingObjectId, d._movingObjectId, detail::MOVING_OBJECT_ID) &&
        areEqual(_procHistoryId, d._procHistoryId) &&
        areEqual(_scId, d._scId) &&
        areEqual(_ssmId, d._ssmId, detail::SSM_ID) &&           
        areEqual(_ra, d._ra) &&
        areEqual(_dec, d._dec) &&
        areEqual(_raErr4wcs, d._raErr4wcs, detail::RA_ERR_4_WCS) &&
        areEqual(_decErr4wcs, d._decErr4wcs, detail::DEC_ERR_4_WCS) &&
        areEqual(_raErr4detection, d._raErr4detection) &&
        areEqual(_decErr4detection, d._decErr4detection) &&
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
        areEqual(_taiRange, d._taiRange) &&
        areEqual(_fwhmA, d._fwhmA) &&
        areEqual(_fwhmB, d._fwhmB) &&
        areEqual(_fwhmTheta, d._fwhmTheta) &&
        areEqual(_lengthDeg, d._lengthDeg) &&
        areEqual(_flux, d._flux) &&
        areEqual(_fluxErr, d._fluxErr) &&        
        areEqual(_psfMag, d._psfMag) &&
        areEqual(_psfMagErr, d._psfMagErr) &&
        areEqual(_apMag, d._apMag) &&
        areEqual(_apMagErr, d._apMagErr) &&
        areEqual(_modelMag, d._modelMag) &&
        areEqual(_modelMagErr, d._modelMagErr, detail::MODEL_MAG_ERR) &&           
        areEqual(_instMag, d._instMag) &&
        areEqual(_instMagErr, d._instMagErr) &&
        areEqual(_nonGrayCorrMag, d._nonGrayCorrMag, detail::NON_GRAY_CORR_MAG) &&
        areEqual(_nonGrayCorrMagErr, d._nonGrayCorrMagErr, detail::NON_GRAY_CORR_MAG_ERR) &&
        areEqual(_atmCorrMag, d._atmCorrMag, detail::ATM_CORR_MAG) &&
        areEqual(_atmCorrMagErr, d._atmCorrMagErr, detail::ATM_CORR_MAG_ERR) &&
        areEqual(_apDia, d._apDia, detail::AP_DIA) &&
        areEqual(_refMag, d._refMag, detail::REF_MAG) &&                   
        areEqual(_ixx, d._ixx, detail::IXX) &&
        areEqual(_ixxErr, d._ixxErr, detail::IXX_ERR) &&
        areEqual(_iyy, d._iyy, detail::IYY) &&
        areEqual(_iyyErr, d._iyyErr, detail::IYY_ERR) &&
        areEqual(_ixy, d._ixy, detail::IXY) &&
        areEqual(_ixyErr, d._ixyErr, detail::IXY_ERR) &&
        areEqual(_snr, d._snr) &&
        areEqual(_chi2, d._chi2) &&
        areEqual(_valX1, d._valX1) &&
        areEqual(_valX2, d._valX2) &&
        areEqual(_valY1, d._valY1) &&
        areEqual(_valY2, d._valY2) &&
        areEqual(_valXY, d._valXY) &&
        areEqual(_obsCode,  d._obsCode, detail::OBS_CODE) &&
        areEqual(_isSynthetic, d._isSynthetic, detail::IS_SYNTHETIC) &&
        areEqual(_mopsStatus, d._mopsStatus, detail::MOPS_STATUS) &&
        areEqual(_flag4association, d._flag4association, detail::FLAG_4_ASSOCIATION) &&
        areEqual(_flag4detection, d._flag4detection, detail::FLAG_4_DETECTION) &&
        areEqual(_flag4wcs, d._flag4wcs, detail::FLAG_4_WCS) &&
        areEqual(_flagClassification, d._flagClassification, detail::FLAG_CLASSIFICATION));
}
