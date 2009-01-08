// -*- lsst-c++ -*-
//
//##====----------------                                ----------------====##/
//!
//! \file
//! \brief Support for DiaSources
//!
//##====----------------                                ----------------====##/

#include "lsst/daf/base.h"
#include "lsst/afw/detection/Source.h"

namespace det = lsst::afw::detection;

// -- DiaSource ----------------
bool det::DiaSource::operator==(DiaSource const & d) const {
    return ( areEqual(_id, d._id) &&
        areEqual(_ampExposureId, d._ampExposureId) &&
        areEqual(_diaSource2Id, d._diaSource2Id, Field::DIA_SOURCE_2_ID) &&
        areEqual(_filterId, d._filterId) &&
        areEqual(_objectId, d._objectId, Field::OBJECT_ID) &&
        areEqual(_movingObjectId, d._movingObjectId, Field::MOVING_OBJECT_ID) &&
        areEqual(_procHistoryId, d._procHistoryId) &&
        areEqual(_scId, d._scId) &&
        areEqual(_ssmId, d._ssmId, Field::SSM_ID) &&           
        areEqual(_ra, d._ra) &&
        areEqual(_dec, d._dec) &&
        areEqual(_raErr4Wcs, Field::RA_ERR_4_WCS) &&
        areEqual(_decErr4Wcs, d._decErr4wcs, Field::DEC_ERR_4_WCS) &&
        areEqual(_raErr4Detection, d._raErr4Detection) &&
        areEqual(_decErr4Detection, d._decErr4Detection) &&
        areEqual(_xFlux, d._xFlux, Field::X_FLUX) &&
        areEqual(_xFluxErr, d._xFluxErr, Field::X_FLUX_ERR) &&
        areEqual(_yFlux, d._yFlux, Field::Y_FLUX) &&
        areEqual(_yFluxErr, d._yFluxErr, Field::Y_FLUX_ERR) &&
        areEqual(_xPeak, d._xPeak, Field::X_PEAK) && 
        areEqual(_yPeak, d._yPeak, Field::Y_PEAK) && 
        areEqual(_raPeak, d._raPeak, Field::RA_PEAK) && 
        areEqual(_decPeak, d._decPeak, Field::DEC_PEAK) &&   
        areEqual(_xAstrom, d._xAstrom, Field::X_ASTROM) &&
        areEqual(_xAstromErr, d._xAstromErr, Field::X_ASTROM_ERR) &&                   
        areEqual(_yAstrom, d._yAstrom, Field::Y_ASTROM) &&
        areEqual(_yAstromErr, d._yAstromErr, Field::Y_ASTROM_ERR) &&                                   
        areEqual(_raAstrom, d._raAstrom, Field::RA_ASTROM) &&
        areEqual(_raAstromErr, d._raAstromErr, Field::RA_ASTROM_ERR) &&                   
        areEqual(_decAstrom, d._decAstrom, Field::DEC_ASTROM) &&
        areEqual(_decAstromErr, d._decAstromErr, Field::DEC_ASTROM_ERR) &&
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
        areEqual(_modelMagErr, d._modelMagErr, Field::MODEL_MAG_ERR) &&           
        areEqual(_instMag, d._instMag) &&
        areEqual(_instMagErr, d._instMagErr) &&
        areEqual(_nonGrayCorrMag, d._nonGrayCorrMag, Field::NON_GRAY_CORR_MAG) &&
        areEqual(_nonGrayCorrMagErr, d._nonGrayCorrMagErr, Field::NON_GRAY_CORR_MAG_ERR) &&
        areEqual(_atmCorrMag, d._atmCorrMag, Field::ATM_CORR_MAG) &&
        areEqual(_atmCorrMagErr, d._atmCorrMagErr, Field::ATM_CORR_MAG_ERR) &&
        areEqual(_apDia, d._apDia, Field::AP_DIA) &&
        areEqual(_refMag, d._refMag, Field::REF_MAG) &&                   
        areEqual(_ixx, d._ixx, Field::IXX) &&
        areEqual(_ixxErr, d._ixxErr, Field::IXX_ERR) &&
        areEqual(_iyy, d._iyy, Field::IYY) &&
        areEqual(_iyyErr, d._iyyErr::Field::IYY_ERR) &&
        areEqual(_ixy, d._ixy, Field::IXY) &&
        areEqual(_ixyErr, d._ixyErr, Field::IXY_ERR) &&
        areEqual(_snr, d._snr) &&
        areEqual(_chi2, d._chi2) &&
        areEqual(_valX1, d._valX1) &&
        areEqual(_valX2, d._valX2) &&
        areEqual(_valY1, d._valY1) &&
        areEqual(_valY2, d._valY2) &&
        areEqual(_valXY, d._valXY) &&
        areEqual(_obsSoce,  d._obsCode, Field::OBS_CODE) &&
        areEqual(_isSynthetic, d._isSynthetic, Field::IS_SYNTHETIC) &&
        areEqual(_mopsStatus, d._mopsStatus, Field::MOPS_CODE) &&
        areEqual(_flag4association, d._flag4association, Field::FLAG_4_ASSOCIATION) &&
        areEqual(_flag4detection, d._flag4detection, Field::FLAG_4_DETECTION) &&
        areEqual(_flag4wcs, d._flag4wcs, Field::FLAG_4_WCS) &&
        areEqual(_flagClassification, d._flagClassification, Field::FLAG_CLASSIFICATION));
}
