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

det::DiaSource::DiaSource()
	: _diaSource2Id(0), _scId(0), _ssmId(0),
      _lengthDeg(0.0),        
      _flux(0.0), _fluxErr(0.0),
      _refMag(0.0), 
      _ixx(0.0), _ixxErr(0.0),
      _iyy(0.0), _iyyErr(0.0),
      _ixy(0.0), _ixyErr(0.0),
      _valX1(0.0), _valX2(0.0),
      _valY1(0.0), _valY2(0.0),
      _valXY(0.0),
      _obsCode(0),
      _isSynthetic(0),
      _mopsStatus(0),
      _flagClassification(0)
{}

det::DiaSource::DiaSource(DiaSource const & other)
	: BaseSourceAttributes<NUM_DIASOURCE_NULLABLE_FIELDS>(other),
	  _diaSource2Id(other._diaSource2Id), 
	  _scId(other._scId), 
	  _ssmId(other._ssmId),
      _lengthDeg(other._lengthDeg),        
      _flux(other._flux), 
      _fluxErr(other._fluxErr),
      _refMag(other._refMag), 
      _ixx(other._ixx), 
      _ixxErr(other._ixxErr),
      _iyy(other._iyy), 
      _iyyErr(other._iyyErr),
      _ixy(other._ixy), 
      _ixyErr(other._ixyErr),
      _valX1(other._valX1), 
      _valX2(other._valX2),
      _valY1(other._valY1), 
      _valY2(other._valY2),
      _valXY(other._valXY),
      _obsCode(other._obsCode),
      _isSynthetic(other._isSynthetic),
      _mopsStatus(other._mopsStatus),
      _flagClassification(other._flagClassification)      
{
	for(int i =0; i < NUM_DIASOURCE_NULLABLE_FIELDS; i++)
		setNull(i, other.isNull(i));
}

// -- DiaSource ----------------
bool det::DiaSource::operator==(DiaSource const & d) const {
    return ( areEqual(_id, d._id) &&
        areEqual(_ampExposureId, d._ampExposureId) &&
        areEqual(_diaSource2Id, d._diaSource2Id,  det::DIA_SOURCE_2_ID) &&
        areEqual(_filterId, d._filterId) &&
        areEqual(_objectId, d._objectId,  det::OBJECT_ID) &&
        areEqual(_movingObjectId, d._movingObjectId,  det::MOVING_OBJECT_ID) &&
        areEqual(_procHistoryId, d._procHistoryId) &&
        areEqual(_scId, d._scId) &&
        areEqual(_ssmId, d._ssmId,  det::SSM_ID) &&           
        areEqual(_ra, d._ra) &&
        areEqual(_dec, d._dec) &&
        areEqual(_raErr4wcs, d._raErr4wcs,  det::RA_ERR_4_WCS) &&
        areEqual(_decErr4wcs, d._decErr4wcs,  det::DEC_ERR_4_WCS) &&
        areEqual(_raErr4detection, d._raErr4detection) &&
        areEqual(_decErr4detection, d._decErr4detection) &&
        areEqual(_xFlux, d._xFlux,  det::X_FLUX) &&
        areEqual(_xFluxErr, d._xFluxErr,  det::X_FLUX_ERR) &&
        areEqual(_yFlux, d._yFlux,  det::Y_FLUX) &&
        areEqual(_yFluxErr, d._yFluxErr,  det::Y_FLUX_ERR) &&
        areEqual(_xPeak, d._xPeak,  det::X_PEAK) && 
        areEqual(_yPeak, d._yPeak,  det::Y_PEAK) && 
        areEqual(_raPeak, d._raPeak,  det::RA_PEAK) && 
        areEqual(_decPeak, d._decPeak,  det::DEC_PEAK) &&   
        areEqual(_xAstrom, d._xAstrom,  det::X_ASTROM) &&
        areEqual(_xAstromErr, d._xAstromErr,  det::X_ASTROM_ERR) &&                   
        areEqual(_yAstrom, d._yAstrom,  det::Y_ASTROM) &&
        areEqual(_yAstromErr, d._yAstromErr,  det::Y_ASTROM_ERR) &&
        areEqual(_raAstrom, d._raAstrom,  det::RA_ASTROM) &&
        areEqual(_raAstromErr, d._raAstromErr,  det::RA_ASTROM_ERR) &&                   
        areEqual(_decAstrom, d._decAstrom,  det::DEC_ASTROM) &&
        areEqual(_decAstromErr, d._decAstromErr,  det::DEC_ASTROM_ERR) &&
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
        areEqual(_modelMagErr, d._modelMagErr,  det::MODEL_MAG_ERR) &&           
        areEqual(_instMag, d._instMag) &&
        areEqual(_instMagErr, d._instMagErr) &&
        areEqual(_nonGrayCorrMag, d._nonGrayCorrMag,  det::NON_GRAY_CORR_MAG) &&
        areEqual(_nonGrayCorrMagErr, d._nonGrayCorrMagErr,  det::NON_GRAY_CORR_MAG_ERR) &&
        areEqual(_atmCorrMag, d._atmCorrMag,  det::ATM_CORR_MAG) &&
        areEqual(_atmCorrMagErr, d._atmCorrMagErr,  det::ATM_CORR_MAG_ERR) &&
        areEqual(_apDia, d._apDia,  det::AP_DIA) &&
        areEqual(_refMag, d._refMag,  det::REF_MAG) &&                   
        areEqual(_ixx, d._ixx,  det::IXX) &&
        areEqual(_ixxErr, d._ixxErr,  det::IXX_ERR) &&
        areEqual(_iyy, d._iyy,  det::IYY) &&
        areEqual(_iyyErr, d._iyyErr,  det::IYY_ERR) &&
        areEqual(_ixy, d._ixy,  det::IXY) &&
        areEqual(_ixyErr, d._ixyErr,  det::IXY_ERR) &&
        areEqual(_snr, d._snr) &&
        areEqual(_chi2, d._chi2) &&
        areEqual(_valX1, d._valX1) &&
        areEqual(_valX2, d._valX2) &&
        areEqual(_valY1, d._valY1) &&
        areEqual(_valY2, d._valY2) &&
        areEqual(_valXY, d._valXY) &&
        areEqual(_obsCode,  d._obsCode,  det::OBS_CODE) &&
        areEqual(_isSynthetic, d._isSynthetic,  det::IS_SYNTHETIC) &&
        areEqual(_mopsStatus, d._mopsStatus,  det::MOPS_STATUS) &&
        areEqual(_flag4association, d._flag4association,  det::FLAG_4_ASSOCIATION) &&
        areEqual(_flag4detection, d._flag4detection,  det::FLAG_4_DETECTION) &&
        areEqual(_flag4wcs, d._flag4wcs,  det::FLAG_4_WCS) &&
        areEqual(_flagClassification, d._flagClassification,  det::FLAG_CLASSIFICATION));
}
