// -*- lsst-c++ -*-

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

/**
 * Default Contructor
 */
det::DiaSource::DiaSource()
    : _ssmId(0), _diaSourceToId(0), 
      _flagClassification(0),  
      _lengthDeg(0.0), 
      _valX1(0.0), _valX2(0.0),
      _valY1(0.0), _valY2(0.0),
      _valXY(0.0),
      _refFlux(0.0),
      _scId(0),      
      _obsCode(std::string()),
      _isSynthetic(0),
      _mopsStatus(0)
{ }

/**
 * Copy Constructor
 */
det::DiaSource::DiaSource(DiaSource const & other)
    : BaseSourceAttributes<NUM_DIASOURCE_NULLABLE_FIELDS>(other),
      _ssmId(other._ssmId),
      _diaSourceToId(other._diaSourceToId), 
      _flagClassification(other._flagClassification),
      _lengthDeg(other._lengthDeg),        
      _valX1(other._valX1), 
      _valX2(other._valX2),
      _valY1(other._valY1), 
      _valY2(other._valY2),
      _valXY(other._valXY),
      _refFlux(other._refFlux),
      _scId(other._scId), 
      _obsCode(other._obsCode),
      _isSynthetic(other._isSynthetic),
      _mopsStatus(other._mopsStatus)
{
    for (int i =0; i != NUM_DIASOURCE_NULLABLE_FIELDS; ++i) {
        setNull(i, other.isNull(i));
    }
}

/**
 * Test for equality between DiaSource
 * \return true if all of the fields are equal or null in both DiaSource
 */
bool det::DiaSource::operator==(DiaSource const & d) const {
    if (areEqual(_id, d._id) &&
        areEqual(_ampExposureId, d._ampExposureId) &&
        areEqual(_diaSourceToId, d._diaSourceToId, det::DIA_SOURCE_TO_ID) &&
        areEqual(_filterId, d._filterId) &&
        areEqual(_objectId, d._objectId, det::OBJECT_ID) &&
        areEqual(_movingObjectId, d._movingObjectId, det::MOVING_OBJECT_ID) &&
        areEqual(_ra, d._ra) &&
        areEqual(_dec, d._dec) &&
        areEqual(_raErrForWcs, d._raErrForWcs, det::RA_ERR_FOR_WCS) &&
        areEqual(_decErrForWcs, d._decErrForWcs, det::DEC_ERR_FOR_WCS) &&
        areEqual(_raErrForDetection, d._raErrForDetection, det::RA_ERR_FOR_DETECTION) &&
        areEqual(_decErrForDetection, d._decErrForDetection, det::DEC_ERR_FOR_DETECTION) &&
        areEqual(_xAstrom, d._xAstrom) &&
        areEqual(_xAstromErr, d._xAstromErr, det::X_ASTROM_ERR) &&
        areEqual(_yAstrom, d._yAstrom) &&
        areEqual(_yAstromErr, d._yAstromErr, det::Y_ASTROM_ERR) &&
        areEqual(_taiMidPoint, d._taiMidPoint) &&
        areEqual(_taiRange, d._taiRange) &&
        areEqual(_psfFlux, d._psfFlux) &&
        areEqual(_psfFluxErr, d._psfFluxErr, det::PSF_FLUX_ERR) &&
        areEqual(_apFlux, d._apFlux) &&
        areEqual(_apFluxErr, d._apFluxErr, det::AP_FLUX_ERR) &&
        areEqual(_modelFlux, d._modelFlux) &&
        areEqual(_modelFluxErr, d._modelFluxErr, det::MODEL_FLUX_ERR) &&           
        areEqual(_instFlux, d._instFlux) &&
        areEqual(_instFluxErr, d._instFluxErr, det::INST_FLUX_ERR) &&
        areEqual(_apDia, d._apDia, det::AP_DIA) &&                 
        areEqual(_ixx, d._ixx, det::IXX) &&
        areEqual(_ixxErr, d._ixxErr, det::IXX_ERR) &&
        areEqual(_iyy, d._iyy, det::IYY) &&
        areEqual(_iyyErr, d._iyyErr, det::IYY_ERR) &&
        areEqual(_ixy, d._ixy, det::IXY) &&
        areEqual(_ixyErr, d._ixyErr, det::IXY_ERR) &&
        areEqual(_snr, d._snr) &&
        areEqual(_chi2, d._chi2) &&
        areEqual(_flagForDetection, d._flagForDetection, det::FLAG_FOR_DETECTION) &&
        areEqual(_flagClassification, d._flagClassification, det::FLAG_CLASSIFICATION) &&
        
        //not defined in DC3a
        areEqual(_procHistoryId, d._procHistoryId) &&
        areEqual(_scId, d._scId) &&
        areEqual(_ssmId, d._ssmId, det::SSM_ID) &&     
        areEqual(_xFlux, d._xFlux, det::X_FLUX) &&
        areEqual(_xFluxErr, d._xFluxErr, det::X_FLUX_ERR) &&
        areEqual(_yFlux, d._yFlux, det::Y_FLUX) &&
        areEqual(_yFluxErr, d._yFluxErr, det::Y_FLUX_ERR) &&
        areEqual(_xPeak, d._xPeak, det::X_PEAK) && 
        areEqual(_yPeak, d._yPeak, det::Y_PEAK) && 
        areEqual(_raPeak, d._raPeak, det::RA_PEAK) && 
        areEqual(_decPeak, d._decPeak, det::DEC_PEAK) &&
        areEqual(_raAstrom, d._raAstrom, det::RA_ASTROM) &&
        areEqual(_raAstromErr, d._raAstromErr, det::RA_ASTROM_ERR) &&                   
        areEqual(_decAstrom, d._decAstrom, det::DEC_ASTROM) &&
        areEqual(_decAstromErr, d._decAstromErr, det::DEC_ASTROM_ERR) &&
        areEqual(_lengthDeg, d._lengthDeg) &&
        areEqual(_nonGrayCorrFlux, d._nonGrayCorrFlux, det::NON_GRAY_CORR_FLUX) &&
        areEqual(_nonGrayCorrFluxErr, d._nonGrayCorrFluxErr, det::NON_GRAY_CORR_FLUX_ERR) &&
        areEqual(_atmCorrFlux, d._atmCorrFlux, det::ATM_CORR_FLUX) &&
        areEqual(_atmCorrFluxErr, d._atmCorrFluxErr, det::ATM_CORR_FLUX_ERR) &&
        areEqual(_refFlux, d._refFlux, det::REF_FLUX) &&
        areEqual(_valX1, d._valX1) &&
        areEqual(_valX2, d._valX2) &&
        areEqual(_valY1, d._valY1) &&
        areEqual(_valY2, d._valY2) &&
        areEqual(_valXY, d._valXY) &&
        areEqual(_obsCode, d._obsCode, det::OBS_CODE) &&
        areEqual(_isSynthetic, d._isSynthetic, det::IS_SYNTHETIC) &&
        areEqual(_mopsStatus, d._mopsStatus, det::MOPS_STATUS) &&
        areEqual(_flagForAssociation, d._flagForAssociation, det::FLAG_FOR_ASSOCIATION) &&
        areEqual(_flagForWcs, d._flagForWcs, det::FLAG_FOR_WCS)        
    ) {
        for (int i = 0; i < NUM_DIASOURCE_NULLABLE_FIELDS; ++i) {
            if (isNull(i) != d.isNull(i)) {
                return false;
            }
        }
        return true;
    }
    
    return false;
}
