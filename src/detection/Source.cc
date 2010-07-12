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
//! \brief Support for Sources
//!
//##====----------------                                ----------------====##/

#include "lsst/daf/base.h"
#include "lsst/afw/detection/Source.h"
#include "lsst/pex/exceptions/Runtime.h"

namespace det = lsst::afw::detection;

/**
 * Default Contructor
 */
det::Source::Source()
    : _raObject(0.0),
      _decObject(0.0), 
      _petroFlux(0.0),
      _petroFluxErr(0.0),
      _sky(0.0),
      _skyErr(0.0)
{ }

/**
 * Copy Constructor
 */
det::Source::Source(Source const & other)
    : BaseSourceAttributes<NUM_SOURCE_NULLABLE_FIELDS>(other),
      _raObject(other._raObject),
      _decObject(other._decObject),
      _petroFlux(other._petroFlux),
      _petroFluxErr(other._petroFluxErr),
      _sky(other._sky),
      _skyErr(other._skyErr)      
{
    for (int i =0; i != NUM_SOURCE_NULLABLE_FIELDS; ++i) {
        setNull(i, other.isNull(i));
    }
}

/**
 * Test for equality between DiaSource
 * \return true if all of the fields are equal or null in both DiaSource
 */
bool det::Source::operator==(Source const & d) const {
    if (areEqual(_id, d._id) &&
        areEqual(_ampExposureId, d._ampExposureId,  AMP_EXPOSURE_ID) &&
        areEqual(_filterId, d._filterId) &&
        areEqual(_objectId, d._objectId,  OBJECT_ID) &&
        areEqual(_movingObjectId, d._movingObjectId,  MOVING_OBJECT_ID) &&
        areEqual(_procHistoryId, d._procHistoryId) &&
        areEqual(_ra, d._ra) &&
        areEqual(_dec, d._dec) &&
        areEqual(_raErrForWcs, d._raErrForWcs) &&
        areEqual(_decErrForWcs, d._decErrForWcs) &&
        areEqual(_raErrForDetection, d._raErrForDetection,  
                RA_ERR_FOR_DETECTION) &&
        areEqual(_decErrForDetection, d._decErrForDetection,  
                DEC_ERR_FOR_DETECTION) &&
        areEqual(_xFlux, d._xFlux,  X_FLUX) &&
        areEqual(_xFluxErr, d._xFluxErr,  X_FLUX_ERR) &&
        areEqual(_yFlux, d._yFlux,  Y_FLUX) &&
        areEqual(_yFluxErr, d._yFluxErr,  Y_FLUX_ERR) &&
        areEqual(_xPeak, d._xPeak,  X_PEAK) &&
        areEqual(_yPeak, d._yPeak,  Y_PEAK) &&
        areEqual(_raPeak, d._raPeak,  RA_PEAK) && 
        areEqual(_decPeak, d._decPeak,  DEC_PEAK) &&
        areEqual(_xAstrom, d._xAstrom,  X_ASTROM) &&
        areEqual(_xAstromErr, d._xAstromErr,  X_ASTROM_ERR) &&
        areEqual(_yAstrom, d._yAstrom,  Y_ASTROM) &&
        areEqual(_yAstromErr, d._yAstromErr,  Y_ASTROM_ERR) &&
        areEqual(_raAstrom, d._raAstrom,  RA_ASTROM) &&
        areEqual(_raAstromErr, d._raAstromErr,  RA_ASTROM_ERR) &&
        areEqual(_decAstrom, d._decAstrom,  DEC_ASTROM) &&
        areEqual(_decAstromErr, d._decAstromErr,  DEC_ASTROM_ERR) &&
        areEqual(_raObject, d._raObject, RA_OBJECT) &&
        areEqual(_decObject, d._decObject, DEC_OBJECT) &&
        areEqual(_taiMidPoint, d._taiMidPoint) &&
        areEqual(_taiRange, d._taiRange,  TAI_RANGE) &&
        areEqual(_psfFlux, d._psfFlux) &&
        areEqual(_psfFluxErr, d._psfFluxErr) &&
        areEqual(_apFlux, d._apFlux) &&
        areEqual(_apFluxErr, d._apFluxErr) &&
        areEqual(_modelFlux, d._modelFlux) &&
        areEqual(_modelFluxErr, d._modelFluxErr) &&
        areEqual(_petroFlux, d._petroFlux,  PETRO_FLUX) &&
        areEqual(_petroFluxErr, d._petroFluxErr,  PETRO_FLUX_ERR) &&            
        areEqual(_instFlux, d._instFlux) &&
        areEqual(_instFluxErr, d._instFluxErr) &&
        areEqual(_nonGrayCorrFlux, d._nonGrayCorrFlux,  NON_GRAY_CORR_FLUX) &&
        areEqual(_nonGrayCorrFluxErr, d._nonGrayCorrFluxErr,  
                NON_GRAY_CORR_FLUX_ERR) &&
        areEqual(_atmCorrFlux, d._atmCorrFlux,  ATM_CORR_FLUX) &&
        areEqual(_atmCorrFluxErr, d._atmCorrFluxErr,  ATM_CORR_FLUX_ERR) &&
        areEqual(_apDia, d._apDia,  AP_DIA) &&
        areEqual(_ixx, d._ixx, det::IXX) &&
        areEqual(_ixxErr, d._ixxErr, det::IXX_ERR) &&
        areEqual(_iyy, d._iyy, det::IYY) &&
        areEqual(_iyyErr, d._iyyErr, det::IYY_ERR) &&
        areEqual(_ixy, d._ixy, det::IXY) &&
        areEqual(_ixyErr, d._ixyErr, det::IXY_ERR) &&
        areEqual(_snr, d._snr) &&
        areEqual(_chi2, d._chi2) &&
        areEqual(_sky, d._sky,  SKY) &&
        areEqual(_skyErr, d._skyErr,  SKY_ERR) &&
        areEqual(_flagForAssociation, d._flagForAssociation,  
                FLAG_FOR_ASSOCIATION) &&
        areEqual(_flagForDetection, d._flagForDetection,  
                FLAG_FOR_DETECTION) &&
        areEqual(_flagForWcs, d._flagForWcs,  FLAG_FOR_WCS)) 
    {
        //check NULLABLE field state equality
        for (int i = 0; i < NUM_SOURCE_NULLABLE_FIELDS; ++i) {
            if (isNull(i) != d.isNull(i)) {
                return false;
            }
        }
        return true;
    }
    
    return false;
}
