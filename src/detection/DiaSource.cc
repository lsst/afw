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
det::DiaSource::DiaSource(
    setNotNull(_id);
    setNotNull(_ampExposureId);
    setNotNull(_filterId);
    setNotNull(_procHistoryId);
    setNotNull(_scId);
    setNotNull(_ssmId);
    setNotNull(_ra);
    setNotNull(_dec);
    setNotNull(_raErr4detection);
    setNotNull(_decErr4detection);
    setNotNull(_taiMidPoint); 
    setNotNull(_taiRange);  
    setNotNull(_fwhmA);
    setNotNull(_fwhmB);
    setNotNull(_fwhmTheta);
    setNotNull(_lengthDeg);
    setNotNull(_flux);
    setNotNull(_fluxErr);
    setNotNull(_psfMag);
    setNotNull(_psfMagErr);
    setNotNull(_apMag);
    setNotNull(_apMagErr);
    setNotNull(_modelMag);
    setNotNull(_instMag);
    setNotNull(_instMagErr);        
    setNotNull(_snr);
    setNotNull(_chi2);
    setNotNull(_valX1);
    setNotNull(_valX2);
    setNotNull(_valY1);
    setNotNull(_valY2);
    setNotNull(_valXY);
}

det::DiaSource::~DiaSource() {
    setAllNull();
}

bool det::DiaSource::isNull    (NullableField const f) const {
    switch(f) {
        case DIA_SOURCE_2_ID:
            return _diaSource2Id == 0;
        case OBJECT_ID:
            return _objectId == 0;
        case MOVING_OBJECT_ID:
            return _movingObjectId == 0;
        case SSM_ID:
            return _ssmId == 0;
        case RA_ERR_4_WCS:
            return _raErr4wcs == 0;
        case DEC_ERR_4_WCS:
            return _decErr4wcs == 0;
        case X_FLUX:
            return _xFlux == 0;
        case X_FLUX_ERR:
            return _xFluxErr == 0;
        case Y_FLUX:
            return _yFlux == 0;
        case Y_FLUX_ERR:
            return _yFluxErr == 0;
        case RA_FLUX:
            return _raFlux == 0;
        case RA_FLUX_ERR:
            return _raFluxErr == 0;
        case DEC_FLUX:
            return _decFlux == 0;
        case DEC_FLUX_ERR:
            return _decFluxErr == 0;
        case X_PEAK:
            return _xPeak == 0;
        case Y_PEAK:
            return _yPeak == 0;
        case RA_PEAK:
            return _raPeak == 0;
        case DEC_PEAK:
            return _decPeak == 0;
        case X_ASTROM:
            return _xAstrom == 0;
        case X_ASTROM_ERR:
            return _xAstromErr == 0;
        case Y_ASTROM:
            return _yAstrom == 0;
        case Y_ASTROM_ERR:
            return _yAstromErr == 0;
        case RA_ASTROM:
            return _raAstrom == 0;
        case RA_ASTROM_ERR:
            return _raAstromErr == 0;
        case DEC_ASTROM:
            return _decAstrom == 0;
        case DEC_ASTROM_ERR:
            return _decAstromErr == 0;        
        case MODEL_MAG_ERR:
            return _modelMagErr == 0;
        case NON_GRAY_CORR_MAG:
            return _nonGrayCorrMag == 0;
        case NON_GRAY_CORR_MAG_ERR:
            return _nonGrayCorrMagErr == 0;
        case ATM_CORR_MAG:
            return _atmCorrMag == 0;
        case ATM_CORR_MAG_ERR:
            return _atmCorrMagErr == 0;               
        case AP_DIA:
            return _apDia == 0;
        case REF_MAG:
            return _refMag == 0;
        case IXX:        
            return _ixx == 0;
        case IXX_ERR:        
            return _ixxErr == 0;
        case IYY:        
            return _iyy == 0;
        case IYY_ERR:        
            return _iyyErr == 0 ;
        case IXY:        
            return _ixy == 0;
        case IXY_ERR:        
            return _ixyErr == 0;  
        case OBS_CODE:        
            return _obsCode == 0;        
        case IS_SYNTHETIC:
            return _isSynthetic == 0;
        case MOPS_STATUS:
            return _mopsStatus == 0;            
        case FLAG_4_ASSOCIATION:
            return _flag4association == 0;
        case FLAG_4_DETECTION:
            return _flag4detection == 0;
        case FLAG_4_WCS: 
            return _flag4wcs == 0;
        case FLAG_FLASSIFICATION:
            return _flagClassification == 0;
        default:
            return false;
    }
}
void det::DiaSource::setNull   (NullableField const f) {setNull(f, true);}
void det::DiaSource::setNotNull(NullableField const f) {setNull(f, false);}
void det::DiaSource::setNull   (NullableField const f, bool const null) {
    if(null)
        switch(f) {
            case DIA_SOURCE_2_ID:
                setNull(_diaSource2Id);
                break;
            case OBJECT_ID:
                setNull(_objectId);
                break;
            case MOVING_OBJECT_ID:
                setNull(_movingObjectId);
                break;
            case SSM_ID:
                setNull(_ssmId);
                break;
            case RA_ERR_4_WCS:
                setNull(_raErr4wcs);
                break;
            case DEC_ERR_4_WCS:
                setNull(_decErr4wcs);
                break;
            case X_FLUX:
                setNull(_xFlux);
                break;
            case X_FLUX_ERR:
                setNull(_xFluxErr);
                break;
            case Y_FLUX:
                setNull(_yFlux);
                break;
            case Y_FLUX_ERR:
                setNull(_yFluxErr);
                break;
            case RA_FLUX:
                setNull(_raFlux);
                break;
            case RA_FLUX_ERR:
                setNull(_raFluxErr);
                break;
            case DEC_FLUX:
                setNull(_decFlux);
                break;
            case DEC_FLUX_ERR:
                setNull(_decFluxErr);
                break;
            case X_PEAK:
                setNull(_xPeak);
                break;
            case Y_PEAK:
                setNull(_yPeak);
                break;
            case RA_PEAK:
                setNull(_raPeak);
                break;
            case DEC_PEAK:
                setNull(_decPeak);
                break;
            case X_ASTROM:
                setNull(_xAstrom);
                break;
            case X_ASTROM_ERR:
                setNull(_xAstromErr);
                break;
            case Y_ASTROM:
                setNull(_yAstrom);
                break;
            case Y_ASTROM_ERR:
                setNull(_yAstromErr);
                break;
            case RA_ASTROM:
                setNull(_raAstrom);
                break;
            case RA_ASTROM_ERR:
                setNull(_raAstromErr);
                break;
            case DEC_ASTROM:
                setNull(_decAstrom);
                break;
            case DEC_ASTROM_ERR:
                setNull(_decAstromErr);
                break;
            case MODEL_MAG_ERR:
                setNull(_modelMagErr);
                break;
            case NON_GRAY_CORR_MAG:
                setNull( _nonGrayCorrMag);
                break;
            case NON_GRAY_CORR_MAG_ERR:
                setNull( _nonGrayCorrMagErr);
                break;
            case ATM_CORR_MAG:
                setNull( _atmCorrMag);
                break;
            case ATM_CORR_MAG_ERR:
                setNull( _atmCorrMagErr);                          
                break;                                
            case AP_DIA:
                setNull(_apDia);
                break;
            case REF_MAG:
                setNull(_refMag);
                break;
            case IXX:        
                setNull(_ixx);
                break;
            case IXX_ERR:        
                setNull(_ixxErr);
                break;
            case IYY:        
                setNull(_iyy);
                break;
            case IYY_ERR:        
                setNull(_iyyErr);
                break;
            case IXY:        
                setNull(_ixy);
                break;
            case IXY_ERR:        
                setNull(_ixyErr);
                break;                
            case OBS_CODE:
                setNull(_obsCode);
                break;            
            case IS_SYNTHETIC:
                setNull(_isSynthetic);
                break;
            case MOPS_STATUS:
                setNull(_mopsStatus);
                break;
            case FLAG_4_ASSOCIATION:
                setNull(_flag4association);
                break;
            case FLAG_4_DETECTION:
                setNull(_flag4detection);
                break;
            case FLAG_4_WCS: 
                setNull(_flag4wcs);
                break;
            case FLAG_CLASSIFICATION:
                setNull(_flagClassification);
                break;
            default:
                return;            
        }
    } else {    
        switch(f) {
            case DIA_SOURCE_2_ID:
                setNotNull
            case OBJECT_ID:
                setNotNull(_objectId);
                break;
            case MOVING_OBJECT_ID:
                setNotNull(_movingObjectId);
                break;
            case SSM_ID:
                setNotNull(_ssmId);
                break;
            case RA_ERR_4_WCS:
                setNotNull(_raErr4wcs);
                break;
            case DEC_ERR_4_WCS:
                setNotNull(_decErr4wcs);
                break;
            case X_FLUX:
                setNotNull(_xFlux);
                break;
            case X_FLUX_ERR:
                setNotNull(_xFluxErr);
                break;
            case Y_FLUX:
                setNotNull(_yFlux);
                break;
            case Y_FLUX_ERR:
                setNotNull(_yFluxErr);
                break;
            case RA_FLUX:
                setNotNull(_raFlux);
                break;
            case RA_FLUX_ERR:
                setNotNull(_raFluxErr);
                break;
            case DEC_FLUX:
                setNotNull(_decFlux);
                break;
            case DEC_FLUX_ERR:
                setNotNull(_decFluxErr);
                break;
            case X_PEAK:
                setNotNull(_xPeak);
                break;
            case Y_PEAK:
                setNotNull(_yPeak);
                break;
            case RA_PEAK:
                setNotNull(_raPeak);
                break;
            case DEC_PEAK:
                setNotNull(_decPeak);
                break;
            case X_ASTROM:
                setNotNull(_xAstrom);
                break;
            case X_ASTROM_ERR:
                setNotNull(_xAstromErr);
                break;
            case Y_ASTROM:
                setNotNull(_yAstrom);
                break;
            case Y_ASTROM_ERR:
                setNotNull(_yAstromErr);
                break;
            case RA_ASTROM:
                setNotNull(_raAstrom);
                break;
            case RA_ASTROM_ERR:
                setNotNull(_raAstromErr);
                break;
            case DEC_ASTROM:
                setNotNull(_decAstrom);
                break;
            case DEC_ASTROM_ERR:
                setNotNull(_decAstromErr);
                break;
            case MODEL_MAG_ERR:
                setNotNull(_modelMagErr);
                break;
            case NON_GRAY_CORR_MAG:
                setNotNull( _nonGrayCorrMag);
                break;
            case NON_GRAY_CORR_MAG_ERR:
                setNotNull( _nonGrayCorrMagErr);
                break;
            case ATM_CORR_MAG:
                setNotNull( _atmCorrMag);
                break;
            case ATM_CORR_MAG_ERR:
                setNotNull( _atmCorrMagErr);                          
                break;                
            case AP_DIA:
                setNotNull(_apDia);
                break;
            case REF_MAG:
                setNotNull(_refMag);
                break;
            case IXX:        
                setNotNull(_ixx);
                break;
            case IXX_ERR:        
                setNotNull(_ixxErr);
                break;
            case IYY:        
                setNotNull(_iyy);
                break;
            case IYY_ERR:        
                setNotNull(_iyyErr);
                break;
            case IXY:        
                setNotNull(_ixy);
                break;
            case IXY_ERR:        
                setNotNull(_ixyErr);
                break;                
            case OBS_CODE:
                setNotNull(_obsCode);
                break;
            case IS_SYNTHETIC:
                setNotNull(_isSynthetic);
                break;
            case MOPS_STATUS:
                setNotNull(_mopsStatus);
                break;
            case FLAG_4_ASSOCIATION:
                setNotNull(_flag4association);
                break;
            case FLAG_4_DETECTION:
                setNotNull(_flag4detection);
                break;
            case FLAG_4_WCS: 
                setNotNull(_flag4wcs);
                break;
            case FLAG_CLASSIFICATION:
                setNotNull(_flagClassification);
                break;
            default:
                return false;
        }
    }
}

void det::DiaSource::setNull   () {
    setNull(_diaSource2Id);
    setNull(_objectId);
    setNull(_movingObjectId);
    setNull(_ssmId);
    setNull(_raErr4wcs);
    setNull(_decErr4wcs);
    setNull(_xFlux);
    setNull(_xFluxErr);   
    setNull(_yFlux);
    setNull(_yFluxErr);   
    setNull(_raFlux);
    setNull(_raFluxErr);   
    setNull(_decFlux);
    setNull(_decFluxErr);   
    setNull(_xPeak);
    setNull(_yPeak);
    setNull(_raPeak);
    setNull(_decPeak);  
    setNull(_xAstrom);
    setNull(_xAstromErr);   
    setNull(_yAstrom);
    setNull(_yAstromErr);   
    setNull(_raAstrom);
    setNull(_raAstromErr);   
    setNull(_decAstrom);
    setNull(_decAstromErr);        
    setNull(_modelMagErr);
    setNull(_nonGrayCorrMag);
    setNull(_nonGrayCorrMagErr);
    setNull(_atmCorrMag);
    setNull(_atmCorrMagErr);
    setNull(_apDia);
    setNull(_refMag);
    setNull(_ixx);
    setNull(_ixxErr);
    setNull(_iyy);
    setNull(_iyyErr);
    setNull(_ixy);
    setNull(_ixyErr);        
    setNull(_obsCode);
    setNull(_isSynthetic);
    setNull(_mopsStatus);
    setNull(_flag4association);
    setNull(_flag4detection);
    setNull(_flag4wcs);
    setNull(_flagClassification);
}
void det::DiaSource::setNotNull() {
    setNotNull(_diaSource2Id);
    setNotNull(_objectId);
    setNotNull(_movingObjectId);
    setNotNull(_ssmId);
    setNotNull(_raErr4wcs);
    setNotNull(_decErr4wcs);
    setNotNull(_xFlux);
    setNotNull(_xFluxErr);   
    setNotNull(_yFlux);
    setNotNull(_yFluxErr);   
    setNotNull(_raFlux);
    setNotNull(_raFluxErr);   
    setNotNull(_decFlux);
    setNotNull(_decFluxErr);   
    setNotNull(_xPeak);
    setNotNull(_yPeak);
    setNotNull(_raPeak);
    setNotNull(_decPeak);  
    setNotNull(_xAstrom);
    setNotNull(_xAstromErr);   
    setNotNull(_yAstrom);
    setNotNull(_yAstromErr);   
    setNotNull(_raAstrom);
    setNotNull(_raAstromErr);   
    setNotNull(_decAstrom);
    setNotNull(_decAstromErr);        
    setNotNull(_modelMagErr);
    setNotNull(_nonGrayCorrMag);
    setNotNull(_nonGrayCorrMagErr);
    setNotNull(_atmCorrMag);
    setNotNull(_atmCorrMagErr);
    setNotNull(_apDia);
    setNotNull(_refMag);
    setNotNull(_ixx);
    setNotNull(_ixxErr);
    setNotNull(_iyy);
    setNotNull(_iyyErr);
    setNotNull(_ixy);
    setNotNull(_ixyErr);  
    setNotNull(_obsCode);      
    setNotNull(_isSynthetic);
    setNotNull(_mopsStatus);
    setNotNull(_flag4association);
    setNotNull(_flag4detection);
    setNotNull(_flag4wcs);
    setNotNull(_flagClassification);
}

void det::DiaSource::setAllNotNull() {
    BaseSourceAttributes::setAllNotNull();
    setNotNull(_diaSource2Id);
    setNotNull(_scId);
    setNotNull(_ssmId);
    setNotNull(_lengthDeg);
    setNotNull(_flux);
    setNotNull(_fluxErr);
    setNotNull(_refMag);
    setNotNull(_ixx);
    setNotNull(_ixxErr);
    setNotNull(_iyy);
    setNotNull(_iyyErr);
    setNotNull(_ixy);
    setNotNull(_ixyErr);        
    setNotNull(_valX1);
    setNotNull(_valX2);
    setNotNull(_valY1);
    setNotNull(_valY2);
    setNotNull(_valXY);
    setNotNull(_obsCode);
    setNotNull(_isSynthetic);
    setNotNull(_mopsStatus);
    setNotNull(_flagClassification);
}

void det::DiaSource::setAllNull() {
    BaseSourceAttributes::setAllNotNull();
    setNull(_diaSource2Id);
    setNull(_scId);
    setNull(_ssmId);
    setNull(_lengthDeg);
    setNull(_flux);
    setNull(_fluxErr);
    setNull(_refMag);
    setNull(_ixx);
    setNull(_ixxErr);
    setNull(_iyy);
    setNull(_iyyErr);
    setNull(_ixy);
    setNull(_ixyErr);        
    setNull(_valX1);
    setNull(_valX2);
    setNull(_valY1);
    setNull(_valY2);
    setNull(_valXY);
    setNull(_obsCode);
    setNull(_isSynthetic);
    setNull(_mopsStatus);
    setNull(_flagClassification);
}

bool det::DiaSource::operator==(DiaSource const & d) const {
    if (this == &d)  {
        return true;
    }
    return (BaseSourceAttributes::operator==(*static_cast<BaseSourceAttributes const*>(&d) &&
            areEqual(_diaSource2Id, d._diaSource2Id) &&
            areEqual(_scId, d._scId)                &&                        
            areEqual(_lengthDeg, d._lengthDeg)      &&        
            areEqual(_flux, d._flux)                &&
            areEqual(_fluxErr, d._fluxErr)          &&
            areEqual(_valX1, d._valX1)              &&
            areEqual(_valX2, d._valX2)              &&
            areEqual(_valY1, d._valY1)              &&
            areEqual(_valY2, d._valY2)              &&
            areEqual(_valXY, d._valXY)              &&
            areEqual(_ssmId, d._ssmId)              &&      
            areEqual(_refMag, d._refMag)            &&                   
            areEqual(_ixx, d._ixx)                  &&
            areEqual(_ixxErr, d._ixxErr)            &&
            areEqual(_iyy, d._iyy)                  &&
            areEqual(_iyyErr, d._iyyErr)            &&
            areEqual(_ixy, d._ixy)                  &&
            areEqual(_ixyErr, d._ixyErr)            &&
            areEqual(_obsSoce,  d._obsCode)         &&
            areEqual(_isSynthetic, d._isSynthetic)  &&
            areEqual(_mopsStatus, d._mopsStatus)    &&
            areEqual(_flagClassification, d._flagClassification));
}
