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

// -- DiaSource ----------------
det::DiaSource::DiaSource() { 
    //make list of new field names    
    std::vector newFields(DiaSource::NUM_FIELDS - SourceBase::NUM_FIELDS);
    newFields[SC_ID - SourceBase::NUM_FIELDS] = "scId";
    newFields[SSM_ID - SourceBase::NUM_FIELDS] = "ssmId";
    newFields[LENGTH_DEG - SourceBase::NUM_FIELDS] = "lengthDeg";
    newFields[FLUX - SourceBase::NUM_FIELDS] = "flux";    
    newFields[FLUX_ERR - SourceBase::NUM_FIELDS] = "fluxErr";    
    newFields[REF_MAG - SourceBase::NUM_FIELDS] = "refMag";    
    newFields[IXX - SourceBase::NUM_FIELDS] = "Ixx";    
    newFields[IXX_ERR - SourceBase::NUM_FIELDS] = "IxxErr";    
    newFields[IYY - SourceBase::NUM_FIELDS] = "Iyy";    
    newFields[IYY_ERR - SourceBase::NUM_FIELDS] = "IyyErr";  
    newFields[IXY - SourceBase::NUM_FIELDS] = "Ixy";    
    newFields[IXY_ERR - SourceBase::NUM_FIELDS] = "IxyErr";    
    newFields[VAL_X1 - SourceBase::NUM_FIELDS] = "valx1";  
    newFields[VAL_X2 - SourceBase::NUM_FIELDS] = "valx2";    
    newFields[VAL_Y1 - SourceBase::NUM_FIELDS] = "valy1";    
    newFields[VAL_Y2 - SourceBase::NUM_FIELDS] = "valy2";  
    newFields[VAL_XY - SourceBase::NUM_FIELDS] = "valxy";  
    newFields[OBS_CODE - SourceBase::NUM_FIELDS] = "obsCode";
    newFields[IS_SYNTHETIC - SourceBase::NUM_FIELDS] = "isSynthetic";
    newFields[MOPS_STATUS - SourceBase::NUM_FIELDS] = "mopsStatus";
    
    //add in the new fields
    addField(newFields);
    
    // for DB storage, need proper ID name
    renameField(ID, "diaSourceId");
    
    //set non null fields
    set(DIA_SOURCE_ID, 0);
    set(AMP_EXPOSURE_ID, 0);
    set(FILTER_ID, 0);
    set(PROC_HISTORY_ID, 0);
    set(SC_ID, 0);
    set(RA, 0.0);
    set(DECL, 0.0);
    set(RA_ERR_4_DETECTION, 0.0f);
    set(DEC_ERR_4_DETECTION, 0.0f);
    set(TAI_MID_POINT, 0.0);
    set(TAI_RANGE, 0.0f);
    set(FWHM_A, 0.0f);
    set(FWHM_B, 0.0f);
    set(FWHM_THETA, 0.0f);
    set(LENGTH_DEG, 0.0f);
    set(FLUX, 0.0f);
    set(FLUX_ERR, 0.0f);
    set(PSF_MAG, 0.0);
    set(PST_MAG_ERR, 0.0f);
    set(AP_MAG, 0.0);
    set(AP_MAG_ERR, 0.0f);
    set(MODEL_MAG, 0.0);
    set(SNR, 0.0f);
    set(CHI2, 0.0f);  
    set(VAL_X1, 0.0);
    set(VAL_X2, 0.0);
    set(VAL_Y1, 0.0);
    set(VAL_Y2, 0.0);
    set(VAL_XY, 0.0);   
}
