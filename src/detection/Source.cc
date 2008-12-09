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

namespace det = lsst::afw::detection;

det::Source::Source() {
    //make list of new field names    
    std::vector newFields(Source::NUM_FIELDS - SourceBase::NUM_FIELDS);
    newFields[PETRO_MAG - SourceBase::NUM_FIELDS] = "petroMag";
    newFields[PETRO_MAG_ERR - SourceBase::NUM_FIELDS] = "petroMagErr";
    newFields[SKY - SourceBase::NUM_FIELDS] = "sky";
    newFields[SKY_ERR - SourceBase::NUM_FIELDS] = "skyErr";    
    
    //add in the new fields
    addField(newFields);
    
    // for DB storage, need proper ID name
    renameField(ID, "sourceId");
    
    //set non null fields
    set(SOURCE_ID, 0);
    set(FILTER_ID, 0);
    set(PROC_HISTORY_ID, 0);
    set(RA, 0.0);
    set(DECL, 0.0);
    set(RA_ERR_4_WCS, 0.0f);
    set(DEC_ERR_4_WCS, 0.0f);
    set(TAI_MID_POINT, 0.0);
    set(FWHM_A, 0.0f);
    set(FWHM_B, 0.0f);
    set(FWHM_THETA, 0.0f);
    set(PSF_MAG, 0.0);
    set(PST_MAG_ERR, 0.0f);
    set(AP_MAG, 0.0);
    set(AP_MAG_ERR, 0.0f);
    set(MODEL_MAG, 0.0);
    set(MODEL_MAG_ERR, 0.0f);
    set(SNR, 0.0f);
    set(CHI2, 0.0f);      
}
