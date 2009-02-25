// -*- lsst-c++ -*-
//
//##====----------------                                ----------------====##/
//
//! \file
//! \brief  Formatter subclasses for DiaSource
//!         and Persistable containers thereof.
//
//##====----------------                                ----------------====##/

#ifndef LSST_AFW_FORMATTERS_DIA_SOURCE_FORMATTER_H
#define LSST_AFW_FORMATTERS_DIA_SOURCE_FORMATTER_H

#include <string>
#include <vector>

#include "lsst/daf/base.h"
#include "lsst/daf/persistence.h"
#include "lsst/afw/detection/DiaSource.h"

namespace lsst {
namespace afw {
namespace formatters {

/*!
    Formatter that supports persistence and retrieval with

    - lsst::daf::persistence::DbStorage
    - lsst::daf::persistence::DbTsvStorage
    - lsst::daf::persistence::BoostStorage

    for PersistableDiaSourceVector instances.
 */
class DiaSourceVectorFormatter : public lsst::daf::persistence::Formatter {
public:

    virtual ~DiaSourceVectorFormatter();

    virtual void write(
        lsst::daf::base::Persistable const *,
        lsst::daf::persistence::Storage::Ptr,
        lsst::daf::base::PropertySet::Ptr
    );
    virtual lsst::daf::base::Persistable* read(
        lsst::daf::persistence::Storage::Ptr,
        lsst::daf::base::PropertySet::Ptr
    );
    virtual void update(
        lsst::daf::base::Persistable*,
        lsst::daf::persistence::Storage::Ptr,
        lsst::daf::base::PropertySet::Ptr
    );

    template <class Archive>
    static void delegateSerialize(
        Archive &,
        unsigned int const,
        lsst::daf::base::Persistable *
    );

private:
    //Ordered list of columns in DiaSource table of the DC3b schema    
    enum Columns {
        DIA_SOURCE_ID = 0,
        AMP_EXPOSURE_ID,
        DIA_SOURCE_TO_ID,
        FILTER_ID,
        OBJECT_ID,
        MOVING_OBJECT_ID,
        PROC_HISTORY_ID,
        SC_ID,
        SSM_ID,
        RA,
        RA_ERR_FOR_DETECTION,
        RA_ERR_FOR_WCS,
        DECL,
        DEC_ERR_FOR_DETECTION,
        DEC_ERR_FOR_WCS,
        X_FLUX,
        X_FLUX_ERR,
        Y_FLUX,
        Y_FLUX_ERR,
        RA_FLUX,
        RA_FLUX_ERR,
        DEC_FLUX,
        DEC_FLUX_ERR,
        X_PEAK,
        Y_PEAK,
        RA_PEAK,
        DEC_PEAK,
        X_ASTROM,
        X_ASTROM_ERR,
        Y_ASTROM,
        Y_ASTROM_ERR,
        RA_ASTROM,
        RA_ASTROM_ERR,
        DEC_ASTROM,
        DEC_ASTROM_ERR,
        TAI_MID_POINT,
        TAI_RANGE,
        FWHM_A,
        FWHM_B,
        FWHM_THETA,
        LENGTH_DEG,
        FLUX,
        FLUX_ERR,
        PSF_FLUX,
        PSF_FLUX_ERR,
        AP_FLUX,
        AP_FLUX_ERR,
        MODEL_FLUX,
        MODEL_FLUX_ERR,
        INST_FLUX,
        INST_FLUX_ERR,
        NON_GRAY_CORR_FLUX,
        NON_GRAY_CORR_FLUX_ERR,
        ATM_CORR_FLUX,
        ATM_CORR_FLUX_ERR,
        AP_DIA,
        REF_FLUX,
        IXX,
        IXX_ERR,
        IYY,
        IYY_ERR,       
        IXY,
        IXY_ERR,
        SNR,
        CHI2,
        VAL_X1,
        VAL_X2,
        VAL_Y1,
        VAL_Y2,
        VAL_XY,
        OBS_CODE,
        IS_SYNTHETIC,
        MOPS_STATUS,
        FLAG_FOR_ASSOCIATION,
        FLAG_FOR_DETECTION,
        FLAG_FOR_WCS,
        FLAG_CLASSIFICATION
    };
    

    lsst::pex::policy::Policy::Ptr _policy;

    explicit DiaSourceVectorFormatter(lsst::pex::policy::Policy::Ptr const & policy);

    static lsst::daf::persistence::Formatter::Ptr createInstance(
        lsst::pex::policy::Policy::Ptr
    );
    static lsst::daf::persistence::FormatterRegistration registration;

    template <typename T>
    static void insertRow(
        T &,
        lsst::afw::detection::DiaSource const &
    );
    static void setupFetch(
        lsst::daf::persistence::DbStorage &,
        lsst::afw::detection::DiaSource &
    );
};


}}} // namespace lsst::afw::formatters

#endif // LSST_AFW_FORMATTERS_DIA_SOURCE_FORMATTER_H

