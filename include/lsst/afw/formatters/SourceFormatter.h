// -*- lsst-c++ -*-
//
//##====----------------                                ----------------====##/
//
//! \file
//! \brief  Formatter subclasses for Source
//!         and Persistable containers thereof.
//
//##====----------------                                ----------------====##/

#ifndef LSST_AFW_FORMATTERS_SOURCE_FORMATTERS_H
#define LSST_AFW_FORMATTERS_SOURCE_FORMATTERS_H

#include <string>
#include <vector>

#include "lsst/daf/base.h"
#include "lsst/daf/persistence.h"
#include "lsst/afw/detection/Source.h"

namespace lsst {
namespace afw {
namespace formatters {

/*!
    Formatter that supports persistence and retrieval with

    - lsst::daf::persistence::DbStorage
    - lsst::daf::persistence::DbTsvStorage
    - lsst::daf::persistence::BoostStorage

    for SourceVector instances.
 */
class SourceVectorFormatter : public lsst::daf::persistence::Formatter {
	typedef lsst::afw::detection::Source Source;
    typedef std::vector<Source> SourceVector;
public:

    virtual ~SourceVectorFormatter();

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

    /**
     * \brief List of columns in Source table in order of the DC3b schema
     */
    enum Columns {
        SOURCE_ID = 0,
        AMP_EXPOSURE_ID,
        FILTER_ID,
        OBJECT_ID,
        MOVING_OBJECT_ID,
        PROC_HISTORY_ID,
        RA,
        RA_ERR_4_DETECTION,
        RA_ERR_4_WCS,
        DECL,
        DEC_ERR_4_DETECTION,
        DEC_ERR_4_WCS,
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
        PSF_MAG,
        PSF_MAG_ERR,
        AP_MAG,
        AP_MAG_ERR,
        MODEL_MAG,
        MODEL_MAG_ERR,
        PETRO_MAG,
        PETRO_MAG_ERR,
        INST_MAG,
        INST_MAG_ERR,
        NON_GRAY_CORR_MAG,
        NON_GRAY_CORR_MAG_ERR,
        ATM_CORR_MAG,
        ATM_CORR_MAG_ERR,
        AP_DIA,
        SNR,
        CHI2,
        SKY,
        SKY_ERR,
        FLAG_4_ASSOCIATION,
        FLAG_4_DETECTION,
        FLAG_4_WCS,   
    };


    lsst::pex::policy::Policy::Ptr _policy;

    explicit SourceVectorFormatter(lsst::pex::policy::Policy::Ptr const & policy);

    static lsst::daf::persistence::Formatter::Ptr createInstance(
        lsst::pex::policy::Policy::Ptr
    );
    static lsst::daf::persistence::FormatterRegistration registration;

    template <typename T>
    static void insertRow(
        T &,
        lsst::afw::detection::Source const &
    );
    static void setupFetch(
        lsst::daf::persistence::DbStorage &,
        lsst::afw::detection::Source &
    );
};


}}} // namespace lsst::afw::formatters

#endif // LSST_AFW_FORMATTERS_SOURCE_FORMATTERS_H

