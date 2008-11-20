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

    for SourceVector instances.
 */
class DiaSourceVectorFormatter : public lsst::daf::persistence::Formatter {
public:

    virtual ~DiaSourceVectorFormatter();

    virtual void write(
        lsst::daf::base::Persistable const *,
        lsst::daf::persistence::Storage::Ptr,
        lsst::daf::base::DataProperty::PtrType
    );
    virtual lsst::daf::base::Persistable* read(
        lsst::daf::persistence::Storage::Ptr,
        lsst::daf::base::DataProperty::PtrType
    );
    virtual void update(
        lsst::daf::base::Persistable*,
        lsst::daf::persistence::Storage::Ptr,
        lsst::daf::base::DataProperty::PtrType
    );

    template <class Archive>
    static void delegateSerialize(
        Archive &,
        unsigned int const,
        lsst::daf::base::Persistable *
    );

private:

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

