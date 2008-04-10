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

#include <lsst/daf/base.h>
#include <lsst/daf/persistence.h>
#include <lsst/afw/detection/Source.h>

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
public:

    virtual ~SourceVectorFormatter();

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

