// -*- lsst-c++ -*-
//
//##====----------------                                ----------------====##/
//
//! \file   SourceFormatters.h
//! \brief  Formatter subclasses for Source
//!         and Persistable containers thereof.
//
//##====----------------                                ----------------====##/

#ifndef LSST_AFW_FORMATTERS_SOURCE_FORMATTERS_H
#define LSST_AFW_FORMATTERS_SOURCE_FORMATTERS_H

#include <string>
#include <vector>

#include <lsst/daf/base.h>
#include <lsst/pex/policy/Policy.h>
#include <lsst/daf/persistence/Formatter.h>
#include <lsst/daf/persistence/Storage.h>
#include <lsst/daf/persistence/DbStorage.h>
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

    lsst::daf::persistence::Formatter::Ptr _policy;

    SourceVectorFormatter(lsst::daf::persistence::Formatter::Ptr const &);

    static lsst::daf::persistence::Formatter::Ptr createInstance(
        lsst::daf::persistence::Formatter::Ptr
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

