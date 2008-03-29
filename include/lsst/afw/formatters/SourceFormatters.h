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

#include <lsst/daf/data/DataProperty.h>
#include <lsst/pex/policy/Policy.h>
#include <lsst/daf/persistence/Formatter.h>
#include <lsst/daf/persistence/DbStorage.h>

#include "lsst/afw/detection/Source.h"

namespace lsst {
namespace afw {
namespace formatters {

using namespace lsst::daf::persistence;
using lsst::pex::policy::Policy;
using lsst::daf::data::DataProperty;

/*!
    Formatter that supports persistence and retrieval with

    - lsst::daf::persistence::DbStorage
    - lsst::daf::persistence::DbTsvStorage
    - lsst::daf::persistence::BoostStorage

    for SourceVector instances.
 */
class SourceVectorFormatter : public Formatter {
public:

    virtual ~SourceVectorFormatter();

    virtual void write(lsst::daf::base::Persistable const *, Storage::Ptr, DataProperty::PtrType);
    virtual lsst::daf::base::Persistable* read(Storage::Ptr, DataProperty::PtrType);
    virtual void update(lsst::daf::base::Persistable*, Storage::Ptr, DataProperty::PtrType);

    template <class Archive> static void delegateSerialize(Archive &, unsigned int const, lsst::daf::base::Persistable *);

private:

    Policy::Ptr _policy;

    SourceVectorFormatter(Policy::Ptr const &);

    static Formatter::Ptr createInstance(Policy::Ptr);
    static FormatterRegistration registration;

    template <typename T> static void insertRow(T &, lsst::afw::detection::Source const &);
    static void setupFetch(DbStorage &, lsst::afw::detection::Source &);
};


}}} // namespace lsst::afw::formatters

#endif // LSST_AFW_FORMATTERS_SOURCE_FORMATTERS_H

