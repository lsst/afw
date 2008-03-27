// -*- lsst-c++ -*-
//
//##====----------------                                ----------------====##/
//
//! \file   DiaSourceFormatters.h
//! \brief  Formatter subclasses for DiaSource
//!         and Persistable containers thereof.
//
//##====----------------                                ----------------====##/

#ifndef LSST_AFW_FORMATTERS__SOURCE_FORMATTERS_H
#define LSST_AFW_FORMATTERS__SOURCE_FORMATTERS_H

#include <string>
#include <vector>

#include <lsst/daf/data/DataProperty.h>
#include <lsst/pex/policy/Policy.h>
#include <lsst/daf/persistence/Formatter.h>
#include <lsst/daf/persistence/DbStorage.h>

#include "lsst/afw/DiaSource.h"


namespace lsst {
namespace afw {
namespace formatters {

using namespace lsst::daf::persitence;
using lsst::pex::policy::Policy;
using lsst::daf::data::DataProperty;


/*!
    Formatter that supports persistence and retrieval with

    - lsst::daf::persitence::DbStorage
    - lsst::daf::persitence::DbTsvStorage
    - lsst::daf::persitence::BoostStorage

    for DiaSourceVector instances.
 */
class DiaSourceVectorFormatter : public Formatter {
public:

    virtual ~DiaSourceVectorFormatter();

    virtual void write(Persistable const *, Storage::Ptr, DataProperty::PtrType);
    virtual Persistable* read(Storage::Ptr, DataProperty::PtrType);
    virtual void update(Persistable*, Storage::Ptr, DataProperty::PtrType);

    template <class Archive> static void delegateSerialize(Archive &, unsigned int const, Persistable *);

private:

    Policy::Ptr _policy;

    DiaSourceVectorFormatter(Policy::Ptr const &);

    static Formatter::Ptr createInstance(Policy::Ptr);
    static FormatterRegistration registration;

    template <typename T> static void insertRow(T &, DiaSource const &);
    static void setupFetch(DbStorage &, DiaSource &);
};


}}} // namespace lsst::afw::formatters

#endif // LSST_AFW_FORMATTERS__SOURCE_FORMATTERS_H

