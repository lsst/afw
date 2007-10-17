// -*- lsst-c++ -*-
//
//##====----------------                                ----------------====##/
//
//! \file   DiaSourceFormatters.h
//! \brief  Formatter subclasses for DiaSource
//!         and Persistable containers thereof.
//
//##====----------------                                ----------------====##/

#ifndef LSST_FW_FORMATTERS_DIA_SOURCE_FORMATTERS_H
#define LSST_FW_FORMATTERS_DIA_SOURCE_FORMATTERS_H

#include <string>
#include <vector>

#include <lsst/mwi/data/DataProperty.h>
#include <lsst/mwi/policy/Policy.h>
#include <lsst/mwi/persistence/Formatter.h>
#include <lsst/mwi/persistence/DbStorage.h>

#include "lsst/fw/DiaSource.h"


namespace lsst {
namespace fw {
namespace formatters {

using namespace lsst::mwi::persistence;
using lsst::mwi::policy::Policy;
using lsst::mwi::data::DataProperty;


/*!
    Formatter that supports persistence and retrieval with

    - lsst::mwi::persistence::DbStorage
    - lsst::mwi::persistence::DbTsvStorage
    - lsst::mwi::persistence::BoostStorage

    for DiaSourceVector instances.
 */
class DiaSourceVectorFormatter : public Formatter {
public:

    virtual ~DiaSourceVectorFormatter();

    virtual void write(Persistable const *, Storage::Ptr, DataProperty::PtrType);
    virtual Persistable::Ptr read(Storage::Ptr, DataProperty::PtrType);
    virtual void update(Persistable::Ptr, Storage::Ptr, DataProperty::PtrType);

    template <class Archive> static void delegateSerialize(Archive &, unsigned int const, Persistable *);

private:

    Policy::Ptr _policy;

    DiaSourceVectorFormatter(Policy::Ptr const &);

    static Formatter::Ptr createInstance(Policy::Ptr);
    static FormatterRegistration registration;

    template <typename T> static void insertRow(T &, DiaSource const &);
    static void setupFetch(DbStorage &, DiaSource &);
};


}}} // end of namespace lsst::fw::formatters

#endif // LSST_FW_FORMATTERS_DIA_SOURCE_FORMATTERS_H

