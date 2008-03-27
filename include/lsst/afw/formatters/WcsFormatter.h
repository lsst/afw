// -*- lsst-c++ -*-
#ifndef LSST_AFW_FORMATTERS_WCSFORMATTER_H
#define LSST_AFW_FORMATTERS_WCSFORMATTER_H

/** \file
 * \brief Interface for WcsFormatter class
 *
 * \author $Author: ktlim $
 * \version $Revision: 2377 $
 * \date $Date$
 *
 * Contact: Kian-Tat Lim (ktl@slac.stanford.edu)
 * \ingroup afw
 */

/** \class lsst::afw::formatters::WcsFormatter
 * \brief Class implementing persistence and retrieval for WCS objects.
 *
 * \ingroup afw
 */

#include "lsst/daf/persistence/Formatter.h"

namespace lsst {
namespace afw {

class WCS;

namespace formatters {
               

using namespace lsst::daf::persitence;

class WcsFormatter : public Formatter {
public:       
    virtual ~WcsFormatter(void);

    virtual void write(Persistable const* persistable, Storage::Ptr storage,
                       lsst::daf::data::DataProperty::PtrType additionalData);
    virtual Persistable* read(
        Storage::Ptr storage,
        lsst::daf::data::DataProperty::PtrType additionalData);
    virtual void update(Persistable* persistable,
                        Storage::Ptr storage,
                        lsst::daf::data::DataProperty::PtrType additionalData);

    static lsst::daf::data::DataProperty::PtrType
        generateDataProperty(WCS const& wcs);
    static Formatter::Ptr createInstance(lsst::pex::policy::Policy::Ptr policy);

    template <class Archive>
    static void delegateSerialize(Archive& ar,
                                  int const version, Persistable* persistable);

private:
    explicit WcsFormatter(lsst::pex::policy::Policy::Ptr policy);

    static FormatterRegistration registration;
};

}}} // namespace lsst::afw::formatters

#endif
