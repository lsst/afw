// -*- lsst-c++ -*-
#ifndef LSST_AFW_FORMATTERS_EXPOSUREFORMATTER_H
#define LSST_AFW_FORMATTERS_EXPOSUREFORMATTER_H

/** \file
 * \brief Interface for ExposureFormatter class
 *
 * \author $Author: ktlim $
 * \version $Revision: 2377 $
 * \date $Date$
 *
 * Contact: Kian-Tat Lim (ktl@slac.stanford.edu)
 * \ingroup afw
 */

/** \class lsst::afw::formatters::ExposureFormatter
 * \brief Class implementing persistence and retrieval for Exposures.
 *
 * \ingroup afw
 */

#include "lsst/daf/persistence/Formatter.h"

namespace lsst {
namespace afw {
namespace formatters {
               
using namespace lsst::daf::persitence;

template<typename ImagePixelT, typename MaskPixelT>
class ExposureFormatter : public Formatter {
public:       
    virtual ~ExposureFormatter(void);

    virtual void write(Persistable const* persistable, Storage::Ptr storage,
                       lsst::daf::data::DataProperty::PtrType additionalData);
    virtual Persistable* read(
        Storage::Ptr storage,
        lsst::daf::data::DataProperty::PtrType additionalData);
    virtual void update(Persistable* persistable,
                        Storage::Ptr storage,
                        lsst::daf::data::DataProperty::PtrType additionalData);

    static Formatter::Ptr createInstance(lsst::pex::policy::Policy::Ptr policy);

    template <class Archive>
        static void delegateSerialize(Archive& ar, int const version,
                                      Persistable* persistable);
private:
    explicit ExposureFormatter(lsst::pex::policy::Policy::Ptr policy);

    lsst::pex::policy::Policy::Ptr _policy;

    static FormatterRegistration registration;
};

}}} // namespace lsst::afw::formatters

#endif
