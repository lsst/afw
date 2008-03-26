// -*- lsst-c++ -*-
#ifndef LSST_FW_FORMATTERS_EXPOSUREFORMATTER_H
#define LSST_FW_FORMATTERS_EXPOSUREFORMATTER_H

/** \file
 * \brief Interface for ExposureFormatter class
 *
 * \author $Author: ktlim $
 * \version $Revision: 2377 $
 * \date $Date$
 *
 * Contact: Kian-Tat Lim (ktl@slac.stanford.edu)
 * \ingroup fw
 */

/** \class lsst::fw::formatters::ExposureFormatter
 * \brief Class implementing persistence and retrieval for Exposures.
 *
 * \ingroup fw
 */

#include "lsst/mwi/persistence/Formatter.h"

namespace lsst {
namespace fw {
namespace formatters {
               
using namespace lsst::mwi::persistence;

template<typename ImagePixelT, typename MaskPixelT>
class ExposureFormatter : public Formatter {
public:       
    virtual ~ExposureFormatter(void);

    virtual void write(Persistable const* persistable, Storage::Ptr storage,
                       lsst::mwi::data::DataProperty::PtrType additionalData);
    virtual Persistable* read(
        Storage::Ptr storage,
        lsst::mwi::data::DataProperty::PtrType additionalData);
    virtual void update(Persistable* persistable,
                        Storage::Ptr storage,
                        lsst::mwi::data::DataProperty::PtrType additionalData);

    static Formatter::Ptr createInstance(lsst::mwi::policy::Policy::Ptr policy);

    template <class Archive>
        static void delegateSerialize(Archive& ar, int const version,
                                      Persistable* persistable);
private:
    explicit ExposureFormatter(lsst::mwi::policy::Policy::Ptr policy);

    lsst::mwi::policy::Policy::Ptr _policy;

    static FormatterRegistration registration;
};

}}} // namespace lsst::fw::formatters

#endif
