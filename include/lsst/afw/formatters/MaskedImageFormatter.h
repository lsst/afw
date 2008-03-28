// -*- lsst-c++ -*-
#ifndef LSST_AFW_FORMATTERS_MASKEDIMAGEFORMATTER_H
#define LSST_AFW_FORMATTERS_MASKEDIMAGEFORMATTER_H

/** \file
 * \brief Interface for MaskedImageFormatter class
 *
 * \author $Author: ktlim $
 * \version $Revision: 2377 $
 * \date $Date$
 *
 * Contact: Kian-Tat Lim (ktl@slac.stanford.edu)
 * \ingroup afw
 */

/** \class lsst::afw::formatters::MaskedImageFormatter
 * \brief Class implementing persistence and retrieval for MaskedImages.
 *
 * \ingroup afw
 */

#include "lsst/daf/persistence/Formatter.h"

namespace lsst {
namespace afw {
namespace formatters {
               
using namespace lsst::daf::persitence;

template<typename ImagePixelT, typename MaskPixelT>
class MaskedImageFormatter : public Formatter {
public:       
    virtual ~MaskedImageFormatter(void);

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
    explicit MaskedImageFormatter(lsst::pex::policy::Policy::Ptr policy);

    static FormatterRegistration registration;
};

}}} // namespace lsst::afw::formatters

#endif
