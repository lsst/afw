// -*- lsst-c++ -*-
#ifndef LSST_FW_FORMATTERS_MASKFORMATTER_H
#define LSST_FW_FORMATTERS_MASKFORMATTER_H

/** \file
 * \brief Interface for MaskFormatter class
 *
 * \author $Author: ktlim $
 * \version $Revision: 2377 $
 * \date $Date$
 *
 * Contact: Kian-Tat Lim (ktl@slac.stanford.edu)
 * \ingroup fw
 */

/** \class lsst::fw::formatters::MaskFormatter
 * \brief Class implementing persistence and retrieval for Masks.
 *
 * \ingroup fw
 */

#include "lsst/mwi/persistence/Formatter.h"

namespace lsst {
namespace fw {
namespace formatters {
               
using namespace lsst::mwi::persistence;

template<typename MaskPixelT>
class MaskFormatter : public Formatter {
public:       
    virtual ~MaskFormatter(void);

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
    static void delegateSerialize(Archive& ar,
                                  int const version, Persistable* persistable);

private:
    explicit MaskFormatter(lsst::mwi::policy::Policy::Ptr policy);

    static FormatterRegistration registration;
};

}}} // namespace lsst::fw::formatters

#endif
