// -*- lsst-c++ -*-

/* 
 * LSST Data Management System
 * See the COPYRIGHT and LICENSE files in the top-level directory of this
 * package for notices and licensing terms.
 */
 
#ifndef LSST_AFW_FORMATTERS_DECORATEDIMAGEFORMATTER_H
#define LSST_AFW_FORMATTERS_DECORATEDIMAGEFORMATTER_H

/** @file
 * @brief Interface for DecoratedImageFormatter class
 *
 * @author $Author: ktlim $
 * @version $Revision: 2377 $
 * @date $Date$
 *
 * Contact: Kian-Tat Lim (ktl@slac.stanford.edu)
 * @ingroup afw
 */

/** @class lsst::afw::formatters::DecoratedImageFormatter
 * @brief Class implementing persistence and retrieval for DecoratedImages.
 *
 * @ingroup afw
 */

#include "lsst/daf/base.h"
#include "lsst/daf/persistence.h"
#include "lsst/pex/policy/Policy.h"

namespace lsst {
namespace afw {
namespace formatters {

template<typename ImagePixelT>
class DecoratedImageFormatter : public lsst::daf::persistence::Formatter {
public:       
    virtual ~DecoratedImageFormatter(void);

    virtual void write(
        lsst::daf::base::Persistable const* persistable,
        lsst::daf::persistence::Storage::Ptr storage,
        lsst::daf::base::PropertySet::Ptr additionalData
    );
    virtual lsst::daf::base::Persistable* read(
        lsst::daf::persistence::Storage::Ptr storage,
        lsst::daf::base::PropertySet::Ptr additionalData
    );
    virtual void update(
        lsst::daf::base::Persistable* persistable,
        lsst::daf::persistence::Storage::Ptr storage,
        lsst::daf::base::PropertySet::Ptr additionalData
    );

    static lsst::daf::persistence::Formatter::Ptr createInstance(
        lsst::pex::policy::Policy::Ptr policy
    );

    template <class Archive>
    static void delegateSerialize(
        Archive& ar,
        int const version,
        lsst::daf::base::Persistable* persistable
    );

private:
    explicit DecoratedImageFormatter(lsst::pex::policy::Policy::Ptr policy);

    static lsst::daf::persistence::FormatterRegistration registration;
};

}}} // namespace lsst::afw::formatters

#endif
