// -*- lsst-c++ -*-
#ifndef LSST_AFW_FORMATTERS_WCSFORMATTER_H
#define LSST_AFW_FORMATTERS_WCSFORMATTER_H

/** @file
 * @brief Interface for WcsFormatter class
 *
 * @author $Author: ktlim $
 * @version $Revision: 2377 $
 * @date $Date: 2009-07-16 13:06:53 -0700 (Thu, 16 Jul 2009) $
 *
 * Contact: Kian-Tat Lim (ktl@slac.stanford.edu)
 * @ingroup afw
 */

/** @class lsst::afw::formatters::WcsFormatter
 * @brief Class implementing persistence and retrieval for Wcs objects.
 *
 * @ingroup afw
 */

#include "lsst/daf/base.h"
#include "lsst/daf/persistence.h"

#include "Eigen/Core.h"

namespace lsst {
namespace afw {
    namespace image {
        class Wcs;
    }
namespace formatters {

class WcsFormatter : public lsst::daf::persistence::Formatter {
public:       
    virtual ~WcsFormatter(void);

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

    static lsst::daf::base::PropertySet::Ptr generatePropertySet(
        lsst::afw::image::Wcs const& wcs
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
    explicit WcsFormatter(lsst::pex::policy::Policy::Ptr policy);

    static lsst::daf::persistence::FormatterRegistration registration;
};

}}} // namespace lsst::afw::formatters

#endif
