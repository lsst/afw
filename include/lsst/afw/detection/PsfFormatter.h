// -*- LSST-C++ -*-
#if !defined(LSST_AFW_DETECTION_PSFFORMATTER_H)
#define LSST_AFW_DETECTION_PSFFORMATTER_H

/** @file
 * @brief Interface for PsfFormatter class
 *
 * @version $Revision: 2150 $
 * @date $Date$
 *
 * Contact: Kian-Tat Lim (ktl@slac.stanford.edu)
 * @ingroup afw
 */

/** @class lsst::afw::detection::PsfFormatter
 * @brief Formatter for persistence of Psf instances.
 *
 * @ingroup afw
 */

#include "lsst/daf/base/Persistable.h"
#include "lsst/daf/persistence/Formatter.h"
#include "lsst/daf/persistence/Storage.h"
#include "lsst/log/Log.h"
#include "lsst/pex/policy/Policy.h"

namespace lsst {
namespace afw {
namespace detection {

class PsfFormatter : public lsst::daf::persistence::Formatter {
public:
    virtual ~PsfFormatter(void);

    virtual void write(lsst::daf::base::Persistable const* persistable,
        lsst::daf::persistence::Storage::Ptr storage,
        lsst::daf::base::PropertySet::Ptr additionalData);

    virtual lsst::daf::base::Persistable* read(lsst::daf::persistence::Storage::Ptr storage,
        lsst::daf::base::PropertySet::Ptr additionalData);

    virtual void update(lsst::daf::base::Persistable* persistable,
        lsst::daf::persistence::Storage::Ptr storage,
        lsst::daf::base::PropertySet::Ptr additionalData);

    template <class Archive>
    static void delegateSerialize(Archive& ar, unsigned int const version,
        lsst::daf::base::Persistable* persistable);

private:
    explicit PsfFormatter(lsst::pex::policy::Policy::Ptr policy);

    lsst::pex::policy::Policy::Ptr _policy;
    lsst::log::Log _log{"afw.detection.PsfFormatter"};

    static lsst::daf::persistence::Formatter::Ptr
        createInstance(lsst::pex::policy::Policy::Ptr policy);

    static lsst::daf::persistence::FormatterRegistration registration;
    static lsst::daf::persistence::FormatterRegistration doubleGaussianPsfRegistration;
    static lsst::daf::persistence::FormatterRegistration pcaPsfRegistration;
};

}}}

#endif
