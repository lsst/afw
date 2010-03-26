// -*- LSST-C++ -*-
#if !defined(LSST_MEAS_ALGORITHMS_PSFFORMATTER_H)
#define LSST_MEAS_ALGORITHMS_PSFFORMATTER_H

/** @file
 * @brief Interface for PsfFormatter class
 *
 * @version $Revision: 2150 $
 * @date $Date$
 *
 * Contact: Kian-Tat Lim (ktl@slac.stanford.edu)
 * @ingroup meas_algorithms
 */

/** @class lsst::meas::algorithms::PsfFormatter
 * @brief Formatter for persistence of PSF instances.
 *
 * @ingroup meas_algorithms
 */

#include "lsst/daf/base/Persistable.h"
#include "lsst/daf/persistence/Formatter.h"
#include "lsst/daf/persistence/Storage.h"
#include "lsst/pex/policy/Policy.h"

namespace lsst {
namespace meas {
namespace algorithms {

namespace dafBase = lsst::daf::base;
namespace pexPolicy = lsst::pex::policy;
namespace dafPersist = lsst::daf::persistence;

class PsfFormatter : public dafPersist::Formatter {
public:
    virtual ~PsfFormatter(void);

    virtual void write(dafBase::Persistable const* persistable,
        dafPersist::Storage::Ptr storage,
        dafBase::PropertySet::Ptr additionalData);

    virtual dafBase::Persistable* read(dafPersist::Storage::Ptr storage,
        dafBase::PropertySet::Ptr additionalData);

    virtual void update(dafBase::Persistable* persistable,
        dafPersist::Storage::Ptr storage,
        dafBase::PropertySet::Ptr additionalData);

    template <class Archive>
    static void delegateSerialize(Archive& ar, unsigned int const version,
        dafBase::Persistable* persistable);

private:
    explicit PsfFormatter(pexPolicy::Policy::Ptr policy);

    pexPolicy::Policy::Ptr _policy;

    static dafPersist::Formatter::Ptr
        createInstance(pexPolicy::Policy::Ptr policy);

    static dafPersist::FormatterRegistration registration;
    static dafPersist::FormatterRegistration dgPsfRegistration;
    static dafPersist::FormatterRegistration pcaPsfRegistration;
};

}}} // namespace lsst::meas::algorithms

#endif
