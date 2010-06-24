// -*- lsst-c++ -*-
#ifndef LSST_AFW_MATH_KERNELFORMATTER_H
#define LSST_AFW_MATH_KERNELFORMATTER_H

/** @file
 * @brief Interface for KernelFormatter class
 *
 * @version $Revision: 2150 $
 * @date $Date$
 *
 * Contact: Kian-Tat Lim (ktl@slac.stanford.edu)
 * @ingroup afw_math
 */

/** @class lsst::afw::math::KernelFormatter
 * @brief Formatter for persistence of Kernel instances.
 *
 * @ingroup afw_math
 */

#include <lsst/afw/math/Kernel.h>
#include <lsst/daf/base/Persistable.h>
#include <lsst/daf/persistence/Formatter.h>
#include <lsst/daf/persistence/Storage.h>
#include <lsst/pex/policy/Policy.h>

namespace lsst {
namespace afw {
namespace formatters {

namespace dafBase = lsst::daf::base;
namespace pexPolicy = lsst::pex::policy;
namespace dafPersist = lsst::daf::persistence;

class KernelFormatter : public dafPersist::Formatter {
public:
    virtual ~KernelFormatter(void);

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
    explicit KernelFormatter(pexPolicy::Policy::Ptr policy);

    pexPolicy::Policy::Ptr _policy;

    static dafPersist::Formatter::Ptr
        createInstance(pexPolicy::Policy::Ptr policy);

    static dafPersist::FormatterRegistration kernelRegistration;
    static dafPersist::FormatterRegistration fixedKernelRegistration;
    static dafPersist::FormatterRegistration analyticKernelRegistration;
    static dafPersist::FormatterRegistration deltaFunctionKernelRegistration;
    static dafPersist::FormatterRegistration
        linearCombinationKernelRegistration;
    static dafPersist::FormatterRegistration separableKernelRegistration;
};

}}} // namespace lsst::daf::persistence

#endif
