// -*- lsst-c++ -*-

/* 
 * LSST Data Management System
 * See the COPYRIGHT and LICENSE files in the top-level directory of this
 * package for notices and licensing terms.
 */
 
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

class KernelFormatter : public lsst::daf::persistence::Formatter {
public:
    virtual ~KernelFormatter(void);

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
    explicit KernelFormatter(lsst::pex::policy::Policy::Ptr policy);

    lsst::pex::policy::Policy::Ptr _policy;

    static lsst::daf::persistence::Formatter::Ptr
        createInstance(lsst::pex::policy::Policy::Ptr policy);

    static lsst::daf::persistence::FormatterRegistration kernelRegistration;
    static lsst::daf::persistence::FormatterRegistration fixedKernelRegistration;
    static lsst::daf::persistence::FormatterRegistration analyticKernelRegistration;
    static lsst::daf::persistence::FormatterRegistration deltaFunctionKernelRegistration;
    static lsst::daf::persistence::FormatterRegistration
        linearCombinationKernelRegistration;
    static lsst::daf::persistence::FormatterRegistration separableKernelRegistration;
};

}}} // namespace lsst::daf::persistence

#endif
