// -*- lsst-c++ -*-

/*
 * LSST Data Management System
 * Copyright 2008, 2009, 2010 LSST Corporation.
 *
 * This product includes software developed by the
 * LSST Project (http://www.lsst.org/).
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the LSST License Statement and
 * the GNU General Public License along with this program.  If not,
 * see <http://www.lsstcorp.org/LegalNotices/>.
 */

/*
 * Implementation of KernelFormatter class
 */

#ifndef __GNUC__
#define __attribute__(x) /*NOTHING*/
#endif
static char const* SVNid __attribute__((unused)) = "$Id$";

#include "lsst/afw/formatters/KernelFormatter.h"

#include <stdexcept>
#include <string>
#include <vector>

#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/xml_oarchive.hpp>
#include <boost/archive/xml_iarchive.hpp>
#include <boost/serialization/nvp.hpp>
#include <boost/serialization/vector.hpp>

#include <boost/serialization/export.hpp>

#include "lsst/afw/math/Kernel.h"
#include "lsst/afw/math/Function.h"
#include "lsst/afw/math/FunctionLibrary.h"
#include "lsst/daf/persistence/FormatterImpl.h"
#include "lsst/daf/persistence/LogicalLocation.h"
#include "lsst/daf/persistence/BoostStorage.h"
#include "lsst/daf/persistence/XmlStorage.h"
#include "lsst/log/Log.h"
#include <lsst/pex/exceptions.h>
#include <lsst/pex/policy/Policy.h>

BOOST_CLASS_EXPORT(lsst::afw::math::Kernel)
BOOST_CLASS_EXPORT(lsst::afw::math::FixedKernel)
BOOST_CLASS_EXPORT(lsst::afw::math::AnalyticKernel)
BOOST_CLASS_EXPORT(lsst::afw::math::DeltaFunctionKernel)
BOOST_CLASS_EXPORT(lsst::afw::math::LinearCombinationKernel)
BOOST_CLASS_EXPORT(lsst::afw::math::SeparableKernel)

BOOST_CLASS_EXPORT(lsst::afw::math::Function<float>)
BOOST_CLASS_EXPORT(lsst::afw::math::NullFunction1<float>)
BOOST_CLASS_EXPORT(lsst::afw::math::NullFunction2<float>)

BOOST_CLASS_EXPORT(lsst::afw::math::Function<double>)
BOOST_CLASS_EXPORT(lsst::afw::math::NullFunction1<double>)
BOOST_CLASS_EXPORT(lsst::afw::math::NullFunction2<double>)

BOOST_CLASS_EXPORT(lsst::afw::math::IntegerDeltaFunction2<float>)
BOOST_CLASS_EXPORT(lsst::afw::math::GaussianFunction1<float>)
BOOST_CLASS_EXPORT(lsst::afw::math::GaussianFunction2<float>)
BOOST_CLASS_EXPORT(lsst::afw::math::DoubleGaussianFunction2<float>)
BOOST_CLASS_EXPORT(lsst::afw::math::PolynomialFunction1<float>)
BOOST_CLASS_EXPORT(lsst::afw::math::PolynomialFunction2<float>)
BOOST_CLASS_EXPORT(lsst::afw::math::Chebyshev1Function1<float>)
BOOST_CLASS_EXPORT(lsst::afw::math::Chebyshev1Function2<float>)
BOOST_CLASS_EXPORT(lsst::afw::math::LanczosFunction1<float>)
BOOST_CLASS_EXPORT(lsst::afw::math::LanczosFunction2<float>)

BOOST_CLASS_EXPORT(lsst::afw::math::IntegerDeltaFunction2<double>)
BOOST_CLASS_EXPORT(lsst::afw::math::GaussianFunction1<double>)
BOOST_CLASS_EXPORT(lsst::afw::math::GaussianFunction2<double>)
BOOST_CLASS_EXPORT(lsst::afw::math::DoubleGaussianFunction2<double>)
BOOST_CLASS_EXPORT(lsst::afw::math::PolynomialFunction1<double>)
BOOST_CLASS_EXPORT(lsst::afw::math::PolynomialFunction2<double>)
BOOST_CLASS_EXPORT(lsst::afw::math::Chebyshev1Function1<double>)
BOOST_CLASS_EXPORT(lsst::afw::math::Chebyshev1Function2<double>)
BOOST_CLASS_EXPORT(lsst::afw::math::LanczosFunction1<double>)
BOOST_CLASS_EXPORT(lsst::afw::math::LanczosFunction2<double>)

namespace {
LOG_LOGGER _log = LOG_GET("afw.math.KernelFormatter");
}

namespace lsst {
namespace afw {
namespace formatters {

namespace dafBase = lsst::daf::base;
namespace dafPersist = lsst::daf::persistence;
namespace pexPolicy = lsst::pex::policy;

using boost::serialization::make_nvp;

dafPersist::FormatterRegistration KernelFormatter::kernelRegistration("Kernel", typeid(math::Kernel),
                                                                      createInstance);
dafPersist::FormatterRegistration KernelFormatter::fixedKernelRegistration("FixedKernel",
                                                                           typeid(math::FixedKernel),
                                                                           createInstance);
dafPersist::FormatterRegistration KernelFormatter::analyticKernelRegistration("AnalyticKernel",
                                                                              typeid(math::AnalyticKernel),
                                                                              createInstance);
dafPersist::FormatterRegistration KernelFormatter::deltaFunctionKernelRegistration(
        "DeltaFunctionKernel", typeid(math::DeltaFunctionKernel), createInstance);
dafPersist::FormatterRegistration KernelFormatter::linearCombinationKernelRegistration(
        "LinearCombinationKernel", typeid(math::LinearCombinationKernel), createInstance);
dafPersist::FormatterRegistration KernelFormatter::separableKernelRegistration("SeparableKernel",
                                                                               typeid(math::SeparableKernel),
                                                                               createInstance);

KernelFormatter::KernelFormatter(std::shared_ptr<pexPolicy::Policy> policy)
        : dafPersist::Formatter(typeid(this)), _policy(policy) {}

KernelFormatter::KernelFormatter(KernelFormatter const&) = default;
KernelFormatter::KernelFormatter(KernelFormatter&&) = default;
KernelFormatter& KernelFormatter::operator=(KernelFormatter const&) = default;
KernelFormatter& KernelFormatter::operator=(KernelFormatter&&) = default;

KernelFormatter::~KernelFormatter() = default;

void KernelFormatter::write(dafBase::Persistable const* persistable,
                            std::shared_ptr<dafPersist::FormatterStorage> storage,
                            std::shared_ptr<dafBase::PropertySet>) {
    LOGL_DEBUG(_log, "KernelFormatter write start");
    math::Kernel const* kp = dynamic_cast<math::Kernel const*>(persistable);
    if (kp == 0) {
        throw LSST_EXCEPT(pex::exceptions::RuntimeError, "Persisting non-Kernel");
    }
    // TODO: Replace this with something better in DM-10776
    auto boost = std::dynamic_pointer_cast<dafPersist::BoostStorage>(storage);
    if (boost) {
        LOGL_DEBUG(_log, "KernelFormatter write BoostStorage");
        boost->getOArchive() & kp;
        LOGL_DEBUG(_log, "KernelFormatter write end");
        return;
    }
    auto xml = std::dynamic_pointer_cast<dafPersist::XmlStorage>(storage);
    if (xml) {
        LOGL_DEBUG(_log, "KernelFormatter write XmlStorage");
        xml->getOArchive() & make_nvp("ptr", kp);
        LOGL_DEBUG(_log, "KernelFormatter write end");
        return;
    }
    throw LSST_EXCEPT(pex::exceptions::RuntimeError, "Unrecognized FormatterStorage for Kernel");
}

dafBase::Persistable* KernelFormatter::read(std::shared_ptr<dafPersist::FormatterStorage> storage,
                                            std::shared_ptr<dafBase::PropertySet>) {
    LOGL_DEBUG(_log, "KernelFormatter read start");
    math::Kernel* kp;
    // TODO: Replace this with something better in DM-10776
    auto boost = std::dynamic_pointer_cast<dafPersist::BoostStorage>(storage);
    if (boost) {
        LOGL_DEBUG(_log, "KernelFormatter read BoostStorage");
        boost->getIArchive() & kp;
        LOGL_DEBUG(_log, "KernelFormatter read end");
        return kp;
    }
    auto xml = std::dynamic_pointer_cast<dafPersist::XmlStorage>(storage);
    if (xml) {
        LOGL_DEBUG(_log, "KernelFormatter read XmlStorage");
        xml->getIArchive() & make_nvp("ptr", kp);
        LOGL_DEBUG(_log, "KernelFormatter read end");
        return kp;
    }
    throw LSST_EXCEPT(pex::exceptions::RuntimeError, "Unrecognized FormatterStorage for Kernel");
}

void KernelFormatter::update(dafBase::Persistable*, std::shared_ptr<dafPersist::FormatterStorage>,
                             std::shared_ptr<dafBase::PropertySet>) {
    throw LSST_EXCEPT(pex::exceptions::RuntimeError, "Unexpected call to update for Kernel");
}

template <class Archive>
void KernelFormatter::delegateSerialize(Archive& ar, unsigned int const, dafBase::Persistable* persistable) {
    LOGL_DEBUG(_log, "KernelFormatter delegateSerialize start");
    math::Kernel* kp = dynamic_cast<math::Kernel*>(persistable);
    if (kp == 0) {
        throw LSST_EXCEPT(pex::exceptions::RuntimeError, "Serializing non-Kernel");
    }
    ar& make_nvp("base", boost::serialization::base_object<dafBase::Persistable>(*kp));
    ar& make_nvp("width", kp->_width);
    ar& make_nvp("height", kp->_height);
    ar& make_nvp("ctrX", kp->_ctrX);
    ar& make_nvp("ctrY", kp->_ctrY);
    ar& make_nvp("nParams", kp->_nKernelParams);
    ar& make_nvp("spatialFunctionList", kp->_spatialFunctionList);

    LOGL_DEBUG(_log, "KernelFormatter delegateSerialize end");
}

// Explicit template specializations confuse Doxygen, tell it to ignore them
/// @cond
template void KernelFormatter::delegateSerialize(boost::archive::text_oarchive& ar, unsigned int const,
                                                 dafBase::Persistable*);
template void KernelFormatter::delegateSerialize(boost::archive::text_iarchive& ar, unsigned int const,
                                                 dafBase::Persistable*);
template void KernelFormatter::delegateSerialize(boost::archive::xml_oarchive& ar, unsigned int const,
                                                 dafBase::Persistable*);
template void KernelFormatter::delegateSerialize(boost::archive::xml_iarchive& ar, unsigned int const,
                                                 dafBase::Persistable*);
template void KernelFormatter::delegateSerialize(boost::archive::binary_oarchive& ar, unsigned int const,
                                                 dafBase::Persistable*);
template void KernelFormatter::delegateSerialize(boost::archive::binary_iarchive& ar, unsigned int const,
                                                 dafBase::Persistable*);
/// @endcond

std::shared_ptr<dafPersist::Formatter> KernelFormatter::createInstance(
        std::shared_ptr<pexPolicy::Policy> policy) {
    return std::shared_ptr<dafPersist::Formatter>(new KernelFormatter(policy));
}
}  // namespace formatters
}  // namespace afw
}  // namespace lsst
