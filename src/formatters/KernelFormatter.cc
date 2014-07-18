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
 


/** \file
 * \brief Implementation of KernelFormatter class
 *
 * \version $Revision: 2151 $
 * \date $Date$
 *
 * Contact: Kian-Tat Lim (ktl@slac.stanford.edu)
 *
 * \ingroup afw_math
 */

#ifndef __GNUC__
#  define __attribute__(x) /*NOTHING*/
#endif
static char const* SVNid __attribute__((unused)) =
    "$Id$";

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
#include <lsst/pex/exceptions.h>
#include <lsst/pex/logging/Trace.h>
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

#define EXEC_TRACE  20
static void execTrace(std::string s, int level = EXEC_TRACE) {
    lsst::pex::logging::Trace("afw.math.KernelFormatter", level, s);
}

namespace afwMath = lsst::afw::math;
namespace afwForm = lsst::afw::formatters;
namespace dafBase = lsst::daf::base;
namespace dafPersist = lsst::daf::persistence;
namespace pexPolicy = lsst::pex::policy;

using boost::serialization::make_nvp;

/** Register this Formatter subclass through a static instance of
 * FormatterRegistration.
 */
dafPersist::FormatterRegistration
afwForm::KernelFormatter::kernelRegistration(
    "Kernel", typeid(afwMath::Kernel), createInstance);
dafPersist::FormatterRegistration
afwForm::KernelFormatter::fixedKernelRegistration(
    "FixedKernel", typeid(afwMath::FixedKernel), createInstance);
dafPersist::FormatterRegistration
afwForm::KernelFormatter::analyticKernelRegistration(
    "AnalyticKernel", typeid(afwMath::AnalyticKernel), createInstance);
dafPersist::FormatterRegistration
afwForm::KernelFormatter::deltaFunctionKernelRegistration(
    "DeltaFunctionKernel", typeid(afwMath::DeltaFunctionKernel),
    createInstance);
dafPersist::FormatterRegistration
afwForm::KernelFormatter::linearCombinationKernelRegistration(
    "LinearCombinationKernel", typeid(afwMath::LinearCombinationKernel),
    createInstance);
dafPersist::FormatterRegistration
afwForm::KernelFormatter::separableKernelRegistration(
    "SeparableKernel", typeid(afwMath::SeparableKernel), createInstance);

/** Constructor.
 * \param[in] policy Policy for configuring this Formatter
 */
afwForm::KernelFormatter::KernelFormatter(
    pexPolicy::Policy::Ptr policy) :
    dafPersist::Formatter(typeid(this)), _policy(policy) {
}

/** Minimal destructor.
 */
afwForm::KernelFormatter::~KernelFormatter(void) {
}

void afwForm::KernelFormatter::write(
    dafBase::Persistable const* persistable,
    dafPersist::Storage::Ptr storage,
    dafBase::PropertySet::Ptr) {
    execTrace("KernelFormatter write start");
    afwMath::Kernel const* kp =
        dynamic_cast<afwMath::Kernel const*>(persistable);
    if (kp == 0) {
        throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeError, "Persisting non-Kernel");
    }
    if (typeid(*storage) == typeid(dafPersist::BoostStorage)) {
        execTrace("KernelFormatter write BoostStorage");
        dafPersist::BoostStorage* boost =
            dynamic_cast<dafPersist::BoostStorage*>(storage.get());
        boost->getOArchive() & kp;
        execTrace("KernelFormatter write end");
        return;
    }
    else if (typeid(*storage) == typeid(dafPersist::XmlStorage)) {
        execTrace("KernelFormatter write XmlStorage");
        dafPersist::XmlStorage* xml =
            dynamic_cast<dafPersist::XmlStorage*>(storage.get());
        xml->getOArchive() & make_nvp("ptr", kp);
        execTrace("KernelFormatter write end");
        return;
    }
    throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeError, "Unrecognized Storage for Kernel");
}

dafBase::Persistable* afwForm::KernelFormatter::read(
    dafPersist::Storage::Ptr storage, dafBase::PropertySet::Ptr) {
    execTrace("KernelFormatter read start");
    afwMath::Kernel* kp;
    if (typeid(*storage) == typeid(dafPersist::BoostStorage)) {
        execTrace("KernelFormatter read BoostStorage");
        dafPersist::BoostStorage* boost =
            dynamic_cast<dafPersist::BoostStorage*>(storage.get());
        boost->getIArchive() & kp;
        execTrace("KernelFormatter read end");
        return kp;
    }
    else if (typeid(*storage) == typeid(dafPersist::XmlStorage)) {
        execTrace("KernelFormatter read XmlStorage");
        dafPersist::XmlStorage* xml =
            dynamic_cast<dafPersist::XmlStorage*>(storage.get());
        xml->getIArchive() & make_nvp("ptr", kp);
        execTrace("KernelFormatter read end");
        return kp;
    }
    throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeError, "Unrecognized Storage for Kernel");
}

void afwForm::KernelFormatter::update(dafBase::Persistable*,
                                   dafPersist::Storage::Ptr,
                                   dafBase::PropertySet::Ptr) {
    throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeError, "Unexpected call to update for Kernel");
}

/** Serialize a Kernel to a Boost archive.  Handles text or XML
 * archives, input or output.
 * \param[in,out] ar Boost archive
 * \param[in] version Version of the Kernel class
 * \param[in,out] persistable Pointer to the Kernel as a Persistable
 */
template <class Archive>
void afwForm::KernelFormatter::delegateSerialize(
    Archive& ar, unsigned int const, dafBase::Persistable* persistable) {
    execTrace("KernelFormatter delegateSerialize start");
    afwMath::Kernel* kp =
        dynamic_cast<afwMath::Kernel*>(persistable);
    if (kp == 0) {
        throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeError, "Serializing non-Kernel");
    }
    ar & make_nvp("base",
                  boost::serialization::base_object<dafBase::Persistable>(*kp));
    ar & make_nvp("width", kp->_width);
    ar & make_nvp("height", kp->_height);
    ar & make_nvp("ctrX", kp->_ctrX);
    ar & make_nvp("ctrY", kp->_ctrY);
    ar & make_nvp("nParams", kp->_nKernelParams);
    ar & make_nvp("spatialFunctionList", kp->_spatialFunctionList);
                  
    execTrace("KernelFormatter delegateSerialize end");
}

template void afwForm::KernelFormatter::delegateSerialize(
    boost::archive::text_oarchive& ar, unsigned int const, dafBase::Persistable*);
template void afwForm::KernelFormatter::delegateSerialize(
    boost::archive::text_iarchive& ar, unsigned int const, dafBase::Persistable*);
template void afwForm::KernelFormatter::delegateSerialize(
    boost::archive::xml_oarchive& ar, unsigned int const, dafBase::Persistable*);
template void afwForm::KernelFormatter::delegateSerialize(
    boost::archive::xml_iarchive& ar, unsigned int const, dafBase::Persistable*);
template void afwForm::KernelFormatter::delegateSerialize(
    boost::archive::binary_oarchive& ar, unsigned int const, dafBase::Persistable*);
template void afwForm::KernelFormatter::delegateSerialize(
    boost::archive::binary_iarchive& ar, unsigned int const, dafBase::Persistable*);


/** Factory method for KernelFormatter.
 * \param[in] policy Policy for configuring the KernelFormatter
 * \return Shared pointer to a new instance
 */
dafPersist::Formatter::Ptr afwForm::KernelFormatter::createInstance(
    pexPolicy::Policy::Ptr policy) {
    return dafPersist::Formatter::Ptr(new afwForm::KernelFormatter(policy));
}
