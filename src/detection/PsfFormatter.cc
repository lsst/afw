// -*- LSST-C++ -*-
/*
 * Implementation of PsfFormatter class
 */
#include <stdexcept>
#include <string>
#include <vector>

#include "boost/serialization/nvp.hpp"

#include "lsst/afw/detection/PsfFormatter.h"
#include "lsst/afw/detection/Psf.h"
#include "lsst/daf/persistence/FormatterImpl.h"
#include "lsst/daf/persistence/LogicalLocation.h"
#include "lsst/daf/persistence/BoostStorage.h"
#include "lsst/daf/persistence/XmlStorage.h"
#include "lsst/pex/exceptions.h"
#include "lsst/log/Log.h"
#include "lsst/pex/policy/Policy.h"

BOOST_CLASS_EXPORT(lsst::afw::detection::Psf)

namespace {
LOG_LOGGER _log = LOG_GET("afw.detection.PsfFormatter");
}

namespace afwDetect = lsst::afw::detection;
namespace afwMath = lsst::afw::math;
namespace dafBase = lsst::daf::base;
namespace dafPersist = lsst::daf::persistence;
namespace pexPolicy = lsst::pex::policy;

using boost::serialization::make_nvp;

dafPersist::FormatterRegistration
afwDetect::PsfFormatter::registration("Psf", typeid(afwDetect::Psf), createInstance);

afwDetect::PsfFormatter::PsfFormatter(
    std::shared_ptr<pexPolicy::Policy> policy) :
    dafPersist::Formatter(typeid(this)), _policy(policy) {}

afwDetect::PsfFormatter::~PsfFormatter(void) {}

void afwDetect::PsfFormatter::write(
    dafBase::Persistable const* persistable,
    std::shared_ptr<dafPersist::Storage> storage,
    std::shared_ptr<dafBase::PropertySet>) {
    LOGL_DEBUG(_log, "PsfFormatter write start");
    afwDetect::Psf const* ps = dynamic_cast<afwDetect::Psf const*>(persistable);
    if (ps == 0) {
        throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeError, "Persisting non-Psf");
    }
    if (typeid(*storage) == typeid(dafPersist::BoostStorage)) {
        LOGL_DEBUG(_log, "PsfFormatter write BoostStorage");
        dafPersist::BoostStorage* boost =
            dynamic_cast<dafPersist::BoostStorage*>(storage.get());
        boost->getOArchive() & ps;
        LOGL_DEBUG(_log, "PsfFormatter write end");
        return;
    }
    else if (typeid(*storage) == typeid(dafPersist::XmlStorage)) {
        LOGL_DEBUG(_log, "PsfFormatter write XmlStorage");
        dafPersist::XmlStorage* xml =
            dynamic_cast<dafPersist::XmlStorage*>(storage.get());
        xml->getOArchive() & make_nvp("psf", ps);
        LOGL_DEBUG(_log, "PsfFormatter write end");
        return;
    }
    throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeError, "Unrecognized Storage for Psf");
}

dafBase::Persistable* afwDetect::PsfFormatter::read(
    std::shared_ptr<dafPersist::Storage> storage, std::shared_ptr<dafBase::PropertySet>) {
    LOGL_DEBUG(_log, "PsfFormatter read start");
    afwDetect::Psf* ps;
    if (typeid(*storage) == typeid(dafPersist::BoostStorage)) {
        LOGL_DEBUG(_log, "PsfFormatter read BoostStorage");
        dafPersist::BoostStorage* boost =
            dynamic_cast<dafPersist::BoostStorage*>(storage.get());
        boost->getIArchive() & ps;
        LOGL_DEBUG(_log, "PsfFormatter read end");
        return ps;
    }
    else if (typeid(*storage) == typeid(dafPersist::XmlStorage)) {
        LOGL_DEBUG(_log, "PsfFormatter read XmlStorage");
        dafPersist::XmlStorage* xml =
            dynamic_cast<dafPersist::XmlStorage*>(storage.get());
        xml->getIArchive() & make_nvp("psf", ps);
        LOGL_DEBUG(_log, "PsfFormatter read end");
        return ps;
    }
    throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeError, "Unrecognized Storage for Psf");
}

void afwDetect::PsfFormatter::update(dafBase::Persistable* ,
                                   std::shared_ptr<dafPersist::Storage>,
                                   std::shared_ptr<dafBase::PropertySet>) {
    throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeError, "Unexpected call to update for Psf");
}

template <class Archive>
void afwDetect::PsfFormatter::delegateSerialize(
        Archive& ar,
        unsigned int const,
        dafBase::Persistable* persistable
                                               ) {
    LOGL_DEBUG(_log, "PsfFormatter delegateSerialize start");
    afwDetect::Psf* ps = dynamic_cast<afwDetect::Psf*>(persistable);
    if (ps == 0) {
        throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeError, "Serializing non-Psf");
    }
#if 0                                   // not present in baseclass
    ar & make_nvp("width", ps->_width) & make_nvp("height", ps->_height);
    ar & make_nvp("k", ps->_kernel);
#endif

    LOGL_DEBUG(_log, "PsfFormatter delegateSerialize end");
}

std::shared_ptr<dafPersist::Formatter> afwDetect::PsfFormatter::createInstance(
    std::shared_ptr<pexPolicy::Policy> policy) {
    return std::shared_ptr<dafPersist::Formatter>(new afwDetect::PsfFormatter(policy));
}
