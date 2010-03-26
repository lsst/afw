// -*- LSST-C++ -*-
/** \file
 * \brief Implementation of PsfFormatter class
 *
 * \version $Revision: 2151 $
 * \date $Date$
 *
 * Contact: Kian-Tat Lim (ktl@slac.stanford.edu)
 *
 * \ingroup meas_algorithms
 */
#include <stdexcept>
#include <string>
#include <vector>

#include "boost/serialization/nvp.hpp"

#include "lsst/meas/algorithms/PsfFormatter.h"

#include "lsst/meas/algorithms/PSF.h"
#include "lsst/meas/algorithms/detail/dgPsf.h"
#include "lsst/meas/algorithms/detail/pcaPsf.h"
#include "lsst/daf/persistence/FormatterImpl.h"
#include "lsst/daf/persistence/LogicalLocation.h"
#include "lsst/daf/persistence/BoostStorage.h"
#include "lsst/daf/persistence/XmlStorage.h"
#include "lsst/pex/exceptions.h"
#include "lsst/pex/logging/Trace.h"
#include "lsst/pex/policy/Policy.h"

BOOST_CLASS_EXPORT(lsst::meas::algorithms::PSF)
BOOST_CLASS_EXPORT(lsst::meas::algorithms::dgPsf)
BOOST_CLASS_EXPORT(lsst::meas::algorithms::pcaPsf)


#define EXEC_TRACE  20
static void execTrace(std::string s, int level = EXEC_TRACE) {
    lsst::pex::logging::Trace("meas.algorithms.PsfFormatter", level, s);
}

namespace measAlgo = lsst::meas::algorithms;
namespace afwMath = lsst::afw::math;
namespace dafBase = lsst::daf::base;
namespace dafPersist = lsst::daf::persistence;
namespace pexPolicy = lsst::pex::policy;

using boost::serialization::make_nvp;

/** Register this Formatter subclass through a static instance of
 * FormatterRegistration.
 */
dafPersist::FormatterRegistration
measAlgo::PsfFormatter::registration(
    "PSF", typeid(measAlgo::PSF), createInstance);
dafPersist::FormatterRegistration
measAlgo::PsfFormatter::dgPsfRegistration(
    "dgPsf", typeid(measAlgo::dgPsf), createInstance);
dafPersist::FormatterRegistration
measAlgo::PsfFormatter::pcaPsfRegistration(
    "pcaPsf", typeid(measAlgo::pcaPsf), createInstance);

/** Constructor.
 * \param[in] policy Policy for configuring this Formatter
 */
measAlgo::PsfFormatter::PsfFormatter(
    pexPolicy::Policy::Ptr policy) :
    dafPersist::Formatter(typeid(this)), _policy(policy) {}

/** Minimal destructor.
 */
measAlgo::PsfFormatter::~PsfFormatter(void) {}

void measAlgo::PsfFormatter::write(
    dafBase::Persistable const* persistable,
    dafPersist::Storage::Ptr storage,
    dafBase::PropertySet::Ptr) {
    execTrace("PsfFormatter write start");
    measAlgo::PSF const* ps = dynamic_cast<measAlgo::PSF const*>(persistable);
    if (ps == 0) {
        throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeErrorException, "Persisting non-PSF");
    }
    if (typeid(*storage) == typeid(dafPersist::BoostStorage)) {
        execTrace("PsfFormatter write BoostStorage");
        dafPersist::BoostStorage* boost =
            dynamic_cast<dafPersist::BoostStorage*>(storage.get());
        boost->getOArchive() & ps;
        execTrace("PsfFormatter write end");
        return;
    }
    else if (typeid(*storage) == typeid(dafPersist::XmlStorage)) {
        execTrace("PsfFormatter write XmlStorage");
        dafPersist::XmlStorage* xml =
            dynamic_cast<dafPersist::XmlStorage*>(storage.get());
        xml->getOArchive() & make_nvp("psf", ps);
        execTrace("PsfFormatter write end");
        return;
    }
    throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeErrorException, "Unrecognized Storage for PSF");
}

dafBase::Persistable* measAlgo::PsfFormatter::read(
    dafPersist::Storage::Ptr storage, dafBase::PropertySet::Ptr) {
    execTrace("PsfFormatter read start");
    measAlgo::PSF* ps;
    if (typeid(*storage) == typeid(dafPersist::BoostStorage)) {
        execTrace("PsfFormatter read BoostStorage");
        dafPersist::BoostStorage* boost =
            dynamic_cast<dafPersist::BoostStorage*>(storage.get());
        boost->getIArchive() & ps;
        execTrace("PsfFormatter read end");
        return ps;
    }
    else if (typeid(*storage) == typeid(dafPersist::XmlStorage)) {
        execTrace("PsfFormatter read XmlStorage");
        dafPersist::XmlStorage* xml =
            dynamic_cast<dafPersist::XmlStorage*>(storage.get());
        xml->getIArchive() & make_nvp("psf", ps);
        execTrace("PsfFormatter read end");
        return ps;
    }
    throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeErrorException, "Unrecognized Storage for PSF");
}

void measAlgo::PsfFormatter::update(dafBase::Persistable* ,
                                   dafPersist::Storage::Ptr,
                                   dafBase::PropertySet::Ptr) {
    throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeErrorException, "Unexpected call to update for PSF");
}

/** Serialize a Psf to a Boost archive.  Handles text or XML
 * archives, input or output.
 * \param[in,out] ar Boost archive
 * \param[in] version Version of the Psf class
 * \param[in,out] persistable Pointer to the Psf as a Persistable
 */
template <class Archive>
void measAlgo::PsfFormatter::delegateSerialize(
    Archive& ar, unsigned int const version, dafBase::Persistable* persistable) {
    execTrace("PsfFormatter delegateSerialize start");
    measAlgo::PSF* ps = dynamic_cast<measAlgo::PSF*>(persistable);
    if (ps == 0) {
        throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeErrorException, "Serializing non-PSF");
    }
    ar & make_nvp("width", ps->_width) & make_nvp("height", ps->_height);
    ar & make_nvp("k", ps->_kernel);

    execTrace("PsfFormatter delegateSerialize end");
}

/** Factory method for PsfFormatter.
 * \param[in] policy Policy for configuring the PsfFormatter
 * \return Shared pointer to a new instance
 */
dafPersist::Formatter::Ptr measAlgo::PsfFormatter::createInstance(
    pexPolicy::Policy::Ptr policy) {
    return dafPersist::Formatter::Ptr(new measAlgo::PsfFormatter(policy));
}
