// -*- lsst-c++ -*-

/** \file
 * \brief Implementation of MaskedImageFormatter class
 *
 * \author $Author: ktlim $
 * \version $Revision: 2151 $
 * \date $Date$
 *
 * Contact: Kian-Tat Lim (ktl@slac.stanford.edu)
 *
 * \ingroup afw
 */

#ifndef __GNUC__
#  define __attribute__(x) /*NOTHING*/
#endif
static char const* SVNid __attribute__((unused)) = "$Id$";

#include <lsst/afw/formatters/MaskedImageFormatter.h>
#include <lsst/afw/image/MaskedImage.h>

#include <lsst/pex/persistence/FormatterImpl.h>
#include <lsst/afw/formatters/ImageFormatter.h>
#include <lsst/afw/formatters/MaskFormatter.h>

#include <lsst/pex/persistence/LogicalLocation.h>
#include <lsst/pex/persistence/BoostStorage.h>
#include <lsst/pex/persistence/FitsStorage.h>
#include <lsst/pex/utils/Trace.h>

#include <boost/serialization/shared_ptr.hpp>

// #include <lsst/afw/image/LSSTFitsResource.h>

#define EXEC_TRACE  20
static void execTrace(std::string s, int level = EXEC_TRACE) {
    lsst::pex::utils::Trace("afw.MaskedImageFormatter", level, s);
}

using namespace lsst::pex::persistence;

namespace lsst::afw::formatters;

template <typename ImagePixelT, typename MaskPixelT>
class MaskedImageFormatterTraits {
public:
    static std::string name;
};

template<> std::string MaskedImageFormatterTraits<boost::uint16_t, maskPixelType>::name("MaskedImageU");
template<> std::string MaskedImageFormatterTraits<float, maskPixelType>::name("MaskedImageF");
template<> std::string MaskedImageFormatterTraits<double, maskPixelType>::name("MaskedImageD");

template <typename ImagePixelT, typename MaskPixelT>
FormatterRegistration MaskedImageFormatter<ImagePixelT, MaskPixelT>::registration(
    MaskedImageFormatterTraits<ImagePixelT, MaskPixelT>::name,
    typeid(MaskedImage<ImagePixelT, MaskPixelT>),
    createInstance);

template <typename ImagePixelT, typename MaskPixelT>
MaskedImageFormatter<ImagePixelT, MaskPixelT>::MaskedImageFormatter(
    lsst::pex::policy::Policy::Ptr policy) :
    Formatter(typeid(*this)) {
}

template <typename ImagePixelT, typename MaskPixelT>
MaskedImageFormatter<ImagePixelT, MaskPixelT>::~MaskedImageFormatter(void) {
}

template <typename ImagePixelT, typename MaskPixelT>
void MaskedImageFormatter<ImagePixelT, MaskPixelT>::write(
    Persistable const* persistable,
    Storage::Ptr storage,
    lsst::daf::data::DataProperty::PtrType additionalData) {
    execTrace("MaskedImageFormatter write start");
    MaskedImage<ImagePixelT, MaskPixelT> const* ip =
        dynamic_cast<MaskedImage<ImagePixelT, MaskPixelT> const*>(persistable);
    if (ip == 0) {
        throw std::runtime_error("Persisting non-MaskedImage");
    }
    if (typeid(*storage) == typeid(BoostStorage)) {
        execTrace("MaskedImageFormatter write BoostStorage");
        BoostStorage* boost = dynamic_cast<BoostStorage*>(storage.get());
        boost->getOArchive() & *ip;
        execTrace("MaskedImageFormatter write end");
        return;
    }
    else if (typeid(*storage) == typeid(FitsStorage)) {
        execTrace("MaskedImageFormatter write FitsStorage");
        FitsStorage* fits = dynamic_cast<FitsStorage*>(storage.get());
        ip->writeFits(fits->getPath());
        execTrace("MaskedImageFormatter write end");
        return;
    }
    throw std::runtime_error("Unrecognized Storage for MaskedImage");
}

template <typename ImagePixelT, typename MaskPixelT>
Persistable* MaskedImageFormatter<ImagePixelT, MaskPixelT>::read(
    Storage::Ptr storage,
    lsst::daf::data::DataProperty::PtrType additionalData) {
    execTrace("MaskedImageFormatter read start");
    MaskedImage<ImagePixelT, MaskPixelT>* ip = new MaskedImage<ImagePixelT, MaskPixelT>;
    if (typeid(*storage) == typeid(BoostStorage)) {
        execTrace("MaskedImageFormatter read BoostStorage");
        BoostStorage* boost = dynamic_cast<BoostStorage*>(storage.get());
        boost->getIArchive() & *ip;
        execTrace("MaskedImageFormatter read end");
        return ip;
    }
    else if (typeid(*storage) == typeid(FitsStorage)) {
        execTrace("MaskedImageFormatter read FitsStorage");
        FitsStorage* fits = dynamic_cast<FitsStorage*>(storage.get());
        ip->readFits(fits->getPath());
        execTrace("MaskedImageFormatter read end");
        return ip;
    }
    throw std::runtime_error("Unrecognized Storage for MaskedImage");
}

template <typename ImagePixelT, typename MaskPixelT>
void MaskedImageFormatter<ImagePixelT, MaskPixelT>::update(
    Persistable* persistable,
    Storage::Ptr storage,
    lsst::daf::data::DataProperty::PtrType additionalData) {
    throw std::runtime_error("Unexpected call to update for MaskedImage");
}

template <typename ImagePixelT, typename MaskPixelT> template <class Archive>
void MaskedImageFormatter<ImagePixelT, MaskPixelT>::delegateSerialize(
    Archive& ar, int const version, Persistable* persistable) {
    execTrace("MaskedImageFormatter delegateSerialize start");
    MaskedImage<ImagePixelT, MaskPixelT>* ip = dynamic_cast<MaskedImage<ImagePixelT, MaskPixelT>*>(persistable);
    if (ip == 0) {
        throw std::runtime_error("Serializing non-MaskedImage");
    }
    ar & ip->_imageRows & ip->_imageCols;
    ar & ip->_imagePtr & ip->_variancePtr & ip->_maskPtr;
    execTrace("MaskedImageFormatter delegateSerialize end");
}

template <typename ImagePixelT, typename MaskPixelT>
Formatter::Ptr MaskedImageFormatter<ImagePixelT, MaskPixelT>::createInstance(
    lsst::pex::policy::Policy::Ptr policy) {
    return Formatter::Ptr(
        new MaskedImageFormatter<ImagePixelT, MaskPixelT>(policy));
}

template class MaskedImageFormatter<uint16_t, maskPixelType>;
template class MaskedImageFormatter<float, maskPixelType>;
template class MaskedImageFormatter<double, maskPixelType>;
