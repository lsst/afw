// -*- lsst-c++ -*-

/** \file
 * \brief Implementation of MaskFormatter class
 *
 * \author $Author: ktlim $
 * \version $Revision: 2151 $
 * \date $Date$
 *
 * Contact: Kian-Tat Lim (ktl@slac.stanford.edu)
 *
 * \ingroup fw
 */

#ifndef __GNUC__
#  define __attribute__(x) /*NOTHING*/
#endif
static char const* SVNid __attribute__((unused)) = "$Id$";

#include "lsst/fw/formatters/MaskFormatter.h"

#include "lsst/mwi/persistence/FormatterImpl.h"
#include "lsst/mwi/data/DataPropertyFormatter.h"

#include "lsst/fw/Mask.h"
#include "lsst/mwi/persistence/LogicalLocation.h"
#include "lsst/mwi/persistence/BoostStorage.h"
#include "lsst/mwi/persistence/FitsStorage.h"
#include "lsst/mwi/utils/Trace.h"

#include <boost/serialization/binary_object.hpp>
#include <boost/serialization/shared_ptr.hpp>
#include <boost/serialization/map.hpp>

// #include "lsst/fw/LSSTFitsResource.h"

#define EXEC_TRACE  20
static void execTrace(std::string s, int level = EXEC_TRACE) {
    lsst::mwi::utils::Trace("fw.MaskFormatter", level, s);
}

using namespace lsst::mwi::persistence;

namespace lsst {
namespace fw {
namespace formatters {

template <typename imagePixelT>
class MaskFormatterTraits {
public:
    static std::string name;
};

template<> std::string MaskFormatterTraits<maskPixelType>::name("Mask");


template <typename MaskPixelT>
FormatterRegistration MaskFormatter<MaskPixelT>::registration(
    MaskFormatterTraits<MaskPixelT>::name,
    typeid(Mask<MaskPixelT>),
    createInstance);

template <typename MaskPixelT>
MaskFormatter<MaskPixelT>::MaskFormatter(
    lsst::mwi::policy::Policy::Ptr policy) :
    Formatter(typeid(*this)) {
}

template <typename MaskPixelT>
MaskFormatter<MaskPixelT>::~MaskFormatter(void) {
}

template <typename MaskPixelT>
void MaskFormatter<MaskPixelT>::write(
    Persistable const* persistable,
    Storage::Ptr storage,
    lsst::mwi::data::DataProperty::PtrType additionalData) {
    execTrace("MaskFormatter write start");
    Mask<MaskPixelT> const* ip =
        dynamic_cast<Mask<MaskPixelT> const*>(persistable);
    if (ip == 0) {
        throw std::runtime_error("Persisting non-Mask");
    }
    if (typeid(*storage) == typeid(BoostStorage)) {
        execTrace("MaskFormatter write BoostStorage");
        BoostStorage* boost = dynamic_cast<BoostStorage*>(storage.get());
        boost->getOArchive() & *ip;
        execTrace("MaskFormatter write end");
        return;
    }
    else if (typeid(*storage) == typeid(FitsStorage)) {
        execTrace("MaskFormatter write FitsStorage");
        FitsStorage* fits = dynamic_cast<FitsStorage*>(storage.get());
        // Need to cast away const because writeFits modifies the metadata.
        Mask<MaskPixelT>* vip = const_cast<Mask<MaskPixelT>*>(ip);
        vip->writeFits(fits->getPath());
        execTrace("MaskFormatter write end");
        return;
    }
    throw std::runtime_error("Unrecognized Storage for Mask");
}

template <typename MaskPixelT>
Persistable* MaskFormatter<MaskPixelT>::read(
    Storage::Ptr storage,
    lsst::mwi::data::DataProperty::PtrType additionalData) {
    execTrace("MaskFormatter read start");
    Mask<MaskPixelT>* ip = new Mask<MaskPixelT>;
    if (typeid(*storage) == typeid(BoostStorage)) {
        execTrace("MaskFormatter read BoostStorage");
        BoostStorage* boost = dynamic_cast<BoostStorage*>(storage.get());
        boost->getIArchive() & *ip;
        execTrace("MaskFormatter read end");
        return ip;
    }
    else if (typeid(*storage) == typeid(FitsStorage)) {
        execTrace("MaskFormatter read FitsStorage");
        FitsStorage* fits = dynamic_cast<FitsStorage*>(storage.get());
        ip->readFits(fits->getPath(), fits->getHdu());
        execTrace("MaskFormatter read end");
        return ip;
    }
    throw std::runtime_error("Unrecognized Storage for Mask");
}

template <typename MaskPixelT>
void MaskFormatter<MaskPixelT>::update(
    Persistable* persistable,
    Storage::Ptr storage,
    lsst::mwi::data::DataProperty::PtrType additionalData) {
    throw std::runtime_error("Unexpected call to update for Mask");
}

template <typename MaskPixelT> template <class Archive>
void MaskFormatter<MaskPixelT>::delegateSerialize(
    Archive& ar, int const version, Persistable* persistable) {
    execTrace("MaskFormatter delegateSerialize start");
    Mask<MaskPixelT>* ip = dynamic_cast<Mask<MaskPixelT>*>(persistable);
    if (ip == 0) {
        throw std::runtime_error("Serializing non-Mask");
    }
    ar & ip->_metaData & ip->_offsetRows & ip->_offsetCols;
    ar & ip->_maskPlaneDict;
    unsigned int cols;
    unsigned int rows;
    unsigned int planes;
    if (Archive::is_saving::value) {
        cols = ip->_vwImagePtr->cols();
        rows = ip->_vwImagePtr->rows();
        planes = ip->_vwImagePtr->planes();
    }
    ar & cols & rows & planes;
    if (Archive::is_loading::value) {
        ip->_vwImagePtr->set_size(cols, rows, planes);
    }
    unsigned int pixels = cols * rows * planes;
    MaskPixelT* data = ip->_vwImagePtr->data();
    ar & boost::serialization::make_binary_object(
        data, pixels * sizeof(MaskPixelT));
    execTrace("MaskFormatter delegateSerialize end");
}

template <typename MaskPixelT>
Formatter::Ptr MaskFormatter<MaskPixelT>::createInstance(
    lsst::mwi::policy::Policy::Ptr policy) {
    return Formatter::Ptr(new MaskFormatter<MaskPixelT>(policy));
}

template class MaskFormatter<maskPixelType>;

}}} // namespace lsst::fw::formatters
