// -*- lsst-c++ -*-

/** \file
 * \brief Implementation of ImageFormatter class
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

#include <boost/serialization/shared_ptr.hpp>
#include <boost/serialization/binary_object.hpp>

#include "lsst/daf/base.h"
#include "lsst/daf/persistence.h"
#include "lsst/daf/persistence/DataPropertyFormatter.h"
#include "lsst/pex/logging/Trace.h"
#include "lsst/afw/formatters/ImageFormatter.h"
#include "lsst/afw/image/Image.h"

// #include "lsst/afw/image/LSSTFitsResource.h"

#define EXEC_TRACE  20
static void execTrace(std::string s, int level = EXEC_TRACE) {
    lsst::pex::logging::Trace("afw.ImageFormatter", level, s);
}

using lsst::daf::base::Persistable;
using lsst::daf::persistence::BoostStorage;
using lsst::daf::persistence::FitsStorage;
using lsst::daf::persistence::Storage;
using lsst::afw::image::Image;

namespace lsst {
namespace afw {
namespace formatters {

template <typename ImagePixelT>
class ImageFormatterTraits {
public:
    static std::string name;
};

template<> std::string ImageFormatterTraits<boost::uint16_t>::name("ImageU");
template<> std::string ImageFormatterTraits<float>::name("ImageF");
template<> std::string ImageFormatterTraits<double>::name("ImageD");


template <typename ImagePixelT>
lsst::daf::persistence::FormatterRegistration ImageFormatter<ImagePixelT>::registration(
    ImageFormatterTraits<ImagePixelT>::name,
    typeid(Image<ImagePixelT>),
    createInstance);

template <typename ImagePixelT>
ImageFormatter<ImagePixelT>::ImageFormatter(
    lsst::pex::policy::Policy::Ptr policy) :
    lsst::daf::persistence::Formatter(typeid(*this)) {
}

template <typename ImagePixelT>
ImageFormatter<ImagePixelT>::~ImageFormatter(void) {
}

template <typename ImagePixelT>
void ImageFormatter<ImagePixelT>::write(
    Persistable const* persistable,
    Storage::Ptr storage,
    lsst::daf::base::DataProperty::PtrType additionalData) {
    execTrace("ImageFormatter write start");
    Image<ImagePixelT> const* ip =
        dynamic_cast<Image<ImagePixelT> const*>(persistable);
    if (ip == 0) {
        throw std::runtime_error("Persisting non-Image");
    }
    if (typeid(*storage) == typeid(BoostStorage)) {
        execTrace("ImageFormatter write BoostStorage");
        BoostStorage* boost = dynamic_cast<BoostStorage*>(storage.get());
        boost->getOArchive() & *ip;
        execTrace("ImageFormatter write end");
        return;
    }
    else if (typeid(*storage) == typeid(FitsStorage)) {
        execTrace("ImageFormatter write FitsStorage");
        FitsStorage* fits = dynamic_cast<FitsStorage*>(storage.get());
        ip->writeFits(fits->getPath());
        // LSSTFitsResource<ImagePixelT> fitsRes;
        // fitsRes.writeFits(*(ip->_vwImagePtr), ip->_metaData, fits->getPath());
        // \todo Do something with these fields?
        // unsigned int _offsetRows;
        // unsigned int _offsetCols;
        execTrace("ImageFormatter write end");
        return;
    }
    throw std::runtime_error("Unrecognized Storage for Image");
}

template <typename ImagePixelT>
Persistable* ImageFormatter<ImagePixelT>::read(
    Storage::Ptr storage,
    lsst::daf::base::DataProperty::PtrType additionalData) {
    execTrace("ImageFormatter read start");
    Image<ImagePixelT>* ip = new Image<ImagePixelT>;
    if (typeid(*storage) == typeid(BoostStorage)) {
        execTrace("ImageFormatter read BoostStorage");
        BoostStorage* boost = dynamic_cast<BoostStorage*>(storage.get());
        boost->getIArchive() & *ip;
        execTrace("ImageFormatter read end");
        return ip;
    }
    else if (typeid(*storage) == typeid(FitsStorage)) {
        execTrace("ImageFormatter read FitsStorage");
        FitsStorage* fits = dynamic_cast<FitsStorage*>(storage.get());
        ip->readFits(fits->getPath(), fits->getHdu());
        // LSSTFitsResource<ImagePixelT> fitsRes;
        // fitsRes.readFits(fits->getPath(), *(ip->_vwImagePtr), ip->_metaData, fits->getHdu());
        // \todo Do something with these fields?
        // unsigned int _offsetRows;
        // unsigned int _offsetCols;
        execTrace("ImageFormatter read end");
        return ip;
    }
    throw std::runtime_error("Unrecognized Storage for Image");
}

template <typename ImagePixelT>
void ImageFormatter<ImagePixelT>::update(
    Persistable* persistable,
    Storage::Ptr storage,
    lsst::daf::base::DataProperty::PtrType additionalData) {
    throw std::runtime_error("Unexpected call to update for Image");
}

template <typename ImagePixelT> template <class Archive>
void ImageFormatter<ImagePixelT>::delegateSerialize(
    Archive& ar, int const version, Persistable* persistable) {
    execTrace("ImageFormatter delegateSerialize start");
    Image<ImagePixelT>* ip = dynamic_cast<Image<ImagePixelT>*>(persistable);
    if (ip == 0) {
        throw std::runtime_error("Serializing non-Image");
    }
    ar & ip->_metaData & ip->_offsetRows & ip->_offsetCols;
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
    ImagePixelT* data = ip->_vwImagePtr->data();
    ar & boost::serialization::make_binary_object(
        data, pixels * sizeof(ImagePixelT));
    execTrace("ImageFormatter delegateSerialize end");
}

template <typename ImagePixelT>
lsst::daf::persistence::Formatter::Ptr ImageFormatter<ImagePixelT>::createInstance(
    lsst::pex::policy::Policy::Ptr policy) {
    return lsst::daf::persistence::Formatter::Ptr(new ImageFormatter<ImagePixelT>(policy));
}

template class ImageFormatter<boost::uint16_t>;
template class ImageFormatter<float>;
template class ImageFormatter<double>;

}}} // namespace lsst::afw::formatters
