// -*- lsst-c++ -*-

/** @file
 * @brief Implementation of ImageFormatter class
 *
 * @author $Author: ktlim $
 * @version $Revision: 2151 $
 * @date $Date$
 *
 * Contact: Kian-Tat Lim (ktl@slac.stanford.edu)
 *
 * @ingroup afw
 */

#ifndef __GNUC__
#  define __attribute__(x) /*NOTHING*/
#endif
static char const* SVNid __attribute__((unused)) = "$Id$";

#include "boost/scoped_ptr.hpp"
#include "boost/serialization/shared_ptr.hpp"
#include "boost/serialization/binary_object.hpp"
#include "boost/serialization/nvp.hpp"

#include "lsst/daf/base.h"
#include "lsst/daf/persistence.h"
#include "lsst/pex/logging/Trace.h"
#include "lsst/afw/formatters/ImageFormatter.h"
#include "lsst/afw/image/Image.h"


#define EXEC_TRACE  20
static void execTrace(std::string s, int level = EXEC_TRACE) {
    lsst::pex::logging::Trace("afw.ImageFormatter", level, s);
}

using boost::serialization::make_nvp;
using lsst::daf::base::Persistable;
using lsst::daf::persistence::BoostStorage;
using lsst::daf::persistence::XmlStorage;
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
template<> std::string ImageFormatterTraits<int>::name("ImageI");
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
    lsst::daf::base::PropertySet::Ptr additionalData) {
    execTrace("ImageFormatter write start");
    Image<ImagePixelT> const* ip = dynamic_cast<Image<ImagePixelT> const*>(persistable);
    if (ip == 0) {
        throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeErrorException, "Persisting non-Image");
    }
    if (typeid(*storage) == typeid(BoostStorage)) {
        execTrace("ImageFormatter write BoostStorage");
        BoostStorage* boost = dynamic_cast<BoostStorage*>(storage.get());
        boost->getOArchive() & *ip;
        execTrace("ImageFormatter write end");
        return;
    }
    else if (typeid(*storage) == typeid(XmlStorage)) {
        execTrace("ImageFormatter write XmlStorage");
        XmlStorage* boost = dynamic_cast<XmlStorage*>(storage.get());
        boost->getOArchive() & make_nvp("img", *ip);
        execTrace("ImageFormatter write end");
        return;
    }
    else if (typeid(*storage) == typeid(FitsStorage)) {
        execTrace("ImageFormatter write FitsStorage");
        FitsStorage* fits = dynamic_cast<FitsStorage*>(storage.get());
        typedef Image<ImagePixelT> Image;

        ip->writeFits(fits->getPath());
        // \todo Do something with these fields?
        // int _X0;
        // int _Y0;
        execTrace("ImageFormatter write end");
        return;
    }
    throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeErrorException, "Unrecognized Storage for Image");
}

template <typename ImagePixelT>
Persistable* ImageFormatter<ImagePixelT>::read(Storage::Ptr storage,
                                               lsst::daf::base::PropertySet::Ptr additionalData) {
    execTrace("ImageFormatter read start");
    if (typeid(*storage) == typeid(BoostStorage)) {
        execTrace("ImageFormatter read BoostStorage");
        BoostStorage* boost = dynamic_cast<BoostStorage*>(storage.get());
        Image<ImagePixelT>* ip = new Image<ImagePixelT>;
        boost->getIArchive() & *ip;
        execTrace("ImageFormatter read end");
        return ip;
    }
    else if (typeid(*storage) == typeid(XmlStorage)) {
        execTrace("ImageFormatter read XmlStorage");
        XmlStorage* boost = dynamic_cast<XmlStorage*>(storage.get());
        Image<ImagePixelT>* ip = new Image<ImagePixelT>;
        boost->getIArchive() & make_nvp("img", *ip);
        execTrace("ImageFormatter read end");
        return ip;
    }
    else if(typeid(*storage) == typeid(FitsStorage)) {

        execTrace("ImageFormatter read FitsStorage");
        FitsStorage* fits = dynamic_cast<FitsStorage*>(storage.get());
        
        Image<ImagePixelT>* ip = new Image<ImagePixelT>(fits->getPath(), fits->getHdu());
        // \note We're throwing away the metadata
        // \todo Do something with these fields?
        // int _X0;
        // int _Y0;
        execTrace("ImageFormatter read end");
        return ip;
    }
    throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeErrorException, "Unrecognized Storage for Image");
}

template <typename ImagePixelT>
void ImageFormatter<ImagePixelT>::update(
    Persistable* persistable,
    Storage::Ptr storage,
    lsst::daf::base::PropertySet::Ptr additionalData) {
    throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeErrorException, "Unexpected call to update for Image");
}

template <typename ImagePixelT> template <class Archive>
void ImageFormatter<ImagePixelT>::delegateSerialize(
    Archive& ar, int const version, Persistable* persistable) {
    execTrace("ImageFormatter delegateSerialize start");
    Image<ImagePixelT>* ip = dynamic_cast<Image<ImagePixelT>*>(persistable);
    if (ip == 0) {
        throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeErrorException, "Serializing non-Image");
    }
    int width, height;
    if (Archive::is_saving::value) {
        width = ip->getWidth();
        height = ip->getHeight();
    }
    ar & make_nvp("width", width) & make_nvp("height", height);
    std::size_t nbytes = width * height * sizeof(ImagePixelT);
    if (Archive::is_loading::value) {
        boost::scoped_ptr<Image<ImagePixelT> > ni(new Image<ImagePixelT>(width, height));
        ImagePixelT * raw = boost::gil::interleaved_view_get_raw_data(view(*ni->_getRawImagePtr()));
        ar & make_nvp("bytes",
                      boost::serialization::make_binary_object(raw, nbytes));
        ip->swap(*ni);
    } else if (width == ip->_getRawImagePtr()->width() && height == ip->_getRawImagePtr()->height()) {
        ImagePixelT * raw = boost::gil::interleaved_view_get_raw_data(view(*ip->_getRawImagePtr()));
        ar & make_nvp("bytes",
                      boost::serialization::make_binary_object(raw, nbytes));
    } else {
        typename Image<ImagePixelT>::_image_t img(width, height);
        boost::gil::copy_pixels(ip->_getRawView(), flipped_up_down_view(view(img)));
        ar & make_nvp("bytes",
                      boost::serialization::make_binary_object(boost::gil::interleaved_view_get_raw_data(view(img)), nbytes));
    }
}

template <typename ImagePixelT>
lsst::daf::persistence::Formatter::Ptr ImageFormatter<ImagePixelT>::createInstance(
    lsst::pex::policy::Policy::Ptr policy) {
    return lsst::daf::persistence::Formatter::Ptr(new ImageFormatter<ImagePixelT>(policy));
}

template class ImageFormatter<boost::uint16_t>;
template class ImageFormatter<int>;
template class ImageFormatter<float>;
template class ImageFormatter<double>;

}}} // namespace lsst::afw::formatters
