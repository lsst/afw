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

#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>

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

namespace afwImg = lsst::afw::image;
namespace afwGeom = lsst::afw::geom;

namespace lsst {
namespace afw {
namespace formatters {

template <typename ImagePixelT>
class ImageFormatterTraits {
public:
    static std::string name();
};

template<> std::string ImageFormatterTraits<boost::uint16_t>::name() {
    static std::string name = "ImageU";
    return name;
}
template<> std::string ImageFormatterTraits<int>::name() {
    static std::string name = "ImageI";
    return name;
}
template<> std::string ImageFormatterTraits<float>::name() {
    static std::string name = "ImageF";
    return name;
}
template<> std::string ImageFormatterTraits<double>::name() {
    static std::string name = "ImageD";
    return name;
}
template<> std::string ImageFormatterTraits<boost::uint64_t>::name() {
    static std::string name = "ImageL";
    return name;
}

template <typename ImagePixelT>
lsst::daf::persistence::FormatterRegistration ImageFormatter<ImagePixelT>::registration(
    ImageFormatterTraits<ImagePixelT>::name(),
    typeid(Image<ImagePixelT>),
    createInstance);

template <typename ImagePixelT>
ImageFormatter<ImagePixelT>::ImageFormatter(
        lsst::pex::policy::Policy::Ptr
                                           )
    :
    lsst::daf::persistence::Formatter(typeid(this))
{
}

template <typename ImagePixelT>
ImageFormatter<ImagePixelT>::~ImageFormatter(void)
{
}

namespace {
namespace dafBase = lsst::daf::base;
namespace afwImage = lsst::afw::image;
}

template <typename ImagePixelT>
void ImageFormatter<ImagePixelT>::write(
    Persistable const* persistable,
    Storage::Ptr storage,
    lsst::daf::base::PropertySet::Ptr) {

    execTrace("ImageFormatter write start");
    Image<ImagePixelT> const* ip = dynamic_cast<Image<ImagePixelT> const*>(persistable);
    if (ip == 0) {
        throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeError, "Persisting non-Image");
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
    throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeError, "Unrecognized Storage for Image");
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
        geom::Box2I box;
        if (additionalData->exists("llcX")) {
            int llcX = additionalData->get<int>("llcX");
            int llcY = additionalData->get<int>("llcY");
            int width = additionalData->get<int>("width");
            int height = additionalData->get<int>("height");
            box = geom::Box2I(
                geom::Point2I(llcX, llcY), 
                geom::Extent2I(width, height)
            );
        }
        afwImg::ImageOrigin origin = afwImg::PARENT;
        if (additionalData->exists("imageOrigin")) {
            std::string originStr = additionalData->get<std::string>("imageOrigin");
            if (originStr == "LOCAL") {
                origin = afwImg::LOCAL;
            } else if (originStr == "PARENT") {
                origin = afwImg::PARENT;
            } else {
                throw LSST_EXCEPT(
                    lsst::pex::exceptions::RuntimeError, 
                    (boost::format("Unknown ImageOrigin type  %s specified in additional"
                                   "data for retrieving Image from fits")%originStr
                        
                    ).str()
                );
            }
        }
        lsst::daf::base::PropertySet::Ptr metadata;

        Image<ImagePixelT>* ip = new Image<ImagePixelT>(
            fits->getPath(), fits->getHdu(),
            lsst::daf::base::PropertySet::Ptr(), 
            box, origin
        );
        // \note We're throwing away the metadata
        // \todo Do something with these fields?
        // int _X0;
        // int _Y0;
        execTrace("ImageFormatter read end");
        return ip;
    }
    throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeError, "Unrecognized Storage for Image");
}

template <typename ImagePixelT>
void ImageFormatter<ImagePixelT>::update(
    Persistable*,
    Storage::Ptr,
    lsst::daf::base::PropertySet::Ptr) {
    throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeError, "Unexpected call to update for Image");
}

template <typename ImagePixelT> template <class Archive>
void ImageFormatter<ImagePixelT>::delegateSerialize(
    Archive& ar, int const, Persistable* persistable) {
    execTrace("ImageFormatter delegateSerialize start");
    Image<ImagePixelT>* ip = dynamic_cast<Image<ImagePixelT>*>(persistable);
    if (ip == 0) {
        throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeError, "Serializing non-Image");
    }
    int width, height;
    if (Archive::is_saving::value) {
        width = ip->getWidth();
        height = ip->getHeight();
    }
    ar & make_nvp("width", width) & make_nvp("height", height);
    if (Archive::is_loading::value) {
        std::unique_ptr<Image<ImagePixelT> > ni(
            new Image<ImagePixelT>(geom::Extent2I(width, height))
        );
        typename Image<ImagePixelT>::Array array = ni->getArray();
        ar & make_nvp("array",
                      boost::serialization::make_array(array.getData(), array.getNumElements()));
        ip->swap(*ni);
    } else {
        ndarray::Array<ImagePixelT, 2, 2> array = ndarray::dynamic_dimension_cast<2>(ip->getArray());
        if(array.empty())
            array = ndarray::copy(ip->getArray());
        ar & make_nvp("array", boost::serialization::make_array(array.getData(), array.getNumElements()));
    }
}

template <typename ImagePixelT>
lsst::daf::persistence::Formatter::Ptr ImageFormatter<ImagePixelT>::createInstance(
    lsst::pex::policy::Policy::Ptr policy) {
    return lsst::daf::persistence::Formatter::Ptr(new ImageFormatter<ImagePixelT>(policy));
}

#define InstantiateFormatter(ImagePixelT) \
    template class ImageFormatter<ImagePixelT >; \
    template void ImageFormatter<ImagePixelT >::delegateSerialize(boost::archive::text_oarchive&, int const, Persistable*); \
    template void ImageFormatter<ImagePixelT >::delegateSerialize(boost::archive::text_iarchive&, int const, Persistable*); \
    template void ImageFormatter<ImagePixelT >::delegateSerialize(boost::archive::xml_oarchive&, int const, Persistable*); \
    template void ImageFormatter<ImagePixelT >::delegateSerialize(boost::archive::xml_iarchive&, int const, Persistable*); \
    template void ImageFormatter<ImagePixelT >::delegateSerialize(boost::archive::binary_oarchive&, int const, Persistable*); \
    template void ImageFormatter<ImagePixelT >::delegateSerialize(boost::archive::binary_iarchive&, int const, Persistable*);

InstantiateFormatter(boost::uint16_t);
InstantiateFormatter(int);
InstantiateFormatter(float);
InstantiateFormatter(double);
InstantiateFormatter(boost::uint64_t);

#undef InstantiateSerializer

}}} // namespace lsst::afw::formatters
